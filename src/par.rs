//! Fixed-chunk execution helpers.
//!
//! Every reduction in this crate is evaluated as a sum over *fixed-size* chunks
//! combined in index order. The serial and the `parallel` paths use the same chunking
//! and fold the per-chunk results in the same order, so results are bit-for-bit identical
//! for any thread count.
//!
//! With the `parallel` feature, work is run by a small persistent [`team`] of threads that
//! *spin briefly between jobs* instead of sleeping. Solvers issue many short parallel regions
//! (a dot product, an SpMV, one Gram-Schmidt pass), and waking a sleeping pool thread costs
//! 50-150 µs on a typical VM, comparable to the work of the region itself. A hot team
//! dispatches in about a microsecond and goes to sleep after a short idle period.

/// Elements per chunk for vector kernels (64 KiB of `f64`, an L2-friendly size).
pub const CHUNK: usize = 8192;

/// Below this length the parallel path is not worth its scheduling overhead.
#[cfg(feature = "parallel")]
const PAR_MIN: usize = 32 * 1024;

#[cfg(feature = "parallel")]
pub use team::{set_default_threads, threads, with_threads};

/// Number of threads parallel regions would use (1 without the `parallel` feature).
#[cfg(not(feature = "parallel"))]
pub fn threads() -> usize {
    1
}

#[inline]
pub fn parallel_enabled(n: usize) -> bool {
    #[cfg(feature = "parallel")]
    {
        n >= PAR_MIN && team::threads() > 1
    }
    #[cfg(not(feature = "parallel"))]
    {
        let _ = n;
        false
    }
}

/// Runs `f(i, t)` on each of `t` threads (`i` in `0..t`), returning when all are done. Falls
/// back to `f(0, 1)` on the calling thread when no team is available.
#[inline]
fn run_team(f: &(dyn Fn(usize, usize) + Sync)) {
    #[cfg(feature = "parallel")]
    {
        team::run(f);
    }
    #[cfg(not(feature = "parallel"))]
    {
        f(0, 1);
    }
}

/// Calls `f(chunk_index, start, end)` once for every chunk of `0..n`.
///
/// In the parallel path each thread takes a *fixed contiguous block* of chunks (by thread
/// index). Repeated sweeps over the same data (the basis kernels) therefore reach the same
/// rows from the same thread, as with MPI ranks. Chunk results are written to per-chunk slots
/// by the caller, so values do not depend on the assignment.
#[inline]
pub fn static_chunks(n: usize, chunk: usize, f: impl Fn(usize, usize, usize) + Sync) {
    let nchunks = n.div_ceil(chunk);
    if parallel_enabled(n) {
        run_team(&|i, t| {
            for c in nchunks * i / t..nchunks * (i + 1) / t {
                f(c, c * chunk, ((c + 1) * chunk).min(n));
            }
        });
        return;
    }
    for c in 0..nchunks {
        f(c, c * chunk, ((c + 1) * chunk).min(n));
    }
}

/// Calls `f(i)` for every `i` in `0..count`, each thread taking a fixed contiguous block.
#[inline]
pub fn static_items(count: usize, parallel: bool, f: impl Fn(usize) + Sync) {
    if parallel {
        run_team(&|i, t| {
            for c in count * i / t..count * (i + 1) / t {
                f(c);
            }
        });
        return;
    }
    for c in 0..count {
        f(c);
    }
}

/// Sum of `f(start, end)` over the chunks of `0..n`, accumulated in chunk order.
#[inline]
pub fn sum_chunks(n: usize, chunk: usize, f: impl Fn(usize, usize) -> f64 + Sync) -> f64 {
    if parallel_enabled(n) {
        let parts = chunk_slots(n, chunk, 1, |s, e, slot| slot[0] = f(s, e));
        return parts.iter().sum();
    }
    let mut acc = 0.0;
    let mut s = 0;
    while s < n {
        let e = (s + chunk).min(n);
        acc += f(s, e);
        s = e;
    }
    acc
}

/// Like [`sum_chunks`] for `N` simultaneous scalar reductions.
#[inline]
pub fn sum_chunks_n<const N: usize>(
    n: usize,
    chunk: usize,
    f: impl Fn(usize, usize) -> [f64; N] + Sync,
) -> [f64; N] {
    let mut acc = [0.0; N];
    if parallel_enabled(n) {
        let parts = chunk_slots(n, chunk, N, |s, e, slot| slot.copy_from_slice(&f(s, e)));
        for p in parts.chunks_exact(N) {
            for i in 0..N {
                acc[i] += p[i];
            }
        }
        return acc;
    }
    let mut s = 0;
    while s < n {
        let e = (s + chunk).min(n);
        let p = f(s, e);
        for i in 0..N {
            acc[i] += p[i];
        }
        s = e;
    }
    acc
}

/// A mutable slice that can be shared across chunk closures which each touch a *disjoint*
/// range. This is what lets fused kernels update several vectors and return reductions
/// in a single pass over [`sum_chunks_n`].
pub struct SharedMut {
    ptr: *mut f64,
    len: usize,
}

// SAFETY: access goes through `range`, whose contract requires disjoint ranges per thread.
unsafe impl Send for SharedMut {}
unsafe impl Sync for SharedMut {}

impl SharedMut {
    pub fn new(s: &mut [f64]) -> Self {
        Self { ptr: s.as_mut_ptr(), len: s.len() }
    }

    /// # Safety
    /// No two live slices obtained from the same `SharedMut` may overlap, and the original
    /// slice must not be used while they are alive.
    #[inline(always)]
    pub unsafe fn range(&self, start: usize, end: usize) -> &mut [f64] {
        assert!(start <= end && end <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(start), end - start) }
    }
}

/// Per-chunk results of `f(start, end, slot)` (`slot` is a zeroed `stride`-long slice), laid
/// out chunk after chunk in one buffer, for the caller to fold in chunk order.
#[inline]
pub fn chunk_slots(n: usize, chunk: usize, stride: usize, f: impl Fn(usize, usize, &mut [f64]) + Sync) -> Vec<f64> {
    let nchunks = n.div_ceil(chunk);
    let mut buf = vec![0.0; nchunks * stride];
    let shared = SharedMut::new(&mut buf);
    static_chunks(n, chunk, |c, s, e| {
        // SAFETY: each chunk index owns the disjoint slot `c*stride..(c+1)*stride`.
        let slot = unsafe { shared.range(c * stride, (c + 1) * stride) };
        f(s, e, slot);
    });
    buf
}

/// Apply `f(start, y_chunk)` to disjoint chunks of `y`.
#[inline]
pub fn for_chunks_mut(y: &mut [f64], chunk: usize, f: impl Fn(usize, &mut [f64]) + Sync) {
    let n = y.len();
    if parallel_enabled(n) {
        let ys = SharedMut::new(y);
        static_chunks(n, chunk, |_, s, e| {
            // SAFETY: chunks are disjoint.
            f(s, unsafe { ys.range(s, e) });
        });
        return;
    }
    for (c, yc) in y.chunks_mut(chunk).enumerate() {
        f(c * chunk, yc);
    }
}

/// Apply `f(start, a_chunk, b_chunk)` to matching disjoint chunks of two slices.
#[inline]
pub fn for_chunks2_mut(
    a: &mut [f64],
    b: &mut [f64],
    chunk: usize,
    f: impl Fn(usize, &mut [f64], &mut [f64]) + Sync,
) {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    if parallel_enabled(n) {
        let (a, b) = (SharedMut::new(a), SharedMut::new(b));
        static_chunks(n, chunk, |_, s, e| {
            // SAFETY: chunks are disjoint.
            f(s, unsafe { a.range(s, e) }, unsafe { b.range(s, e) });
        });
        return;
    }
    for (c, (ac, bc)) in a.chunks_mut(chunk).zip(b.chunks_mut(chunk)).enumerate() {
        f(c * chunk, ac, bc);
    }
}

#[cfg(feature = "parallel")]
mod team {
    //! A persistent team of spinning-then-parking worker threads.
    use std::cell::{Cell, UnsafeCell};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering::*};
    use std::sync::{Condvar, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    /// How long an idle worker keeps polling for the next job before it parks.
    const SPIN: Duration = Duration::from_micros(400);

    type Job = &'static (dyn Fn(usize, usize) + Sync);

    struct Shared {
        epoch: AtomicUsize,
        job: UnsafeCell<Option<Job>>,
        done: AtomicUsize,
        panicked: AtomicBool,
        sleepers: AtomicUsize,
        lock: Mutex<()>,
        cv: Condvar,
        busy: AtomicBool,
    }
    // SAFETY: `job` is written only by the single caller holding `busy`, before the release
    // increment of `epoch`; workers read it only after observing that increment.
    unsafe impl Sync for Shared {}

    pub struct Team {
        size: usize,
        shared: &'static Shared,
    }

    impl Team {
        fn new(size: usize) -> &'static Team {
            let shared: &'static Shared = Box::leak(Box::new(Shared {
                epoch: AtomicUsize::new(0),
                job: UnsafeCell::new(None),
                done: AtomicUsize::new(0),
                panicked: AtomicBool::new(false),
                sleepers: AtomicUsize::new(0),
                lock: Mutex::new(()),
                cv: Condvar::new(),
                busy: AtomicBool::new(false),
            }));
            for idx in 1..size {
                std::thread::Builder::new()
                    .name(format!("rsolvers-{idx}"))
                    .spawn(move || worker(shared, idx, size))
                    .expect("failed to spawn worker thread");
            }
            Box::leak(Box::new(Team { size, shared }))
        }

        fn run(&self, f: &(dyn Fn(usize, usize) + Sync)) {
            let sh = self.shared;
            if sh.busy.swap(true, Acquire) {
                // Nested or concurrent use of this team: do the work on this thread.
                f(0, 1);
                return;
            }
            // SAFETY: the reference is only used until every worker reports completion below,
            // which happens before this function returns.
            let job: Job = unsafe { std::mem::transmute::<&(dyn Fn(usize, usize) + Sync), Job>(f) };
            sh.done.store(0, Relaxed);
            sh.panicked.store(false, Relaxed);
            unsafe { *sh.job.get() = Some(job) };
            sh.epoch.fetch_add(1, SeqCst);
            if sh.sleepers.load(SeqCst) > 0 {
                let _g = sh.lock.lock().unwrap();
                sh.cv.notify_all();
            }
            let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(0, self.size)));
            while sh.done.load(Acquire) != self.size - 1 {
                std::hint::spin_loop();
            }
            unsafe { *sh.job.get() = None };
            let worker_panicked = sh.panicked.load(Relaxed);
            sh.busy.store(false, Release);
            if let Err(p) = mine {
                std::panic::resume_unwind(p);
            }
            assert!(!worker_panicked, "a worker thread panicked inside a parallel region");
        }
    }

    fn worker(sh: &'static Shared, idx: usize, size: usize) {
        let mut seen = 0usize;
        loop {
            let started = Instant::now();
            let mut spins = 0u32;
            while sh.epoch.load(Acquire) == seen {
                std::hint::spin_loop();
                spins += 1;
                if spins % 256 == 0 && started.elapsed() > SPIN {
                    sh.sleepers.fetch_add(1, SeqCst);
                    let mut g = sh.lock.lock().unwrap();
                    while sh.epoch.load(SeqCst) == seen {
                        g = sh.cv.wait(g).unwrap();
                    }
                    drop(g);
                    sh.sleepers.fetch_sub(1, SeqCst);
                }
            }
            seen = sh.epoch.load(Acquire);
            // SAFETY: published before the epoch increment we just observed.
            let job = unsafe { (*sh.job.get()).expect("job missing") };
            if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| job(idx, size))).is_err() {
                sh.panicked.store(true, Relaxed);
            }
            sh.done.fetch_add(1, Release);
        }
    }

    static DEFAULT: AtomicUsize = AtomicUsize::new(0);
    static TEAMS: Mutex<Vec<&'static Team>> = Mutex::new(Vec::new());
    thread_local! {
        static ACTIVE: Cell<usize> = const { Cell::new(0) };
    }

    fn default_threads() -> usize {
        static ENV: OnceLock<usize> = OnceLock::new();
        let d = DEFAULT.load(Relaxed);
        if d != 0 {
            return d;
        }
        *ENV.get_or_init(|| {
            std::env::var("RSOLVERS_THREADS")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&n| n >= 1)
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        })
    }

    /// Number of threads parallel regions on this thread currently use.
    pub fn threads() -> usize {
        match ACTIVE.with(Cell::get) {
            0 => default_threads(),
            n => n,
        }
    }

    /// Sets the thread count used when no [`with_threads`] scope is active (default: the
    /// `RSOLVERS_THREADS` environment variable, else the number of available cores).
    pub fn set_default_threads(n: usize) {
        DEFAULT.store(n.max(1), Relaxed);
    }

    /// Runs `f` with parallel regions using `n` threads (the calling thread plus `n - 1`
    /// workers). Results do not depend on `n`.
    pub fn with_threads<T>(n: usize, f: impl FnOnce() -> T) -> T {
        struct Restore(usize);
        impl Drop for Restore {
            fn drop(&mut self) {
                ACTIVE.with(|a| a.set(self.0));
            }
        }
        let _restore = Restore(ACTIVE.with(|a| a.replace(n.max(1))));
        f()
    }

    fn team_for(n: usize) -> &'static Team {
        let mut teams = TEAMS.lock().unwrap();
        if let Some(t) = teams.iter().find(|t| t.size == n) {
            return t;
        }
        let t = Team::new(n);
        teams.push(t);
        t
    }

    pub fn run(f: &(dyn Fn(usize, usize) + Sync)) {
        let n = threads();
        if n <= 1 {
            f(0, 1);
        } else {
            team_for(n).run(f);
        }
    }
}

#[cfg(all(test, feature = "parallel"))]
mod tests {
    use super::*;

    #[test]
    fn chunk_sums_do_not_depend_on_thread_count() {
        let n = 200_000;
        let x: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1013) as f64 * 1e-3 - 0.5).collect();
        let sum = |t| with_threads(t, || sum_chunks(n, CHUNK, |s, e| x[s..e].iter().sum::<f64>()));
        let reference = sum(1);
        for t in [2, 3, 4, 7] {
            assert_eq!(sum(t).to_bits(), reference.to_bits(), "threads {t}");
        }
    }

    #[test]
    fn every_chunk_is_visited_once() {
        let n = 100_000;
        for t in [1, 3, 4] {
            let mut v = vec![0.0; n];
            with_threads(t, || for_chunks_mut(&mut v, 1000, |s, c| c.iter_mut().for_each(|x| *x += s as f64 + 1.0)));
            assert!(v.iter().enumerate().all(|(i, &x)| x == (i / 1000 * 1000) as f64 + 1.0));
        }
    }

    #[test]
    fn nested_regions_fall_back_to_serial() {
        let n = 100_000;
        let total = with_threads(4, || {
            sum_chunks(n, CHUNK, |s, e| sum_chunks(e - s, 1024, |a, b| (b - a) as f64))
        });
        assert_eq!(total, n as f64);
    }

    #[test]
    fn team_survives_idle_periods() {
        with_threads(3, || {
            for _ in 0..3 {
                let t = sum_chunks(100_000, CHUNK, |s, e| (e - s) as f64);
                assert_eq!(t, 100_000.0);
                std::thread::sleep(std::time::Duration::from_millis(5)); // workers park, then wake
            }
        });
    }
}
