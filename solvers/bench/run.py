#!/usr/bin/env python3
"""Run the Rust and PETSc/SLEPc benchmarks on identical problem files and report.

  python3 bench/run.py run   --suite quick|full --out bench/results/<name>.jsonl
  python3 bench/run.py report bench/results/<name>.jsonl > bench/results/<name>.md

Variants (each optional; skipped when its binary is missing):
  rust          target/release/rsolvers-bench          (portable x86-64 build)
  rust-native   target-native/release/rsolvers-bench   (RUSTFLAGS=-C target-cpu=native)
  petsc         bench/c/distro/*                       (system PETSc/SLEPc packages)
  petsc-native  bench/c/native/*                       (PETSc/SLEPc built from source with -march=native;
                                                        set PETSC_NATIVE_LIB to its lib dirs)
"""
import argparse, json, os, subprocess, sys, shutil, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "bench", "data")
RUST = os.path.join(ROOT, "target", "release", "rsolvers-bench")
RUST_NATIVE = os.environ.get("RUST_NATIVE_BIN", "/tmp/claude/target-native/release/rsolvers-bench")
C = os.path.join(ROOT, "bench", "c")

PROBLEMS = {
    "p2d_256": "poisson2d:256x256",
    "p2d_1024": "poisson2d:1024x1024",
    "p3d_100": "poisson3d:100",
    "cd2d_512": "convdiff2d:512x512:5:3",
    "spd_rand": "randspd:200000:12",
    "p2d_256x250": "poisson2d:256x250",
    "p2d_48x40": "poisson2d:48x40",
}

# (kind, problem, solver/which, pc, extra args)
def ksp(problem, solver, pc, **kw):
    return dict(kind="ksp", problem=problem, solver=solver, pc=pc, **kw)

def eps(problem, which, nev, **kw):
    return dict(kind="eps", problem=problem, which=which, nev=nev, **kw)

SUITES = {
    # Cases where the first full run showed Rust behind, plus a few wins as controls.
    "recheck": (
        [ksp("cd2d_512", "gmres", pc) for pc in ("none", "sor", "ilu")]
        + [ksp("p2d_256", "gmres", pc) for pc in ("none", "sor", "ilu")]
        + [ksp("spd_rand", "cg", "icc"), ksp("p3d_100", "cg", "sor", reps=2)]
        + [eps("p2d_256x250", "largest", 4, reps=2), eps("p2d_256x250", "largest", 10, reps=2),
           eps("spd_rand", "largest", 4, reps=2), eps("spd_rand", "largest", 10, reps=2),
           eps("p2d_48x40", "smallest", 4, tol=1e-6, maxit=2000)]
        + [dict(c, par=4) for c in (
            ksp("p3d_100", "cg", "jacobi"), ksp("cd2d_512", "gmres", "none"), ksp("cd2d_512", "gmres", "jacobi"),
            ksp("p2d_256", "gmres", "ilu"),
            eps("p2d_256x250", "largest", 10, reps=2), eps("spd_rand", "largest", 10, reps=2))]
        + [dict(kind="micro", problem=p) for p in ("p2d_256", "spd_rand")]
    ),
    "quick": [
        ksp("p2d_256", "cg", "none"), ksp("p2d_256", "cg", "jacobi"), ksp("p2d_256", "cg", "icc"),
        ksp("cd2d_512", "gmres", "ilu"),
        eps("p2d_256x250", "largest", 4),
    ],
    "full": (
        [ksp(p, "cg", pc) for p in ("p2d_256", "p2d_1024", "p3d_100", "spd_rand") for pc in ("none", "jacobi", "sor", "icc")]
        + [ksp(p, "gmres", pc) for p in ("cd2d_512", "p2d_256") for pc in ("none", "jacobi", "sor", "ilu")]
        + [eps("p2d_256x250", "largest", 4), eps("p2d_256x250", "largest", 10, reps=2),
           eps("spd_rand", "largest", 4, reps=2), eps("spd_rand", "largest", 10, reps=2), eps("p2d_48x40", "smallest", 4, tol=1e-6, maxit=2000)]
        + [dict(c, par=4) for c in (
            ksp("p2d_1024", "cg", "none"), ksp("p2d_1024", "cg", "jacobi"), ksp("p3d_100", "cg", "jacobi"),
            ksp("cd2d_512", "gmres", "none"), ksp("cd2d_512", "gmres", "jacobi"),
            eps("p2d_256x250", "largest", 10, reps=2), eps("spd_rand", "largest", 10, reps=2))]
        + [dict(kind="micro", problem=p) for p in ("p2d_256", "p2d_1024", "p3d_100", "spd_rand")]
    ),
}

def ensure_problem(name):
    path = os.path.join(DATA, name + ".bin")
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        subprocess.run([RUST, "gen", path, PROBLEMS[name]], check=True)
    return path

def natlib():
    return os.environ.get("PETSC_NATIVE_LIB", "")

def variants(par):
    v = []
    if os.path.exists(RUST): v.append("rust")
    if os.path.exists(RUST_NATIVE): v.append("rust-native")
    if os.path.exists(os.path.join(C, "distro", "petsc_ksp")): v.append("petsc")
    if os.path.exists(os.path.join(C, "native", "petsc_ksp")) and natlib(): v.append("petsc-native")
    return v

def command(variant, case, path, par):
    threads = par or 1
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", RSOLVERS_THREADS=str(threads))
    kind = case["kind"]
    if variant.startswith("rust"):
        exe = RUST if variant == "rust" else RUST_NATIVE
        cmd = [exe, kind, path, "--threads", str(threads)]
        if kind == "ksp":
            cmd += ["--solver", case["solver"], "--pc", case["pc"], "--reps", str(case.get("reps", 3))]
        elif kind == "eps":
            cmd += ["--which", case["which"], "--nev", str(case["nev"]), "--reps", str(case.get("reps", 3)),
                    "--tol", str(case.get("tol", 1e-8))]
            if "maxit" in case: cmd += ["--maxit", str(case["maxit"])]
        return cmd, env
    d = "distro" if variant == "petsc" else "native"
    if variant == "petsc-native":
        env["LD_LIBRARY_PATH"] = natlib() + ":" + env.get("LD_LIBRARY_PATH", "")
    tool = {"ksp": "petsc_ksp", "eps": "slepc_eps", "micro": "petsc_micro"}[kind]
    cmd = [os.path.join(C, d, tool), path]
    if par:
        cmd = ["mpiexec", "--allow-run-as-root", "--oversubscribe", "--bind-to", "core", "-n", str(par)] + cmd
    if kind == "ksp":
        cmd += ["-solver", case["solver"], "-pc", case["pc"], "-reps", str(case.get("reps", 3))]
    elif kind == "eps":
        cmd += ["-which", case["which"], "-nev", str(case["nev"]), "-reps", str(case.get("reps", 3)),
                "-tol", str(case.get("tol", 1e-8))]
        if "maxit" in case: cmd += ["-maxit", str(case["maxit"])]
    else:
        cmd += ["-reps", "100"]
    return cmd, env

def run(args):
    cases = SUITES[args.suite]
    out = open(args.out, "a")
    for case in cases:
        par = case.get("par", 0)
        path = ensure_problem(case["problem"])
        for variant in (args.variants.split(",") if args.variants else variants(par)):
            if case["kind"] == "micro" and False:
                continue
            cmd, env = command(variant, case, path, par)
            t0 = time.time()
            try:
                res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT {variant} {case}", file=sys.stderr); continue
            lines = [l for l in res.stdout.splitlines() if l.startswith("{")]
            if res.returncode != 0 or not lines:
                print(f"FAILED {variant} {case}\n{res.stderr[-400:]}", file=sys.stderr); continue
            rec = json.loads(lines[-1])
            rec.update(variant=variant, problem=case["problem"], parallel=par or 1, wall=time.time() - t0)
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(f"{variant:13s} {case['problem']:14s} {json.dumps({k: v for k, v in case.items() if k not in ('problem',)})} "
                  f"-> {rec.get('solve_min_s', rec.get('spmv_s')):.4g}s", flush=True)

def report(args):
    recs = [json.loads(l) for l in open(args.file) if l.strip()]
    order = ["petsc", "petsc-native", "rust", "rust-native"]
    groups = {}
    for r in recs:
        if r["kind"] == "micro": key = ("micro", r["problem"], r["parallel"], "")
        elif r["kind"] == "ksp": key = ("ksp", r["problem"], r["parallel"], f'{r["solver"]}+{r["pc"]}')
        else: key = ("eps", r["problem"], r["parallel"], f'{r["which"]} nev={r["nev"]}')
        groups.setdefault(key, {})[r["variant"]] = r
    present = [v for v in order if any(v in g for g in groups.values())]
    def sec(x): return f"{x*1e3:.1f} ms" if x < 1 else f"{x:.2f} s"
    for kind, title in (("ksp", "Linear solves (KSP): min wall time of repeated solves"), ("eps", "Eigensolves (EPS Krylov–Schur): min wall time"), ("micro", "Kernels (best of 100)")):
        rows = [(k, g) for k, g in groups.items() if k[0] == kind]
        if not rows: continue
        print(f"\n### {title}\n")
        if kind == "micro":
            print("| problem | n | nnz | kernel | " + " | ".join(present) + " |")
            print("|---|---|---|---|" + "---|" * len(present))
            for k, g in rows:
                base = next(iter(g.values()))
                kernels = [kk for kk in ("spmv_s", "dot_s", "axpy_s", "pc_jacobi_s", "pc_sor_s", "pc_ilu_s", "pc_icc_s", "mdot16_s", "maxpy16_s")
                           if any(kk in g[v] for v in g)]
                for kern in kernels:
                    cells = [sec(g[v][kern]) if v in g and kern in g[v] else "–" for v in present]
                    print(f"| {k[1]} | {base['n']} | {base['nnz']} | {kern[:-2]} | " + " | ".join(cells) + " |")
            continue
        hdr = "| problem | cores | case | iters / restarts | " + " | ".join(present) + " | best Rust vs best PETSc |"
        print(hdr); print("|---|---|---|---|" + "---|" * len(present) + "---|")
        for k, g in rows:
            its = {v: g[v].get("iterations") for v in g}
            itxt = "/".join(sorted({str(i) for i in its.values()}, key=lambda s: int(s))) if its else ""
            cells = [sec(g[v]["solve_min_s"]) if v in g else "–" for v in present]
            pet = [g[v]["solve_min_s"] for v in ("petsc", "petsc-native") if v in g]
            rus = [g[v]["solve_min_s"] for v in ("rust", "rust-native") if v in g]
            ratio = f"{min(pet)/min(rus):.2f}× " + ("faster" if min(pet) > min(rus) else "(slower)") if pet and rus else "–"
            if pet and rus and min(pet) <= min(rus): ratio = f"{min(rus)/min(pet):.2f}× slower"
            print(f"| {k[1]} | {k[2]} | {k[3]} | {itxt} | " + " | ".join(cells) + f" | {ratio} |")
    # correctness cross-check
    bad = []
    for k, g in groups.items():
        if k[0] == "eps" and "petsc" in g and "rust" in g:
            a, b = g["petsc"]["values"], g["rust"]["values"]
            m = min(len(a), len(b))
            err = max(abs(x - y) / abs(x) for x, y in zip(a[:m], b[:m]))
            print(f"\nEPS agreement {k[1]} {k[3]} p={k[2]}: max rel. eigenvalue difference PETSc/Rust = {err:.2e}")
    for k, g in groups.items():
        if k[0] == "ksp" and "petsc" in g and "rust" in g:
            if g["petsc"]["iterations"] != g["rust"]["iterations"]:
                bad.append((k, g["petsc"]["iterations"], g["rust"]["iterations"]))
    if bad:
        print("\nIteration-count differences (PETSc vs Rust):")
        for k, a, b in bad: print(f"  {k}: {a} vs {b}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--suite", default="quick"); r.add_argument("--out", required=True)
    r.add_argument("--variants", default=""); r.add_argument("--timeout", type=int, default=900)
    p = sub.add_parser("report"); p.add_argument("file")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)
