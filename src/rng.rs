//! Tiny deterministic generator (SplitMix64) for reproducible start vectors and test matrices.

#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in `[-1, 1)`.
    pub fn next_signed(&mut self) -> f64 {
        2.0 * self.next_f64() - 1.0
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// A deterministic vector with entries in `[-1, 1)`.
pub fn random_vector(n: usize, seed: u64) -> Vec<f64> {
    let mut r = SplitMix64::new(seed);
    (0..n).map(|_| r.next_signed()).collect()
}
