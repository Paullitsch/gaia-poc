//! Gradient-free optimizers for GAIA.
//!
//! CMA-ES: Covariance Matrix Adaptation Evolution Strategy
//! The gold standard of gradient-free optimization.

/// Simple PRNG for reproducible optimization.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng { state: seed.wrapping_add(1) }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    /// Gaussian sample via Box-Muller.
    pub fn randn(&mut self) -> f64 {
        let u1 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-10);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    pub fn randn_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.randn()).collect()
    }

    /// Uniform random in [lo, hi).
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + u * (hi - lo)
    }
}

/// CMA-ES optimizer.
///
/// Diagonal CMA for high dimensions (>1000 params).
/// Full covariance for smaller problems.
pub struct CmaEs {
    pub n: usize,
    pub sigma: f64,
    pub mean: Vec<f64>,
    pub pop_size: usize,
    pub gen: usize,
    /// Diagonal of the covariance (diagonal CMA for large n).
    diag: Vec<f64>,
    /// Path for sigma adaptation.
    ps: Vec<f64>,
    /// Path for covariance adaptation.
    pc: Vec<f64>,
    /// Use diagonal-only mode (for n > 1000).
    pub use_diagonal: bool,
    // CMA-ES constants
    mu: usize,
    weights: Vec<f64>,
    mu_eff: f64,
    cs: f64,
    ds: f64,
    cc: f64,
    c1: f64,
    cmu: f64,
    chi_n: f64,
    rng: Rng,
}

impl CmaEs {
    pub fn new(n: usize, sigma0: f64, pop_size: Option<usize>) -> Self {
        let lam = pop_size.unwrap_or(4 + (3.0 * (n as f64).ln()) as usize);
        let mu = lam / 2;

        // Weights
        let raw_w: Vec<f64> = (0..mu).map(|i| ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())).collect();
        let sum_w: f64 = raw_w.iter().sum();
        let weights: Vec<f64> = raw_w.iter().map(|w| w / sum_w).collect();
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Strategy parameters
        let cs = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let ds = 1.0 + 2.0 * (((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let cc = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);
        let c1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
        let cmu_raw = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n as f64 + 2.0).powi(2) + mu_eff);
        let cmu = cmu_raw.min(1.0 - c1);
        let chi_n = (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));

        let use_diagonal = n > 1000;

        CmaEs {
            n,
            sigma: sigma0,
            mean: vec![0.0; n],
            pop_size: lam,
            gen: 0,
            diag: vec![1.0; n],
            ps: vec![0.0; n],
            pc: vec![0.0; n],
            use_diagonal,
            mu,
            weights,
            mu_eff,
            cs,
            ds,
            cc,
            c1,
            cmu,
            chi_n,
            rng: Rng::new(42),
        }
    }

    /// Sample a population of candidates.
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        let mut pop = Vec::with_capacity(self.pop_size);
        for _ in 0..self.pop_size {
            let z = self.rng.randn_vec(self.n);
            let x: Vec<f64> = (0..self.n)
                .map(|i| self.mean[i] + self.sigma * self.diag[i].sqrt() * z[i])
                .collect();
            pop.push(x);
        }
        pop
    }

    /// Update distribution given candidates and their fitnesses.
    /// Candidates should be the full population from `ask()`.
    /// Fitnesses: higher is better (maximization).
    pub fn tell(&mut self, candidates: &[Vec<f64>], fitnesses: &[f64]) {
        assert_eq!(candidates.len(), self.pop_size);
        assert_eq!(fitnesses.len(), self.pop_size);

        // Sort by fitness (descending — best first)
        let mut indices: Vec<usize> = (0..self.pop_size).collect();
        indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

        // Weighted recombination of mu best
        let old_mean = self.mean.clone();
        for i in 0..self.n {
            self.mean[i] = 0.0;
            for k in 0..self.mu {
                self.mean[i] += self.weights[k] * candidates[indices[k]][i];
            }
        }

        // Evolution paths
        let invsqrt_diag: Vec<f64> = self.diag.iter().map(|d| 1.0 / d.sqrt()).collect();
        let factor = (self.cs * (2.0 - self.cs) * self.mu_eff).sqrt() / self.sigma;

        for i in 0..self.n {
            self.ps[i] = (1.0 - self.cs) * self.ps[i]
                + factor * invsqrt_diag[i] * (self.mean[i] - old_mean[i]);
        }

        let ps_norm: f64 = self.ps.iter().map(|p| p * p).sum::<f64>().sqrt();
        let hs = if ps_norm / (1.0 - (1.0 - self.cs).powi(2 * (self.gen as i32 + 1))).sqrt()
            < (1.4 + 2.0 / (self.n as f64 + 1.0)) * self.chi_n
        { 1.0 } else { 0.0 };

        let factor_c = (self.cc * (2.0 - self.cc) * self.mu_eff).sqrt() / self.sigma;
        for i in 0..self.n {
            self.pc[i] = (1.0 - self.cc) * self.pc[i]
                + hs * factor_c * (self.mean[i] - old_mean[i]);
        }

        // Covariance (diagonal) update
        for i in 0..self.n {
            let rank1 = self.pc[i] * self.pc[i];
            let mut rank_mu = 0.0;
            for k in 0..self.mu {
                let diff = (candidates[indices[k]][i] - old_mean[i]) / self.sigma;
                rank_mu += self.weights[k] * diff * diff;
            }
            self.diag[i] = (1.0 - self.c1 - self.cmu) * self.diag[i]
                + self.c1 * (rank1 + (1.0 - hs) * self.cc * (2.0 - self.cc) * self.diag[i])
                + self.cmu * rank_mu;
            // Clamp for stability
            self.diag[i] = self.diag[i].max(1e-20);
        }

        // Step-size adaptation
        self.sigma *= ((self.cs / self.ds) * (ps_norm / self.chi_n - 1.0)).exp();
        self.sigma = self.sigma.max(1e-20).min(1e10);

        self.gen += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cma_sphere() {
        // Minimize sphere function (maximize negative)
        let n = 10;
        let mut cma = CmaEs::new(n, 1.0, None);

        for _ in 0..200 {
            let pop = cma.ask();
            let fitnesses: Vec<f64> = pop.iter()
                .map(|x| -x.iter().map(|v| v * v).sum::<f64>())
                .collect();
            cma.tell(&pop, &fitnesses);
        }

        let dist: f64 = cma.mean.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(dist < 0.01, "CMA-ES should converge on sphere, got dist={:.4}", dist);
    }

    #[test]
    fn test_cma_pop_size() {
        let cma = CmaEs::new(100, 0.5, None);
        assert_eq!(cma.pop_size, 4 + (3.0 * 100.0_f64.ln()) as usize);
    }
}
