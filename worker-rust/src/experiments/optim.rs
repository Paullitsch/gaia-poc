//! Gradient-free optimizers for GAIA.
//!
//! CMA-ES: Covariance Matrix Adaptation Evolution Strategy
//! Reference: Hansen's purecma.py (github.com/CMA-ES/pycma)

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

// ─── Matrix helpers (for full covariance) ─────────────────────────────

/// Symmetric matrix eigendecomposition via Jacobi iteration.
/// Returns (eigenvalues, eigenvectors_as_columns).
/// Each column of the returned matrix is an eigenvector.
fn symmetric_eigen(a: &[Vec<f64>]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = a.len();
    // Work on a copy
    let mut m: Vec<Vec<f64>> = a.to_vec();
    // Identity matrix for eigenvectors
    let mut v: Vec<Vec<f64>> = (0..n).map(|i| {
        let mut row = vec![0.0; n];
        row[i] = 1.0;
        row
    }).collect();

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut p = 0usize;
        let mut q = 1usize;
        let mut max_val = 0.0f64;
        for i in 0..n {
            for j in (i+1)..n {
                let val = m[i][j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < 1e-12 { break; }

        // Compute rotation
        let theta = if (m[p][p] - m[q][q]).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * m[p][q] / (m[p][p] - m[q][q])).atan()
        };
        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation
        let mut new_m = m.clone();
        for i in 0..n {
            if i != p && i != q {
                new_m[i][p] = c * m[i][p] + s * m[i][q];
                new_m[p][i] = new_m[i][p];
                new_m[i][q] = -s * m[i][p] + c * m[i][q];
                new_m[q][i] = new_m[i][q];
            }
        }
        new_m[p][p] = c*c*m[p][p] + 2.0*s*c*m[p][q] + s*s*m[q][q];
        new_m[q][q] = s*s*m[p][p] - 2.0*s*c*m[p][q] + c*c*m[q][q];
        new_m[p][q] = 0.0;
        new_m[q][p] = 0.0;
        m = new_m;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = c * vip + s * viq;
            v[i][q] = -s * vip + c * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| m[i][i].max(1e-20)).collect();
    (eigenvalues, v)
}

/// CMA-ES optimizer.
///
/// Full covariance for n <= 2000, diagonal for larger.
/// Reference: Hansen's purecma.py (canonical CMA-ES).
pub struct CmaEs {
    pub n: usize,
    pub sigma: f64,
    pub mean: Vec<f64>,
    pub pop_size: usize,
    pub gen: usize,

    // Covariance representation
    pub use_diagonal: bool,
    /// Diagonal mode: variance per coordinate
    diag: Vec<f64>,
    /// Full mode: n×n covariance matrix
    c_full: Vec<Vec<f64>>,
    /// Cached eigenvalues (full mode)
    eigenvalues: Vec<f64>,
    /// Cached eigenvectors as columns (full mode) — v[row][col]
    eigenvectors: Vec<Vec<f64>>,
    /// Counter for lazy eigen update
    eigen_eval_count: usize,
    lazy_gap_evals: f64,

    /// Evolution path for sigma
    ps: Vec<f64>,
    /// Evolution path for covariance
    pc: Vec<f64>,

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

        // Weights (log-scale, normalized)
        let raw_w: Vec<f64> = (0..mu).map(|i| ((mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())).collect();
        let sum_w: f64 = raw_w.iter().sum();
        let weights: Vec<f64> = raw_w.iter().map(|w| w / sum_w).collect();
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Strategy parameters (Hansen's canonical formulas)
        let cs = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        // Hansen's damps formula (from purecma.py):
        let ds = 2.0 * mu_eff / lam as f64 + 0.3 + cs;
        let cc = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);
        let c1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
        let cmu_raw = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n as f64 + 2.0).powi(2) + mu_eff);
        let cmu = cmu_raw.min(1.0 - c1);
        let chi_n = (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));

        let use_diagonal = n > 2000;
        let lazy_gap_evals = if use_diagonal { f64::MAX } else {
            0.5 * n as f64 * lam as f64 * (c1 + cmu).recip() / (n as f64 * n as f64)
        };

        // Initialize covariance
        let diag = vec![1.0; n];
        let c_full = if !use_diagonal {
            (0..n).map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            }).collect()
        } else {
            vec![]
        };
        let eigenvalues = vec![1.0; n];
        let eigenvectors = if !use_diagonal {
            (0..n).map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            }).collect()
        } else {
            vec![]
        };

        CmaEs {
            n,
            sigma: sigma0,
            mean: vec![0.0; n],
            pop_size: lam,
            gen: 0,
            use_diagonal,
            diag,
            c_full,
            eigenvalues,
            eigenvectors,
            eigen_eval_count: 0,
            lazy_gap_evals,
            ps: vec![0.0; n],
            pc: vec![0.0; n],
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

    /// Update eigensystem if enough evaluations have passed (lazy update).
    fn update_eigensystem(&mut self) {
        if self.use_diagonal { return; }
        let evals_since = (self.gen * self.pop_size) as f64 - self.eigen_eval_count as f64;
        if evals_since < self.lazy_gap_evals && self.gen > 0 { return; }

        let (eigenvalues, eigenvectors) = symmetric_eigen(&self.c_full);
        self.eigenvalues = eigenvalues;
        self.eigenvectors = eigenvectors;
        self.eigen_eval_count = self.gen * self.pop_size;
    }

    /// Compute C^(-1/2) * vector using cached eigensystem.
    /// C^(-1/2) = B * D^(-1) * B^T where B=eigenvectors, D=diag(sqrt(eigenvalues))
    fn c_invsqrt_mul(&self, v: &[f64]) -> Vec<f64> {
        let n = self.n;
        // z = B^T * v
        let mut z = vec![0.0; n];
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..n {
                s += self.eigenvectors[i][j] * v[i]; // eigenvectors[i][j] = B_ij, column j
            }
            z[j] = s;
        }
        // z = D^(-1) * z
        for j in 0..n {
            z[j] /= self.eigenvalues[j].sqrt().max(1e-20);
        }
        // result = B * z
        let mut result = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += self.eigenvectors[i][j] * z[j];
            }
            result[i] = s;
        }
        result
    }

    /// Sample a population of candidates.
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        if !self.use_diagonal {
            self.update_eigensystem();
        }

        let mut pop = Vec::with_capacity(self.pop_size);
        for _ in 0..self.pop_size {
            let z = self.rng.randn_vec(self.n);

            if self.use_diagonal {
                let x: Vec<f64> = (0..self.n)
                    .map(|i| self.mean[i] + self.sigma * self.diag[i].sqrt() * z[i])
                    .collect();
                pop.push(x);
            } else {
                // x = mean + sigma * B * D * z
                // where B = eigenvectors (columns), D = diag(sqrt(eigenvalues))
                let mut y = vec![0.0; self.n];
                for i in 0..self.n {
                    let dz = self.eigenvalues[i].sqrt() * z[i];
                    for j in 0..self.n {
                        y[j] += self.eigenvectors[j][i] * dz;
                    }
                }
                let x: Vec<f64> = (0..self.n)
                    .map(|i| self.mean[i] + self.sigma * y[i])
                    .collect();
                pop.push(x);
            }
        }
        pop
    }

    /// Update distribution. Fitnesses: higher is better (maximization).
    pub fn tell(&mut self, candidates: &[Vec<f64>], fitnesses: &[f64]) {
        assert_eq!(candidates.len(), self.pop_size);
        assert_eq!(fitnesses.len(), self.pop_size);

        let n = self.n;

        // Sort by fitness (descending — best first)
        let mut indices: Vec<usize> = (0..self.pop_size).collect();
        indices.sort_by(|&a, &b| fitnesses[b].partial_cmp(&fitnesses[a]).unwrap());

        // Weighted recombination
        let old_mean = self.mean.clone();
        for i in 0..n {
            self.mean[i] = 0.0;
            for k in 0..self.mu {
                self.mean[i] += self.weights[k] * candidates[indices[k]][i];
            }
        }

        // y = xmean_new - xmean_old
        let y: Vec<f64> = (0..n).map(|i| self.mean[i] - old_mean[i]).collect();

        if self.use_diagonal {
            // ── Diagonal mode (same as before but with Hansen's damps) ──
            let invsqrt_diag: Vec<f64> = self.diag.iter().map(|d| 1.0 / d.sqrt()).collect();
            let factor = (self.cs * (2.0 - self.cs) * self.mu_eff).sqrt() / self.sigma;
            for i in 0..n {
                self.ps[i] = (1.0 - self.cs) * self.ps[i] + factor * invsqrt_diag[i] * y[i];
            }

            let ps_norm: f64 = self.ps.iter().map(|p| p * p).sum::<f64>().sqrt();
            let hsig = if ps_norm / (1.0 - (1.0 - self.cs).powi(2 * (self.gen as i32 + 1))).sqrt()
                < (1.4 + 2.0 / (n as f64 + 1.0)) * self.chi_n
            { 1.0 } else { 0.0 };

            let factor_c = (self.cc * (2.0 - self.cc) * self.mu_eff).sqrt() / self.sigma;
            for i in 0..n {
                self.pc[i] = (1.0 - self.cc) * self.pc[i] + hsig * factor_c * y[i];
            }

            // Diagonal covariance update
            for i in 0..n {
                let rank1 = self.pc[i] * self.pc[i];
                let mut rank_mu = 0.0;
                for k in 0..self.mu {
                    let diff = (candidates[indices[k]][i] - old_mean[i]) / self.sigma;
                    rank_mu += self.weights[k] * diff * diff;
                }
                let c1a = self.c1 * (1.0 - (1.0 - hsig * hsig) * self.cc * (2.0 - self.cc));
                self.diag[i] = (1.0 - c1a - self.cmu) * self.diag[i]
                    + self.c1 * rank1
                    + self.cmu * rank_mu;
                self.diag[i] = self.diag[i].max(1e-20);
            }

            // Step-size adaptation
            let cn = self.cs / self.ds;
            self.sigma *= (cn * (ps_norm / self.chi_n - 1.0)).min(1.0).exp();
        } else {
            // ── Full covariance mode (Hansen's canonical algorithm) ──

            // z = C^(-1/2) * y
            let z = self.c_invsqrt_mul(&y);

            // Update evolution path ps
            let csn = (self.cs * (2.0 - self.cs) * self.mu_eff).sqrt() / self.sigma;
            for i in 0..n {
                self.ps[i] = (1.0 - self.cs) * self.ps[i] + csn * z[i];
            }

            // Heaviside function for stalling check
            let ps_norm_sq: f64 = self.ps.iter().map(|p| p * p).sum();
            let gen_factor = 1.0 - (1.0 - self.cs).powi(2 * (self.gen as i32 + 1));
            let hsig = if ps_norm_sq / n as f64 / gen_factor < 2.0 + 4.0 / (n as f64 + 1.0) {
                1.0
            } else {
                0.0
            };

            // Update evolution path pc
            let ccn = (self.cc * (2.0 - self.cc) * self.mu_eff).sqrt() / self.sigma;
            for i in 0..n {
                self.pc[i] = (1.0 - self.cc) * self.pc[i] + ccn * hsig * y[i];
            }

            // Adapt covariance matrix C
            let c1a = self.c1 * (1.0 - (1.0 - hsig * hsig) * self.cc * (2.0 - self.cc));
            let old_weight = 1.0 - c1a - self.cmu * self.weights.iter().sum::<f64>();

            for i in 0..n {
                for j in 0..=i {
                    // Shrink old C
                    self.c_full[i][j] *= old_weight;

                    // Rank-one update: c1 * pc * pc^T
                    self.c_full[i][j] += self.c1 * self.pc[i] * self.pc[j];

                    // Rank-mu update: cmu * sum(w_k * dx_k * dx_k^T)
                    let mut rank_mu = 0.0;
                    for k in 0..self.mu {
                        let di = (candidates[indices[k]][i] - old_mean[i]) / self.sigma;
                        let dj = (candidates[indices[k]][j] - old_mean[j]) / self.sigma;
                        rank_mu += self.weights[k] * di * dj;
                    }
                    self.c_full[i][j] += self.cmu * rank_mu;

                    // Symmetric
                    self.c_full[j][i] = self.c_full[i][j];
                }
            }

            // Step-size adaptation
            let ps_norm = ps_norm_sq.sqrt();
            let cn = self.cs / self.ds;
            self.sigma *= (cn * (ps_norm / self.chi_n - 1.0)).min(1.0).exp();
        }

        self.sigma = self.sigma.max(1e-20).min(1e10);
        self.gen += 1;
    }
}

/// Compute centered ranks: maps fitness values to [-0.5, 0.5] based on rank.
/// Reference: OpenAI's evolution-strategies-starter/es.py
pub fn compute_centered_ranks(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n <= 1 { return vec![0.0; n]; }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    for (rank, &idx) in indices.iter().enumerate() {
        ranks[idx] = rank as f64 / (n - 1) as f64 - 0.5;
    }
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cma_sphere() {
        // Minimize sphere function (maximize negative)
        let n = 10;
        let mut cma = CmaEs::new(n, 1.0, None);

        for _ in 0..300 {
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

    #[test]
    fn test_centered_ranks() {
        let x = vec![10.0, 30.0, 20.0, 40.0];
        let ranks = compute_centered_ranks(&x);
        // 10→rank0, 20→rank1, 30→rank2, 40→rank3
        // centered: 0/3-0.5=-0.5, 2/3-0.5=0.167, 1/3-0.5=-0.167, 3/3-0.5=0.5
        assert!((ranks[0] - (-0.5)).abs() < 0.01);
        assert!((ranks[3] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_full_covariance_sphere() {
        // Small dimension — should use full covariance
        let n = 5;
        let mut cma = CmaEs::new(n, 1.0, None);
        assert!(!cma.use_diagonal);

        for _ in 0..200 {
            let pop = cma.ask();
            let fitnesses: Vec<f64> = pop.iter()
                .map(|x| -x.iter().map(|v| v * v).sum::<f64>())
                .collect();
            cma.tell(&pop, &fitnesses);
        }

        let dist: f64 = cma.mean.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(dist < 0.01, "Full CMA-ES should converge on sphere, got dist={:.4}", dist);
    }
}
