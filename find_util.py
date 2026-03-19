"""
find_util.py — find the service rate μ that achieves target server
utilizations ρ = 0.3 / 0.5 / 0.8 for the BCAUG89 trace.

For a work-conserving single-server queue the long-run utilization is:

    ρ = λ / μ  →  μ = λ / ρ  where  λ = 1 / mean(IAT)

The direct formula is solved first; the queue simulation is then run to
verify the empirical utilization:

    ρ_empirical = Σ S_k / B[-1]

where S_k are i.i.d. Exp(μ) service times and B[-1] is the final
departure time (total elapsed time with the server open).
"""

import time
import numpy as np

DATA_PATH    = "BCAUG89"
TARGET_RHO   = [0.3, 0.5, 0.8]
SEED         = 42


def load_iats(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)


def simulate(iats: np.ndarray, mu: float, rng: np.random.Generator):
    """Lindley recursion; returns (S, A, B)."""
    n = len(iats)
    S = rng.exponential(1.0 / mu, size=n)
    A = np.cumsum(iats)
    B = np.empty(n, dtype=np.float64)
    B[0] = A[0] + S[0]
    for k in range(1, n):
        B[k] = (A[k] if A[k] >= B[k - 1] else B[k - 1]) + S[k]
    return S, A, B


def empirical_utilization(S: np.ndarray, B: np.ndarray) -> float:
    """Fraction of time the server is busy = total service / total elapsed."""
    return float(S.sum() / B[-1])


def main():
    print(f"Loading {DATA_PATH} …")
    iats = load_iats(DATA_PATH)
    n       = len(iats)
    mean_iat = iats.mean()
    lam      = 1.0 / mean_iat

    print(f"  {n:,} IATs")
    print(f"  mean IAT : {mean_iat * 1e3:.4f} ms")
    print(f"  λ        : {lam:.2f} /s")
    print()

    header = (f"  {'ρ_target':>8}  {'μ (1/s)':>10}  "
              f"{'1/μ (ms)':>10}  {'ρ_empirical':>12}  {'time':>6}")
    print(header)
    print("  " + "─" * (len(header) - 2))

    for rho in TARGET_RHO:
        mu  = lam / rho                          # direct solution
        rng = np.random.default_rng(SEED)
        t0  = time.time()
        S, A, B = simulate(iats, mu, rng)
        rho_emp = empirical_utilization(S, B)
        elapsed = time.time() - t0
        print(f"  {rho:>8.1f}  {mu:>10.2f}  "
              f"{1e3 / mu:>10.4f}  {rho_emp:>12.6f}  {elapsed:>5.1f}s")


if __name__ == "__main__":
    main()
