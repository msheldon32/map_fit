"""
viz_trace.py — standalone diagnostic visualisation of the BCAUG89 trace
and a pure trace-driven single-server queue simulation.

No model is involved.  The goal is to produce the standard plots found in
the LRD / self-similar traffic literature so the simulation can be compared
against published works (Leland et al. 1993, Paxson & Floyd 1995,
Andersen & Nielsen 1998, etc.).

Output files
------------
  iat_dist.png        — marginal distribution of raw and log(IAT)
  iat_acf.png         — sample ACF of log(IAT), up to IAT_ACF_LAGS lags
  variance_time.png   — variance-time plot (Hurst parameter estimate)
  queue_ccdf.png      — log(P(L > x)) vs log(x) at several ρ values,
                        overlaid with the M/M/1 geometric-tail reference

Potential sources of discrepancy with published works
------------------------------------------------------
  - Queue length definition: this script counts the number IN SYSTEM at
    each arrival epoch (includes the customer in service).  Some papers
    plot number WAITING (subtract 1 when L > 0).
  - Service distribution: exponential (M service) is used here; published
    plots sometimes use deterministic (D service) or CBR.
  - Utilisation calibration: μ = λ / ρ where λ = 1 / arithmetic_mean(IAT).
    For heavy-tailed IATs the arithmetic mean >> geometric mean, so the
    effective ρ seen by the server may differ from the nominal value.
  - Warm-up: no samples are discarded.  For large ρ the queue may not
    reach stationarity within a single trace.
  - Epoch: queue length is measured at arrival epochs (not time-averaged).
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH    = "BCAUG89"
RHO_VALUES   = [0.5, 0.7, 0.8, 0.9]
SEED         = 42
IAT_ACF_LAGS = 500          # max lag for ACF plot
VT_M_MAX     = 2 ** 16      # largest block size for variance-time plot


# ─── Data loading ───────────────────────────────────────────────────────────────

def load_iats(path: str) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)


# ─── Queue simulation ────────────────────────────────────────────────────────────

def simulate_queue(iats: np.ndarray, mu: float, rng: np.random.Generator):
    """G/M/1 queue via Lindley's recursion.  Returns queue length in system
    (including the customer being served) at each arrival epoch."""
    n = len(iats)
    S = rng.exponential(1.0 / mu, size=n)
    A = np.cumsum(iats)

    B = np.empty(n, dtype=np.float64)
    B[0] = A[0] + S[0]
    for k in range(1, n):
        B[k] = (A[k] if A[k] >= B[k - 1] else B[k - 1]) + S[k]

    # B is non-decreasing; B[j] >= A[j] >= A[k] for j >= k, so
    # searchsorted over the whole array safely counts only past departures.
    n_dep = np.searchsorted(B, A, side="left")
    return (np.arange(1, n + 1, dtype=np.int32) - n_dep).clip(min=0)


def empirical_ccdf(ql: np.ndarray):
    """Returns (unique_values, P(L > value)) for the empirical CCDF."""
    vals, counts = np.unique(ql, return_counts=True)
    cum   = np.cumsum(counts)
    ccdf  = (len(ql) - cum) / len(ql)   # P(L > vals[k])
    return vals, ccdf


# ─── Statistics ──────────────────────────────────────────────────────────────────

def acf_fft(x: np.ndarray, n_lags: int) -> np.ndarray:
    """Sample ACF via FFT (O(N log N)).  Returns lags 0 … n_lags."""
    x  = x - x.mean()
    n  = len(x)
    f  = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n_lags + 1].real
    ac /= ac[0]
    return ac


def variance_time(x: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    """Var of the sample mean over non-overlapping blocks of m samples.

    For an LRD process with Hurst parameter H:
        Var ~ m^(2H-2)  →  log-log slope = 2H - 2

    I.i.d.: slope = -1  (H = 0.5)
    LRD:    slope > -1  (H > 0.5)
    """
    variances = np.empty(len(m_values))
    for i, m in enumerate(m_values):
        n_blocks   = len(x) // m
        block_means = x[: n_blocks * m].reshape(n_blocks, m).mean(axis=1)
        variances[i] = np.var(block_means, ddof=1)
    return variances


def hurst_estimate(m_values: np.ndarray, variances: np.ndarray) -> float:
    """OLS slope on log-log variance-time, returns H = (slope + 2) / 2."""
    lm = np.log10(m_values.astype(float))
    lv = np.log10(variances)
    slope = np.polyfit(lm, lv, 1)[0]
    return (slope + 2.0) / 2.0


# ─── Plots ───────────────────────────────────────────────────────────────────────

def plot_iat_dist(iats: np.ndarray):
    log_iats = np.log10(iats)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Raw IAT (truncated at 99.5th pct for readability)
    axes[0].hist(iats * 1e3, bins=300, density=True,
                 color="steelblue", alpha=0.75)
    axes[0].set_xlim(0, np.percentile(iats * 1e3, 99.5))
    axes[0].set_xlabel("IAT (ms)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Raw IAT — marginal distribution\n"
                      f"mean={iats.mean()*1e3:.3f} ms  "
                      f"std={iats.std()*1e3:.3f} ms  "
                      f"CV={iats.std()/iats.mean():.2f}")

    # log(IAT)
    axes[1].hist(log_iats, bins=300, density=True,
                 color="steelblue", alpha=0.75)
    axes[1].set_xlabel("log₁₀(IAT / s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"log₁₀(IAT) — marginal distribution\n"
                      f"mean={log_iats.mean():.3f}  "
                      f"std={log_iats.std():.3f}  "
                      f"skew={float(((log_iats - log_iats.mean())**3).mean() / log_iats.std()**3):.2f}")

    fig.tight_layout()
    fig.savefig("iat_dist.png", dpi=150)
    plt.close(fig)
    print("  Saved iat_dist.png")


def plot_iat_acf(iats: np.ndarray, n_lags: int):
    log_iats = np.log10(iats)
    lags     = np.arange(n_lags + 1)
    acf      = acf_fft(log_iats, n_lags)
    ci       = 1.96 / np.sqrt(len(iats))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(lags, acf, color="steelblue", linewidth=1.0)
    ax.axhline(0,   color="black", linewidth=0.5)
    ax.axhline( ci, color="red",  linewidth=0.8, linestyle="--",
               label="95 % CI (i.i.d.)")
    ax.axhline(-ci, color="red",  linewidth=0.8, linestyle="--")
    ax.set_xlabel("Lag (packets)")
    ax.set_ylabel("ACF")
    ax.set_title("Sample ACF of log₁₀(IAT) — BCAUG89\n"
                 "(slow decay above CI indicates long-range dependence)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("iat_acf.png", dpi=150)
    plt.close(fig)
    print("  Saved iat_acf.png")


def plot_variance_time(iats: np.ndarray, m_max: int):
    log_iats = np.log10(iats)
    m_values = np.unique(np.round(
        np.geomspace(1, min(m_max, len(iats) // 20), 40)
    ).astype(int))
    variances = variance_time(log_iats, m_values)
    H         = hurst_estimate(m_values, variances)

    # Reference lines
    lm   = np.log10(m_values.astype(float))
    lv   = np.log10(variances)
    fit  = np.polyfit(lm, lv, 1)
    fit_line = np.poly1d(fit)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(lm, lv, s=20, color="steelblue", zorder=3, label="Empirical")
    ax.plot(lm, fit_line(lm), color="steelblue", linewidth=1.2,
            linestyle="--", label=f"OLS fit  (H ≈ {H:.3f})")

    # i.i.d. reference (slope = -1)
    ref_iid = lv[0] + (-1.0) * (lm - lm[0])
    ax.plot(lm, ref_iid, color="red", linewidth=1.0, linestyle=":",
            label="i.i.d. slope −1  (H = 0.5)")

    ax.set_xlabel("log₁₀(block size m)")
    ax.set_ylabel("log₁₀(Var of block mean)")
    ax.set_title("Variance-time plot — log₁₀(IAT)\n"
                 "Slope = 2H − 2;  LRD ⟺ slope > −1")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("variance_time.png", dpi=150)
    plt.close(fig)
    print(f"  Saved variance_time.png  (H ≈ {H:.3f})")


def plot_queue_ccdf(iats: np.ndarray, rho_values: list, seed: int):
    lam = 1.0 / iats.mean()
    print(f"  Full trace: {len(iats):,} IATs  "
          f"(mean={iats.mean():.4e} s,  λ={lam:.1f} /s)")
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(rho_values)))

    fig, ax = plt.subplots(figsize=(8, 6))

    for rho, color in zip(rho_values, colors):
        mu  = lam / rho
        rng = np.random.default_rng(seed)
        t0  = time.time()
        ql  = simulate_queue(iats, mu, rng)
        print(f"    ρ={rho} mu={mu} mean QL={ql.mean():.2f}  max QL={ql.max()}  "
              f"n_customers={len(iats)}  [{time.time()-t0:.1f}s]")

        vals, ccdf = empirical_ccdf(ql)
        mask = (ccdf > 0) & (vals > 0)

        # Trace CCDF
        ax.plot(np.log10(vals[mask]), np.log10(ccdf[mask]),
                color=color, linewidth=1.8, label=f"Trace  ρ={rho}")

        # M/M/1 reference: P(L > x) = ρ^(x+1)
        x_ref  = np.arange(1, int(vals[mask].max()) + 1)
        p_ref  = rho ** (x_ref + 1)
        m_mask = p_ref > 0
        ax.plot(np.log10(x_ref[m_mask]), np.log10(p_ref[m_mask]),
                color=color, linewidth=1.0, linestyle="--", alpha=0.55)

    # Legend proxy for M/M/1 dashes
    ax.plot([], [], color="grey", linewidth=1.0, linestyle="--",
            label="M/M/1 reference (dashed)")

    ax.set_xlabel("log₁₀(x)")
    ax.set_ylabel("log₁₀(P(L > x))")
    ax.set_xlim(0, 8)
    ax.set_ylim(-6, 0)
    ax.set_title("Queue-length CCDF — BCAUG89 G/M/1 vs M/M/1\n"
                 "Queue length = number in system at arrival epochs\n"
                 "Heavier tail than M/M/1 → LRD / self-similar arrivals")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("queue_ccdf.png", dpi=150)
    plt.close(fig)
    print("  Saved queue_ccdf.png")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading {DATA_PATH} …")
    iats = load_iats(DATA_PATH)
    N    = len(iats)
    print(f"  {N:,} inter-arrival times")
    print(f"  min      = {iats.min():.4e} s")
    print(f"  max      = {iats.max():.4e} s")
    print(f"  mean     = {iats.mean():.4e} s  ({1/iats.mean():.1f} pkt/s)")
    print(f"  median   = {np.median(iats):.4e} s")
    print(f"  std      = {iats.std():.4e} s")
    print(f"  CV       = {iats.std()/iats.mean():.3f}")
    print(f"  geom mean= {10 ** np.log10(iats).mean():.4e} s  "
          f"(ratio to arith mean: {iats.mean() / 10 ** np.log10(iats).mean():.2f}x)")
    print(f"  p99      = {np.percentile(iats, 99):.4e} s")
    print(f"  p99.9    = {np.percentile(iats, 99.9):.4e} s")
    print()

    print("Plotting IAT distribution …")
    plot_iat_dist(iats)

    print("Plotting ACF …")
    plot_iat_acf(iats, IAT_ACF_LAGS)

    print("Plotting variance-time …")
    plot_variance_time(iats, VT_M_MAX)

    print("Simulating queues …")
    plot_queue_ccdf(iats, RHO_VALUES, SEED)

    print("\nDone.")


if __name__ == "__main__":
    main()
