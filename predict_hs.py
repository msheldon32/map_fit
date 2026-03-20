"""
Heteroscedastic S4 for next-value prediction on the BCAUG89 trace.

Extends predict.py with a second model output: the conditional scale α_t.
Instead of predicting only the conditional mean μ_t, the model predicts:

    (μ_t, α_t) = f_θ(x_{1:t})

where α_t > 0 is the predicted local standard deviation of the residual.

Loss (Gaussian NLL, replaces MSE):

    L_t = (x_{t+1} − μ_t)² / (2 α_t²) + log α_t

Generation (bootstrap_train mode):

    x_{t+1} = μ_t + α_t · z*   where z* ~ block-bootstrap({z_t}_train)
    z_t = ε_t / α_t  (standardised residual; ~N(0,1) if model is calibrated)

The key advantage over predict.py: α_t is high during bursty regions and low
during metronomic ones, so the generated IAT sequence is heteroscedastic in
the same way as the true process.  In predict.py the noise amplitude is a
fixed global multiplier; here it is state-dependent.

Architectural difference from predict.py:
  head: Linear(d_model → 1)   →   Linear(d_model → 2)
  output split:  mu = out[..., 0]
                 alpha = softplus(out[..., 1]) + 1e-3

Everything else (TBPTT, ZOH, HiPPO init, state continuity, rate
normalisation, queue simulation) is identical to predict.py.
"""

import argparse
import math
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─── Configuration ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 512
D_MODEL    = 16
D_STATE    = 4
N_LAYERS   = 2
D_FF       = 64
DROPOUT    = 0.0
EPOCHS     = 15
LR         = 1e-3
TRAIN_FRAC = 0.80
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH  = "BCAUG89"


# ─── Data ──────────────────────────────────────────────────────────────────────

def load_and_preprocess(path: str):
    raw      = np.loadtxt(path, dtype=np.float64)
    log_data = np.log10(raw).astype(np.float32)
    mean     = float(log_data.mean())
    std      = float(log_data.std())
    normed   = (log_data - mean) / std
    return normed, mean, std, raw


# ─── State utilities ───────────────────────────────────────────────────────────

def detach_states(states):
    if states is None:
        return None
    return [(sr.detach(), si.detach()) for sr, si in states]


# ─── S4 – S4D-Lin ─────────────────────────────────────────────────────────────

class S4DLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        lam_real = -0.5 * torch.ones(d_state)
        lam_imag = math.pi * torch.arange(d_state, dtype=torch.float32)

        self.log_A_real = nn.Parameter(
            torch.log(-lam_real).unsqueeze(0).expand(d_model, -1).clone()
        )
        self.A_imag = nn.Parameter(
            lam_imag.unsqueeze(0).expand(d_model, -1).clone()
        )
        self.B_real = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.B_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

        self.D        = nn.Parameter(torch.ones(d_model))
        self.log_dt   = nn.Parameter(torch.zeros(1))
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state=None):
        B, L, H = x.shape
        dt = torch.exp(self.log_dt)

        A_real = -torch.exp(self.log_A_real)
        Abar_r = torch.exp(A_real * dt) * torch.cos(self.A_imag * dt)
        Abar_i = torch.exp(A_real * dt) * torch.sin(self.A_imag * dt)
        Bbar_r = dt * self.B_real
        Bbar_i = dt * self.B_imag

        if state is None:
            sr = x.new_zeros(B, H, self.d_state)
            si = x.new_zeros(B, H, self.d_state)
        else:
            sr, si = state

        outs = []
        for t in range(L):
            u    = x[:, t]
            Bu_r = Bbar_r * u.unsqueeze(-1)
            Bu_i = Bbar_i * u.unsqueeze(-1)
            new_r = Abar_r * sr - Abar_i * si + Bu_r
            new_i = Abar_r * si + Abar_i * sr + Bu_i
            sr, si = new_r, new_i
            y = (self.C_real * sr - self.C_imag * si).sum(-1) + self.D * u
            outs.append(y)

        y = torch.stack(outs, dim=1)
        return self.drop(self.out_proj(y)), (sr, si)


class S4Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ssm   = S4DLayer(d_model, d_state, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor, state=None):
        ssm_out, new_state = self.ssm(self.norm1(x), state)
        x = x + ssm_out
        x = x + self.ff(self.norm2(x))
        return x, new_state


class S4Predictor(nn.Module):
    """Heteroscedastic S4: predicts (μ_t, α_t) at each timestep."""

    def __init__(self, d_model=D_MODEL, d_state=D_STATE,
                 n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.enc  = nn.Linear(1, d_model)
        self.body = nn.ModuleList([
            S4Block(d_model, d_state, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Two outputs: mean (index 0) and pre-softplus scale (index 1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)
        )

    def forward(self, x: torch.Tensor, states=None):
        """
        x:      (B, L, 1)
        states: list of per-layer (sr, si), or None
        returns: mu (B, L),  alpha (B, L),  new_states list
        """
        x = self.enc(x)
        new_states = []
        for i, blk in enumerate(self.body):
            x, new_st = blk(x, states[i] if states is not None else None)
            new_states.append(new_st)
        out   = self.head(self.norm(x))                 # (B, L, 2)
        mu    = out[..., 0]                              # (B, L)
        alpha = F.softplus(out[..., 1]) + 1e-3          # (B, L), strictly positive
        return mu, alpha, new_states


# ─── Training ──────────────────────────────────────────────────────────────────

def _gaussian_nll(mu: torch.Tensor, alpha: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
    """Mean Gaussian NLL: E[(x-μ)²/(2α²) + log α]."""
    return ((target - mu) ** 2 / (2.0 * alpha ** 2) + torch.log(alpha)).mean()


def fit_sequential(model, train_data: np.ndarray, val_data: np.ndarray,
                   name: str, device: str = DEVICE):
    """
    TBPTT training with Gaussian NLL loss.

    The model predicts (μ_t, α_t); the loss is the Gaussian negative log-
    likelihood rather than MSE.  Everything else (state carry-over, cosine
    annealing, gradient clipping) is identical to predict.py.
    """
    model  = model.to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'─'*60}")
    print(f"{name}-HS  ({n_params:,} parameters)  [Gaussian NLL loss]")
    print(f"{'─'*60}")

    tr = torch.FloatTensor(train_data).to(device)
    vl = torch.FloatTensor(val_data).to(device)

    best_val, best_state_dict = float("inf"), None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        # ── Training pass ──────────────────────────────────────────────────
        model.train()
        states     = None
        train_loss = 0.0
        n_chunks   = 0

        for t in range(0, len(tr) - 1, CHUNK_SIZE):
            chunk_in  = tr[t   : t + CHUNK_SIZE    ].unsqueeze(0).unsqueeze(-1)
            chunk_tgt = tr[t+1 : t + CHUNK_SIZE + 1].unsqueeze(0)
            if chunk_in.shape[1] < 2:
                break

            states = detach_states(states)
            opt.zero_grad()
            mu, alpha, states = model(chunk_in, states)
            n = min(mu.shape[1], chunk_tgt.shape[1])
            loss = _gaussian_nll(mu[:, :n], alpha[:, :n], chunk_tgt[:, :n])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            n_chunks   += 1

        sched.step()

        # ── Validation pass ────────────────────────────────────────────────
        model.eval()
        val_states = detach_states(states)
        val_loss   = 0.0
        val_chunks = 0

        with torch.no_grad():
            for t in range(0, len(vl) - 1, CHUNK_SIZE):
                chunk_in  = vl[t   : t + CHUNK_SIZE    ].unsqueeze(0).unsqueeze(-1)
                chunk_tgt = vl[t+1 : t + CHUNK_SIZE + 1].unsqueeze(0)
                if chunk_in.shape[1] < 2:
                    break
                mu, alpha, val_states = model(chunk_in, val_states)
                n = min(mu.shape[1], chunk_tgt.shape[1])
                val_loss  += _gaussian_nll(mu[:, :n], alpha[:, :n],
                                           chunk_tgt[:, :n]).item()
                val_chunks += 1

        tr_l = train_loss / max(n_chunks,   1)
        vl_l = val_loss   / max(val_chunks, 1)

        if vl_l < best_val:
            best_val = vl_l
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 3 == 0 or ep == EPOCHS:
            print(f"  ep {ep:3d}/{EPOCHS}  "
                  f"train={tr_l:.5f}  val={vl_l:.5f}  "
                  f"best={best_val:.5f}  [{time.time()-t0:.0f}s]")

    model.load_state_dict(best_state_dict)
    model.eval()
    return model


# ─── Evaluation & prediction ───────────────────────────────────────────────────

def evaluate_and_predict(model, train_data: np.ndarray, val_data: np.ndarray,
                         mean: float, std: float, raw: np.ndarray,
                         name: str, device: str = DEVICE):
    tr = torch.FloatTensor(train_data).to(device)
    vl = torch.FloatTensor(val_data).to(device)

    model.eval()
    with torch.no_grad():
        states = None
        for t in range(0, len(tr), CHUNK_SIZE):
            chunk_in = tr[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
            if chunk_in.shape[1] == 0:
                break
            _, _, states = model(chunk_in, states)

        val_mu_norm    = []
        val_alpha_norm = []
        val_trues_norm = []
        final_pred_norm = None

        for t in range(0, len(vl), CHUNK_SIZE):
            chunk_in = vl[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
            if chunk_in.shape[1] == 0:
                break
            mu, alpha, states = model(chunk_in, states)
            mu    = mu.squeeze(0)
            alpha = alpha.squeeze(0)

            n_known = min(len(mu), len(vl) - t - 1)
            val_mu_norm.extend(mu[:n_known].cpu().tolist())
            val_alpha_norm.extend(alpha[:n_known].cpu().tolist())
            val_trues_norm.extend(vl[t+1 : t+1+n_known].cpu().tolist())

            if t + chunk_in.shape[1] >= len(vl):
                final_pred_norm = mu[-1].item()

    val_mu    = np.array(val_mu_norm)
    val_alpha = np.array(val_alpha_norm)
    val_trues = np.array(val_trues_norm)

    # Diagnostics: mean predicted alpha and calibration of standardised residuals
    residuals = val_trues - val_mu
    z_scores  = residuals / val_alpha
    print(f"\n{name}-HS  calibration on val set")
    print(f"  mean α       : {val_alpha.mean():.4f}  (expected ~residual std ≈ 1.0)")
    print(f"  std(z-scores): {z_scores.std():.4f}  (calibrated → ~1.0)")
    print(f"  |z| > 2 (%):  {100*(np.abs(z_scores) > 2).mean():.1f}  (Gaussian → ~4.6 %)")

    # Last 20 val predictions (mean only)
    preds_raw = 10 ** (val_mu[-20:] * std + mean)
    trues_raw = 10 ** (val_trues[-20:] * std + mean)

    print(f"\n{name}-HS – last 20 validation predictions")
    print(f"  {'#':>4}  {'Predicted':>14}  {'Actual':>14}  {'α':>8}  {'Err %':>8}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*8}  {'─'*8}")
    for i, (p, t_, a) in enumerate(zip(preds_raw, trues_raw, val_alpha[-20:])):
        err = 100.0 * (p - t_) / t_
        print(f"  {i+1:4d}  {p:14.6e}  {t_:14.6e}  {a:8.4f}  {err:+8.2f}%")

    pred_raw = float(10 ** (final_pred_norm * std + mean))
    return final_pred_norm, pred_raw


# ─── Queue simulation ──────────────────────────────────────────────────────────

def _simulate_queue(iats_s: np.ndarray, mu: float, rng: np.random.Generator):
    n = len(iats_s)
    S = rng.exponential(1.0 / mu, size=n)
    A = np.cumsum(iats_s)
    B = np.empty(n, dtype=np.float64)
    B[0] = A[0] + S[0]
    for k in range(1, n):
        B[k] = (A[k] if A[k] >= B[k - 1] else B[k - 1]) + S[k]
    n_departed = np.searchsorted(B, A, side="left")
    ql = (np.arange(1, n + 1, dtype=np.int32) - n_departed).clip(min=0)
    return ql, A, B


@torch.no_grad()
def _hs_teacher_forcing_preds(model, train_data: np.ndarray, val_data: np.ndarray,
                               n: int, device: str = DEVICE):
    """Teacher-forcing pass over val; returns (mu_preds, alpha_preds, trues)."""
    model.eval()
    tr = torch.FloatTensor(train_data).to(device)
    vl = torch.FloatTensor(val_data[: n + 1]).to(device)

    states = None
    for t in range(0, len(tr), CHUNK_SIZE):
        chunk_in = tr[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
        if chunk_in.shape[1] == 0:
            break
        _, _, states = model(chunk_in, states)

    mu_preds    = []
    alpha_preds = []
    for t in range(0, n, CHUNK_SIZE):
        chunk_in = vl[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
        if chunk_in.shape[1] == 0:
            break
        mu, alpha, states = model(chunk_in, states)
        mu_preds.extend(mu.squeeze(0).cpu().tolist())
        alpha_preds.extend(alpha.squeeze(0).cpu().tolist())

    mu_preds    = np.array(mu_preds[:n])
    alpha_preds = np.array(alpha_preds[:n])
    trues_norm  = val_data[1 : n + 1]
    return mu_preds, alpha_preds, trues_norm


@torch.no_grad()
def _hs_train_residuals(model, train_data: np.ndarray,
                         device: str = DEVICE):
    """Teacher-forcing pass over training data.

    Returns (z_standardised, alphas):
      z_t     = ε_t / α_t   (standardised residuals; ~N(0,1) if calibrated)
      alphas  = α_t array   (predicted local scales)
    """
    model.eval()
    tr     = torch.FloatTensor(train_data).to(device)
    states = None
    mu_list    = []
    alpha_list = []
    for t in range(0, len(tr) - 1, CHUNK_SIZE):
        chunk_in = tr[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
        if chunk_in.shape[1] == 0:
            break
        mu, alpha, states = model(chunk_in, states)
        n_known = min(mu.shape[1], len(tr) - t - 1)
        mu_list.extend(mu.squeeze(0)[:n_known].cpu().tolist())
        alpha_list.extend(alpha.squeeze(0)[:n_known].cpu().tolist())

    mu_arr    = np.array(mu_list)
    alpha_arr = np.array(alpha_list)
    trues     = train_data[1 : len(mu_arr) + 1]
    eps       = trues - mu_arr
    z         = eps / alpha_arr
    return z, alpha_arr


@torch.no_grad()
def _hs_bootstrap_train_iats(model, train_data: np.ndarray,
                               mean: float, std: float,
                               z_train: np.ndarray,
                               n_gen: int, noise_block_size: int,
                               noise_scale: float, seed: int,
                               device: str = DEVICE) -> np.ndarray:
    """Autoregressive generation with heteroscedastic noise.

    At each step:
      1. Model predicts (μ_t, α_t) from the fed-back previous input.
      2. z* is drawn from the block-bootstrapped training standardised residuals.
      3. x_{t+1} = μ_t + noise_scale · α_t · z*

    The local scale α_t modulates the noise amplitude per-step, unlike
    predict.py where a single global noise_scale is applied uniformly.
    """
    model.eval()
    tr = torch.FloatTensor(train_data).to(device)

    states = None
    for t in range(0, len(tr), CHUNK_SIZE):
        chunk_in = tr[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
        if chunk_in.shape[1] == 0:
            break
        _, _, states = model(chunk_in, states)

    # Pre-sample standardised noise from training z distribution
    rng    = np.random.default_rng(seed)
    z_boot = _block_bootstrap(z_train, n_gen, noise_block_size, rng)

    x = torch.tensor([[[train_data[-1]]]], dtype=torch.float32, device=device)
    mu, alpha, states = model(x, states)

    iats_raw = np.empty(n_gen, dtype=np.float64)
    for i in range(n_gen):
        mu_val    = mu[0, 0].item()
        alpha_val = alpha[0, 0].item()
        pred_norm = mu_val + noise_scale * alpha_val * z_boot[i]
        iats_raw[i] = 10 ** (pred_norm * std + mean)
        x = torch.tensor([[[pred_norm]]], dtype=torch.float32, device=device)
        mu, alpha, states = model(x, states)

    return iats_raw


def _block_bootstrap(arr: np.ndarray, n: int, block_size: int,
                     rng: np.random.Generator) -> np.ndarray:
    m      = len(arr)
    result = np.empty(n, dtype=arr.dtype)
    filled = 0
    while filled < n:
        start = int(rng.integers(0, m - block_size + 1))
        block = arr[start : start + block_size]
        take  = min(block_size, n - filled)
        result[filled : filled + take] = block[:take]
        filled += take
    return result


def queue_test(model, train_data: np.ndarray, val_data: np.ndarray,
               mean: float, std: float, raw_val: np.ndarray,
               mu: float = None, n_customers: int = None,
               generation: str = "bootstrap_train",
               noise_block_size: int = 500, noise_scale: float = 1.0,
               name: str = "Model", device: str = DEVICE, seed: int = 42):
    """Queue test for heteroscedastic model.

    generation options
    ------------------
    "bootstrap_train" (default) — autoregressive; per-step noise = α_t · z*
                                  where z* is bootstrapped from training
                                  standardised residuals {ε_t / α_t}_train.
    "teacher"                   — teacher-forcing on val data; noise = α_t · z*
                                  where z* is bootstrapped from val standardised
                                  residuals.
    """
    n = (len(raw_val) - 1) if n_customers is None else min(n_customers, len(raw_val) - 1)

    lam_trace = 1.0 / float(raw_val[:n].mean())
    if mu is None:
        mu = lam_trace / 0.80
    rho_trace = lam_trace / mu

    mode_str = f"{generation}+hs_noise(scale={noise_scale})"
    print(f"\n{'─'*60}")
    print(f"Queue test  ({name}-HS, {mode_str})")
    print(f"  customers  : {n:,}")
    print(f"  mu         : {mu:.2f} /s   (mean service {1e3/mu:.2f} ms)")
    print(f"  λ (trace)  : {lam_trace:.2f} /s")
    print(f"  ρ (trace)  : {rho_trace:.3f}")
    print(f"{'─'*60}")

    rng_trace = np.random.default_rng(seed)
    ql_trace, _, _ = _simulate_queue(raw_val[:n], mu, rng_trace)

    print(f"  Generating {n:,} IATs ({mode_str}) …", flush=True)
    t0 = time.time()

    if generation == "teacher":
        mu_preds, alpha_preds, trues_norm = _hs_teacher_forcing_preds(
            model, train_data, val_data, n, device
        )
        residuals = trues_norm - mu_preds
        z_scores  = residuals / alpha_preds
        rng_noise = np.random.default_rng(seed + 1)
        z_boot    = _block_bootstrap(z_scores, n, noise_block_size, rng_noise)
        model_iats = 10 ** ((mu_preds + noise_scale * alpha_preds * z_boot) * std + mean)
        print(f"  Val residual RMSE: {float(np.sqrt((residuals**2).mean())):.4f}  "
              f"mean α: {alpha_preds.mean():.4f}  std(z): {z_scores.std():.4f}")

    else:  # bootstrap_train
        z_train, alpha_train = _hs_train_residuals(model, train_data, device)
        print(f"  Train std(z): {z_train.std():.4f}  mean α: {alpha_train.mean():.4f}  "
              f"block_size: {noise_block_size}  noise_scale: {noise_scale}")
        model_iats = _hs_bootstrap_train_iats(
            model, train_data, mean, std, z_train,
            n, noise_block_size, noise_scale, seed + 1, device
        )

    print(f"  Generated in {time.time() - t0:.1f} s")

    # ── Rate normalisation ──────────────────────────────────────────────────────
    trace_mean_iat = float(raw_val[:n].mean())
    model_mean_iat = float(model_iats.mean())
    rate_scale     = trace_mean_iat / model_mean_iat
    model_iats     = model_iats * rate_scale
    print(f"  Rate scale applied: {rate_scale:.4f}  "
          f"(model mean IAT {model_mean_iat*1e3:.3f} ms → "
          f"{trace_mean_iat*1e3:.3f} ms)")

    lam_model = 1.0 / float(model_iats.mean())
    rho_model = lam_model / mu

    rng_model = np.random.default_rng(seed)
    ql_model, _, _ = _simulate_queue(model_iats, mu, rng_model)

    def _stats(ql):
        return {
            "mean": float(ql.mean()),
            "std":  float(ql.std()),
            "p50":  float(np.percentile(ql, 50)),
            "p90":  float(np.percentile(ql, 90)),
            "p99":  float(np.percentile(ql, 99)),
            "max":  float(ql.max()),
        }

    st, sm = _stats(ql_trace), _stats(ql_model)
    w1 = float(np.abs(np.sort(ql_trace.astype(np.float64))
                      - np.sort(ql_model.astype(np.float64))).mean())

    rows = [
        ("mean QL",  st["mean"],  sm["mean"]),
        ("std QL",   st["std"],   sm["std"]),
        ("p50",      st["p50"],   sm["p50"]),
        ("p90",      st["p90"],   sm["p90"]),
        ("p99",      st["p99"],   sm["p99"]),
        ("max QL",   st["max"],   sm["max"]),
        ("λ (1/s)",  lam_trace,   lam_model),
        ("ρ",        rho_trace,   rho_model),
    ]

    print(f"\n  {'Metric':>10}   {'Trace':>10}   {'Model':>10}   {'|Δ|':>10}")
    print(f"  {'─'*10}   {'─'*10}   {'─'*10}   {'─'*10}")
    for label, tv, mv in rows:
        print(f"  {label:>10}   {tv:>10.3f}   {mv:>10.3f}   {abs(tv - mv):>10.3f}")

    print(f"\n  Wasserstein-1 (queue-length distributions): {w1:.4f}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for ql, label, color in [
        (ql_trace, "Trace",              "steelblue"),
        (ql_model, f"Model-HS ({name})", "darkorange"),
    ]:
        vals, counts = np.unique(ql, return_counts=True)
        cum  = np.cumsum(counts)
        ccdf = (len(ql) - cum) / len(ql)
        mask = (ccdf > 0) & (vals > 0)
        ax.plot(np.log10(vals[mask]), np.log10(ccdf[mask]),
                label=label, color=color, linewidth=1.5)

    ax.set_xlabel("log₁₀(x)")
    ax.set_ylabel("log₁₀(P(L > x))")
    ax.set_xlim(0, 6)
    ax.set_ylim(-6, 0)
    ax.set_title(f"Queue-length CCDF — {name}-HS  (ρ={rho_trace:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = f"queue_ccdf_{name.lower()}_hs.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  CCDF plot saved to {plot_path}")

    return ql_trace, ql_model, model_iats


# ─── Main ──────────────────────────────────────────────────────────────────────

def main(retrain: bool = False):
    print(f"Device: {DEVICE}")
    print(f"Loading {DATA_PATH} …")

    data, mean, std, raw = load_and_preprocess(DATA_PATH)
    N = len(data)

    print(f"  {N:,} inter-arrival times loaded")
    print(f"  Raw  : min={raw.min():.4e}  max={raw.max():.4e}  mean={raw.mean():.4e}")
    print(f"  Log-normalised: μ≈0, σ≈1  (μ={data.mean():.3f}, σ={data.std():.3f})")

    n_train    = int(N * TRAIN_FRAC)
    train_data = data[:n_train]
    val_data   = data[n_train:]

    print(f"  Train: {n_train:,}   Val: {N - n_train:,}")

    predictions = {}
    raw_val = raw[n_train:]

    for name, fresh_model in [("S4", S4Predictor())]:
        ckpt = None if retrain else load_model(name)
        if ckpt is not None:
            model, _, _ = ckpt
        else:
            model = fit_sequential(fresh_model, train_data, val_data, name)
            save_model(model, name, mean, std)

        pred_norm, pred_raw = evaluate_and_predict(
            model, train_data, val_data, mean, std, raw, name
        )
        predictions[name] = (pred_norm, pred_raw)

        queue_test(model, train_data, val_data, mean, std, raw_val,
                   noise_scale=1.0, name=name)

    print(f"\n{'='*60}")
    print("PREDICTION: 1,000,000th inter-arrival time in BCAUG89")
    print(f"{'='*60}")
    print(f"Last known value (999,999th):  {raw[-1]:.6e} s")
    print()
    for name, (pn, pr) in predictions.items():
        print(f"  {name:4s}  →  {pr:.6e} s  (log-norm: {pn:+.4f})")
    print()


def save_model(model, name: str, mean: float, std: float):
    path = f"{name.lower()}_hs_checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(),
        "model_class": type(model).__name__,
        "mean": mean,
        "std":  std,
    }, path)
    print(f"  Saved checkpoint → {path}")


def load_model(name: str, device: str = DEVICE):
    path = f"{name.lower()}_hs_checkpoint.pt"
    if not os.path.exists(path):
        return None
    ckpt  = torch.load(path, map_location=device)
    model = S4Predictor().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded checkpoint ← {path}")
    return model, ckpt["mean"], ckpt["std"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Ignore saved checkpoint and retrain from scratch")
    args = parser.parse_args()
    main(retrain=args.retrain)
