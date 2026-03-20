"""
S4 and S5 for next-value prediction on the BCAUG89 network traffic trace.

BCAUG89: Bellcore August 1989 inter-arrival times, 999,999 samples.
Goal: Predict the 1,000,000th inter-arrival time.

Architecture differences:
  S4 (S4D-Lin): SISO – one independent diagonal SSM per model channel.
                Each of the H channels evolves its own N-dimensional complex state.
  S5          : MIMO – one shared diagonal SSM across all H channels.
                A single N-dimensional complex state is driven by all H inputs
                simultaneously (via matrix B) and projected back to H outputs (via C).

Both use HiPPO-LegS diagonal initialisation and ZOH discretisation.

Training: TBPTT – the SSM hidden state is carried across chunk boundaries so the
model maintains a running estimate of the MAP hidden state across the full trace,
rather than resetting at every window boundary.
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
CHUNK_SIZE = 512   # TBPTT truncation length
D_MODEL    = 16    # model (hidden) dimension H
D_STATE    = 4     # SSM state dimension N
N_LAYERS   = 2     # number of stacked SSM blocks
D_FF       = 64    # feed-forward expansion
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
    """Detach all state tensors to stop gradients at chunk boundaries."""
    if states is None:
        return None
    return [(sr.detach(), si.detach()) for sr, si in states]


# ─── S4 – S4D-Lin (SISO × H independent channels) ────────────────────────────

class S4DLayer(nn.Module):
    """
    Diagonal S4 (S4D-Lin).  Each of the H model channels has its own
    independent N-dimensional complex diagonal SSM.

    State per channel: x_t = Ā·x_{t-1} + B̄·u_t  (element-wise, complex)
    Output per channel: y_t = Re(C·x_t) + D·u_t

    State shape: (B, H, N) complex, stored as two real tensors (sr, si).
    """

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
        """
        x:     (B, L, H)
        state: (sr, si) each (B, H, N), or None for zero initialisation
        returns: y (B, L, H),  new_state (sr, si) each (B, H, N)
        """
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
            u    = x[:, t]                              # (B, H)
            Bu_r = Bbar_r * u.unsqueeze(-1)             # (B, H, N)
            Bu_i = Bbar_i * u.unsqueeze(-1)
            new_r = Abar_r * sr - Abar_i * si + Bu_r
            new_i = Abar_r * si + Abar_i * sr + Bu_i
            sr, si = new_r, new_i
            y = (self.C_real * sr - self.C_imag * si).sum(-1) + self.D * u
            outs.append(y)

        y = torch.stack(outs, dim=1)                    # (B, L, H)
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
    """S4 model: at each timestep t, predict the next value x[t+1]."""

    def __init__(self, d_model=D_MODEL, d_state=D_STATE,
                 n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.enc  = nn.Linear(1, d_model)
        self.body = nn.ModuleList([
            S4Block(d_model, d_state, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor, states=None):
        """
        x:      (B, L, 1)
        states: list of per-layer (sr, si), or None
        returns: pred (B, L),  new_states list
        """
        x = self.enc(x)
        new_states = []
        for i, blk in enumerate(self.body):
            x, new_st = blk(x, states[i] if states is not None else None)
            new_states.append(new_st)
        return self.head(self.norm(x)).squeeze(-1), new_states


# ─── S5 – Simplified S5 (MIMO shared state across all channels) ───────────────

class S5Layer(nn.Module):
    """
    Simplified S5.  A single N-dimensional complex diagonal state is shared
    across all H model channels.

    State update: x_t = Λ̄·x_{t-1} + B̄·u_t   (Λ̄ diagonal N×N, B̄ ∈ ℂ^{N×H})
    Output:       y_t = Re(C·x_t) + D·u_t      (C ∈ ℂ^{H×N})

    State shape: (B, N) complex, stored as two real tensors (sr, si).
    """

    def __init__(self, d_model: int, d_state: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        lam_real = -0.5 * torch.ones(d_state)
        lam_imag = math.pi * torch.arange(d_state, dtype=torch.float32)
        self.log_Lam_real = nn.Parameter(torch.log(-lam_real))
        self.Lam_imag     = nn.Parameter(lam_imag.clone())

        self.B_real = nn.Parameter(torch.randn(d_state, d_model) * 0.02)
        self.B_imag = nn.Parameter(torch.randn(d_state, d_model) * 0.02)
        self.C_real = nn.Parameter(torch.randn(d_model, d_state) * 0.02)
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.02)

        self.D        = nn.Parameter(torch.ones(d_model))
        self.log_dt   = nn.Parameter(torch.zeros(1))
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state=None):
        """
        x:     (B, L, H)
        state: (sr, si) each (B, N), or None
        returns: y (B, L, H),  new_state (sr, si) each (B, N)
        """
        B, L, H = x.shape
        dt     = torch.exp(self.log_dt)
        Lam_r  = -torch.exp(self.log_Lam_real)
        Lbar_r = torch.exp(Lam_r * dt) * torch.cos(self.Lam_imag * dt)
        Lbar_i = torch.exp(Lam_r * dt) * torch.sin(self.Lam_imag * dt)
        Bbar_r = dt * self.B_real
        Bbar_i = dt * self.B_imag

        if state is None:
            sr = x.new_zeros(B, self.d_state)
            si = x.new_zeros(B, self.d_state)
        else:
            sr, si = state

        outs = []
        for t in range(L):
            u     = x[:, t]
            Bu_r  = u @ Bbar_r.T
            Bu_i  = u @ Bbar_i.T
            new_r = Lbar_r * sr - Lbar_i * si + Bu_r
            new_i = Lbar_r * si + Lbar_i * sr + Bu_i
            sr, si = new_r, new_i
            y = (sr @ self.C_real.T - si @ self.C_imag.T) + self.D * u
            outs.append(y)

        y = torch.stack(outs, dim=1)
        return self.drop(self.out_proj(y)), (sr, si)


class S5Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ssm   = S5Layer(d_model, d_state, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x: torch.Tensor, state=None):
        ssm_out, new_state = self.ssm(self.norm1(x), state)
        x = x + ssm_out
        x = x + self.ff(self.norm2(x))
        return x, new_state


class S5Predictor(nn.Module):
    """S5 model: at each timestep t, predict the next value x[t+1]."""

    def __init__(self, d_model=D_MODEL, d_state=D_STATE,
                 n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.enc  = nn.Linear(1, d_model)
        self.body = nn.ModuleList([
            S5Block(d_model, d_state, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(self, x: torch.Tensor, states=None):
        x = self.enc(x)
        new_states = []
        for i, blk in enumerate(self.body):
            x, new_st = blk(x, states[i] if states is not None else None)
            new_states.append(new_st)
        return self.head(self.norm(x)).squeeze(-1), new_states


# ─── Training ──────────────────────────────────────────────────────────────────

def fit_sequential(model, train_data: np.ndarray, val_data: np.ndarray,
                   name: str, device: str = DEVICE):
    """
    TBPTT training over the full trace.

    For each chunk of length CHUNK_SIZE, the hidden state from the previous
    chunk is detached (stopping gradients) and passed as the initial state,
    so the model always sees the full preceding history at inference time.

    Validation continues from the final training state of each epoch, so the
    state transition from the training region into the validation region is
    seamless — matching the sequential structure of the MAP process.
    """
    model  = model.to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'─'*60}")
    print(f"{name}  ({n_params:,} parameters)")
    print(f"{'─'*60}")

    tr = torch.FloatTensor(train_data).to(device)   # (N_train,)
    vl = torch.FloatTensor(val_data).to(device)     # (N_val,)

    best_val, best_state_dict = float("inf"), None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        # ── Training pass ──────────────────────────────────────────────────
        model.train()
        states     = None
        train_loss = 0.0
        n_chunks   = 0

        for t in range(0, len(tr) - 1, CHUNK_SIZE):
            chunk_in  = tr[t   : t + CHUNK_SIZE    ].unsqueeze(0).unsqueeze(-1)  # (1,C,1)
            chunk_tgt = tr[t+1 : t + CHUNK_SIZE + 1].unsqueeze(0)               # (1,C)
            if chunk_in.shape[1] < 2:
                break

            states = detach_states(states)
            opt.zero_grad()
            pred, states = model(chunk_in, states)
            n = min(pred.shape[1], chunk_tgt.shape[1])
            loss = F.mse_loss(pred[:, :n], chunk_tgt[:, :n])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            n_chunks   += 1

        sched.step()

        # ── Validation pass (state carried from end of training) ───────────
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
                pred, val_states = model(chunk_in, val_states)
                n = min(pred.shape[1], chunk_tgt.shape[1])
                val_loss  += F.mse_loss(pred[:, :n], chunk_tgt[:, :n]).item()
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
    """
    One final sequential pass with the best model:
      1. Warm up state over the training portion.
      2. Run through validation, collecting (pred, true) at every step.
      3. Print last 20 val predictions.
      4. Report the prediction for the value beyond the end of the trace.
         (pred[t] is trained to predict data[t+1], so pred at the last
          val input already gives us the 1,000,000th IAT estimate.)
    """
    tr = torch.FloatTensor(train_data).to(device)
    vl = torch.FloatTensor(val_data).to(device)

    model.eval()
    with torch.no_grad():
        # ── Warm up over training data ─────────────────────────────────────
        states = None
        for t in range(0, len(tr), CHUNK_SIZE):
            chunk_in = tr[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
            if chunk_in.shape[1] == 0:
                break
            _, states = model(chunk_in, states)

        # ── Pass through validation, collecting predictions ────────────────
        val_preds_norm = []
        val_trues_norm = []
        final_pred_norm = None

        for t in range(0, len(vl), CHUNK_SIZE):
            chunk_in = vl[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
            if chunk_in.shape[1] == 0:
                break
            pred, states = model(chunk_in, states)
            pred = pred.squeeze(0)  # (C,)

            # pred[k] predicts vl[t+k+1]; collect only those with a known target
            n_known = min(len(pred), len(vl) - t - 1)
            val_preds_norm.extend(pred[:n_known].cpu().tolist())
            val_trues_norm.extend(vl[t+1 : t+1+n_known].cpu().tolist())

            # The prediction for the value beyond the trace is pred[-1] on
            # whichever chunk contains vl[-1] as its last input element.
            if t + chunk_in.shape[1] >= len(vl):
                final_pred_norm = pred[-1].item()

    val_preds = np.array(val_preds_norm)
    val_trues = np.array(val_trues_norm)

    # ── Last 20 validation predictions ────────────────────────────────────
    preds_raw = 10 ** (val_preds[-20:] * std + mean)
    trues_raw = 10 ** (val_trues[-20:] * std + mean)

    print(f"\n{name} – last 20 validation predictions")
    print(f"  {'#':>4}  {'Predicted':>14}  {'Actual':>14}  {'Err %':>8}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*8}")
    for i, (p, t_) in enumerate(zip(preds_raw, trues_raw)):
        err = 100.0 * (p - t_) / t_
        print(f"  {i+1:4d}  {p:14.6e}  {t_:14.6e}  {err:+8.2f}%")

    pred_raw = float(10 ** (final_pred_norm * std + mean))
    return final_pred_norm, pred_raw


# ─── Queue simulation ──────────────────────────────────────────────────────────

def _simulate_queue(iats_s: np.ndarray, mu: float, rng: np.random.Generator):
    """Single-server G/M/1 queue simulation via Lindley's recursion.

    Parameters
    ----------
    iats_s : (n,) inter-arrival times in seconds
    mu     : service rate (1/s); service times are i.i.d. Exp(mu)
    rng    : numpy random Generator (controls service times)

    Returns
    -------
    ql : (n,) int — queue length (including in-service customer) at each arrival
    A  : (n,) float — arrival times (seconds)
    B  : (n,) float — departure times (seconds)
    """
    n = len(iats_s)
    S = rng.exponential(1.0 / mu, size=n)   # i.i.d. service times
    A = np.cumsum(iats_s)                    # arrival times

    # Lindley's recursion: B[k] = max(A[k], B[k-1]) + S[k]
    B = np.empty(n, dtype=np.float64)
    B[0] = A[0] + S[0]
    for k in range(1, n):
        B[k] = (A[k] if A[k] >= B[k - 1] else B[k - 1]) + S[k]

    # Queue length when customer k arrives:
    #   = (customers arrived so far) − (customers departed before A[k])
    # B is non-decreasing and B[j] >= A[j] >= A[k] for j >= k, so
    # searchsorted over all of B correctly counts only j < k.
    n_departed = np.searchsorted(B, A, side="left")
    ql = (np.arange(1, n + 1, dtype=np.int32) - n_departed).clip(min=0)
    return ql, A, B


@torch.no_grad()
def _teacher_forcing_preds(model, train_data: np.ndarray, val_data: np.ndarray,
                            n: int, device: str = DEVICE):
    """Return (preds_norm, trues_norm) for n steps in log-normalised space.

    preds_norm[t] = model's prediction for val_data[t+1] given true history.
    trues_norm[t] = val_data[t+1]  (the ground truth).

    Keeping both in normalised space lets the caller compute residuals and
    add calibrated noise before inverse-transforming.
    """
    model.eval()
    tr = torch.FloatTensor(train_data).to(device)
    vl = torch.FloatTensor(val_data[: n + 1]).to(device)

    states = None
    for t in range(0, len(tr), CHUNK_SIZE):
        chunk_in = tr[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
        if chunk_in.shape[1] == 0:
            break
        _, states = model(chunk_in, states)

    preds_norm = []
    for t in range(0, n, CHUNK_SIZE):
        chunk_in = vl[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
        if chunk_in.shape[1] == 0:
            break
        pred, states = model(chunk_in, states)
        preds_norm.extend(pred.squeeze(0).cpu().tolist())

    preds_norm = np.array(preds_norm[:n])
    trues_norm = val_data[1 : n + 1]          # pred[t] targets val[t+1]
    return preds_norm, trues_norm


@torch.no_grad()
def _train_residuals(model, train_data: np.ndarray,
                     device: str = DEVICE) -> np.ndarray:
    """Teacher-forcing pass over training data; returns residuals in log-norm space.

    Runs from a cold start (no prior warm-up), so early residuals may be
    slightly inflated while the state accumulates history.  In practice the
    state settles within a few hundred steps, which is negligible relative to
    the ~800 k training samples.
    """
    model.eval()
    tr     = torch.FloatTensor(train_data).to(device)
    states = None
    preds  = []
    for t in range(0, len(tr) - 1, CHUNK_SIZE):
        chunk_in = tr[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
        if chunk_in.shape[1] == 0:
            break
        pred, states = model(chunk_in, states)
        n_known = min(pred.shape[1], len(tr) - t - 1)
        preds.extend(pred.squeeze(0)[:n_known].cpu().tolist())
    preds = np.array(preds)
    trues = train_data[1 : len(preds) + 1]
    return trues - preds   # ε_t = x_true[t+1] − pred[t]


@torch.no_grad()
def _bootstrap_train_iats(model, train_data: np.ndarray,
                           mean: float, std: float,
                           residuals: np.ndarray, n_gen: int,
                           noise_block_size: int, noise_scale: float,
                           seed: int, device: str = DEVICE) -> np.ndarray:
    """Generate n_gen IATs autoregressively, injecting block-bootstrapped
    training residuals at each step.

    Unlike teacher forcing this mode never sees validation data: it warms up
    on training data, then feeds each noisy prediction back as the next input.
    The noise is pre-sampled from the training residual distribution so the
    generated marginal matches what the model saw during training.
    """
    model.eval()
    tr = torch.FloatTensor(train_data).to(device)

    # Warm up state over the full training set
    states = None
    for t in range(0, len(tr), CHUNK_SIZE):
        chunk_in = tr[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
        if chunk_in.shape[1] == 0:
            break
        _, states = model(chunk_in, states)

    # Pre-sample all noise from training residuals
    rng   = np.random.default_rng(seed)
    noise = _block_bootstrap(residuals, n_gen, noise_block_size, rng) * noise_scale

    # Seed the autoregressive loop with the last training value
    x = torch.tensor([[[train_data[-1]]]], dtype=torch.float32, device=device)
    pred, states = model(x, states)

    iats_raw = np.empty(n_gen, dtype=np.float64)
    for i in range(n_gen):
        pred_norm   = pred[0, 0].item() + noise[i]
        iats_raw[i] = 10 ** (pred_norm * std + mean)
        x = torch.tensor([[[pred_norm]]], dtype=torch.float32, device=device)
        pred, states = model(x, states)

    return iats_raw


@torch.no_grad()
def _autoregressive_iats(model, train_data: np.ndarray, val_seed: float,
                          mean: float, std: float, n_gen: int,
                          noise_std: float = 0.0,
                          device: str = DEVICE) -> np.ndarray:
    """Autoregressively generate n_gen IATs, feeding each prediction back.

    Without noise (noise_std=0) this collapses to the conditional mean after
    a few steps (exposure-bias / teacher-forcing mismatch).  Set noise_std to
    the model's validation RMSE (in log-normalised space) to prevent collapse
    while preserving the marginal variance.
    """
    model.eval()
    tr = torch.FloatTensor(train_data).to(device)

    states = None
    for t in range(0, len(tr), CHUNK_SIZE):
        chunk_in = tr[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
        if chunk_in.shape[1] == 0:
            break
        _, states = model(chunk_in, states)

    x = torch.tensor([[[val_seed]]], dtype=torch.float32, device=device)
    pred, states = model(x, states)

    rng = np.random.default_rng(0)
    iats_raw = np.empty(n_gen, dtype=np.float64)
    for i in range(n_gen):
        pred_norm = pred[0, 0].item() + (rng.normal(0.0, noise_std) if noise_std else 0.0)
        iats_raw[i] = 10 ** (pred_norm * std + mean)
        x = torch.tensor([[[pred_norm]]], dtype=torch.float32, device=device)
        pred, states = model(x, states)

    return iats_raw


def _block_bootstrap(arr: np.ndarray, n: int, block_size: int,
                     rng: np.random.Generator) -> np.ndarray:
    """Draw n samples from arr using non-overlapping random blocks.

    Preserves short-to-medium-range autocorrelation in arr, which i.i.d.
    resampling would destroy.  Block boundaries are chosen uniformly at random;
    the final block is truncated to exactly n total samples.
    """
    m = len(arr)
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
               generation: str = "bootstrap_train", noise_std: float = 0.0,
               calibrate_noise: bool = True, noise_block_size: int = 500,
               noise_scale: float = 1.0,
               name: str = "Model", device: str = DEVICE, seed: int = 42):
    """Compare a single-server queue driven by trace IATs vs model IATs.

    Arrival process
    ---------------
    - Trace : actual raw IATs from raw_val[:n_customers]
    - Model : IATs produced by the model, controlled by `generation`:
        "teacher"       — model sees true IATs at each step (no compounding
                          error).  With calibrate_noise=True (default),
                          bootstrapped residuals are added so the generated
                          sequence matches the trace's marginal distribution,
                          not just its conditional mean.
        "autoregressive"  — each prediction is fed back as the next input.
                            Collapses to the conditional mean without noise;
                            set noise_std ≈ val RMSE (log-norm space) to prevent
                            this (e.g. noise_std=0.9 for a typical S4 run).
        "bootstrap_train" — autoregressive (never sees val data), but injects
                            block-bootstrapped residuals drawn from the training
                            set at each step.  Avoids exposure bias while keeping
                            the generated distribution grounded in training
                            statistics rather than validation ground truth.

    Why calibrate_noise matters
    ---------------------------
    An MSE-trained model predicts the conditional mean, which has far less
    variance than the true IAT distribution.  A queue driven by near-constant
    IATs behaves like M/D/1, not like the actual bursty process.  Bootstrap-
    resampling the empirical residuals ε_t = x_true - x_pred and adding them
    back recovers the correct marginal distribution while keeping the model's
    predicted ordering (conditional mean).  The autocorrelation structure is
    still approximated (residuals are resampled i.i.d.), but the marginal is
    exact by construction.

    Service process
    ---------------
    Exponential with rate mu (M service).  The same numpy random seed is used
    for both simulations so service times are identical.

    Parameters
    ----------
    model           : trained S4Predictor or S5Predictor
    train_data      : log-normalised training array (model warm-up)
    val_data        : log-normalised validation array
    mean, std       : log-normalisation parameters
    raw_val         : raw IATs (seconds) for the validation set
    mu              : service rate (1/s).  None → rho = 0.8 vs trace mean.
    n_customers     : customers to simulate (default 10 000)
    generation      : "teacher" (default), "autoregressive", or "bootstrap_train"
    noise_std       : std of noise injected in autoregressive mode (log-norm)
    calibrate_noise : add bootstrapped residuals in teacher mode (default True)
    name            : model label for output
    device          : torch device
    seed            : RNG seed shared between both simulations

    Returns
    -------
    ql_trace, ql_model : queue-length arrays (int, length n_customers)
    model_iats         : model IAT array (float64, seconds)
    """
    n = (len(raw_val) - 1) if n_customers is None else min(n_customers, len(raw_val) - 1)

    # ── Service rate ────────────────────────────────────────────────────────────
    lam_trace = 1.0 / float(raw_val[:n].mean())
    if mu is None:
        mu = lam_trace / 0.80
    rho_trace = lam_trace / mu

    mode_str = generation + ("+calibrated_noise" if generation == "teacher" and calibrate_noise else
                             f"+train_residuals(scale={noise_scale})" if generation == "bootstrap_train" else "")
    print(f"\n{'─'*60}")
    print(f"Queue test  ({name}, {mode_str})")
    print(f"  customers  : {n:,}")
    print(f"  mu         : {mu:.2f} /s   (mean service {1e3/mu:.2f} ms)")
    print(f"  λ (trace)  : {lam_trace:.2f} /s")
    print(f"  ρ (trace)  : {rho_trace:.3f}")
    print(f"{'─'*60}")

    # ── Trace simulation ────────────────────────────────────────────────────────
    rng_trace = np.random.default_rng(seed)
    ql_trace, _, _ = _simulate_queue(raw_val[:n], mu, rng_trace)

    # ── Model IAT generation ────────────────────────────────────────────────────
    print(f"  Generating {n:,} IATs ({mode_str}) …", flush=True)
    t0 = time.time()

    if generation == "teacher":
        preds_norm, trues_norm = _teacher_forcing_preds(
            model, train_data, val_data, n, device
        )
        if calibrate_noise:
            # Residuals ε_t = x_{t+1} - pred_t
            residuals = trues_norm - preds_norm
            rng_noise = np.random.default_rng(seed + 1)
            # Block bootstrap: sample contiguous blocks to preserve short-to-
            # medium-range autocorrelation (LRD of BCAUG89 means i.i.d. draws
            # produce too little queue variance).
            noise = _block_bootstrap(residuals, n, noise_block_size, rng_noise)
            noise = noise * noise_scale
            model_iats = 10 ** ((preds_norm + noise) * std + mean)
            residual_rmse = float(np.sqrt(np.mean(residuals ** 2)))
            print(f"  Residual RMSE (log-norm): {residual_rmse:.4f}  "
                  f"block_size: {noise_block_size}  noise_scale: {noise_scale}")
        else:
            model_iats = 10 ** (preds_norm * std + mean)
    elif generation == "bootstrap_train":
        train_res = _train_residuals(model, train_data, device)
        residual_rmse = float(np.sqrt(np.mean(train_res ** 2)))
        print(f"  Train residual RMSE (log-norm): {residual_rmse:.4f}  "
              f"block_size: {noise_block_size}  noise_scale: {noise_scale}")
        model_iats = _bootstrap_train_iats(
            model, train_data, mean, std, train_res,
            n, noise_block_size, noise_scale, seed + 1, device
        )
    else:
        model_iats = _autoregressive_iats(
            model, train_data, val_data[0], mean, std, n, noise_std, device
        )

    print(f"  Generated in {time.time() - t0:.1f} s")

    # ── Rate normalisation ──────────────────────────────────────────────────────
    # For heavy-tailed processes (BCAUG89) the arithmetic mean IAT is inflated
    # by rare long silences; model-generated IATs follow an approximately
    # log-normal marginal whose arithmetic mean is exp(μ_log + σ²/2) — smaller
    # than the trace arithmetic mean.  Without correction, λ_model > λ_trace
    # and ρ_model > 1 (unstable), even though the model was trained on a stable
    # process.  We rescale model IATs to share the trace arithmetic mean so the
    # comparison isolates distributional shape / autocorrelation, not rate bias.
    trace_mean_iat = float(raw_val[:n].mean())
    model_mean_iat = float(model_iats.mean())
    rate_scale = trace_mean_iat / model_mean_iat
    model_iats  = model_iats * rate_scale
    print(f"  Rate scale applied: {rate_scale:.4f}  "
          f"(model mean IAT {model_mean_iat*1e3:.3f} ms → "
          f"{trace_mean_iat*1e3:.3f} ms)")

    lam_model = 1.0 / float(model_iats.mean())   # == lam_trace after scaling
    rho_model = lam_model / mu

    # ── Model simulation (same service seed) ────────────────────────────────────
    rng_model = np.random.default_rng(seed)
    ql_model, _, _ = _simulate_queue(model_iats, mu, rng_model)

    # ── Statistics ──────────────────────────────────────────────────────────────
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

    # Wasserstein-1 between empirical queue-length distributions
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

    # ── Last 20 IAT comparison ──────────────────────────────────────────────────
    print(f"\n  Last 20 generated IATs vs trace")
    print(f"  {'#':>4}  {'Trace (s)':>12}  {'Model (s)':>12}  {'Err %':>8}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*12}  {'─'*8}")
    for i, (t_, m_) in enumerate(zip(raw_val[n - 20 : n], model_iats[n - 20 :])):
        err = 100.0 * (m_ - t_) / t_
        print(f"  {i + 1:4d}  {t_:12.4e}  {m_:12.4e}  {err:+8.2f}%")

    # ── CCDF plot: log(P(L > x)) vs log(x) ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))

    for ql, label, color in [
        (ql_trace, "Trace",          "steelblue"),
        (ql_model, f"Model ({name})", "darkorange"),
    ]:
        # Empirical CCDF at each unique queue-length value.
        # P(L > v) = #{L_i > v} / n  for each unique v.
        vals, counts = np.unique(ql, return_counts=True)
        cum = np.cumsum(counts)
        ccdf = (len(ql) - cum) / len(ql)   # P(L > vals[k])

        # Drop the last point (ccdf = 0, log undefined) and any vals = 0
        mask = (ccdf > 0) & (vals > 0)
        ax.plot(np.log10(vals[mask]), np.log10(ccdf[mask]),
                label=label, color=color, linewidth=1.5)

    ax.set_xlabel("log₁₀(x)")
    ax.set_ylabel("log₁₀(P(L > x))")
    ax.set_xlim(0, 6)
    ax.set_ylim(-6, 0)
    ax.set_title(f"Queue-length CCDF — {name}  (ρ={rho_trace:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = f"queue_ccdf_{name.lower()}.png"
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

    for name, fresh_model in [("S4", S4Predictor())]:  # , ("S5", S5Predictor())]:
        ckpt = None if retrain else load_model(name)
        if ckpt is not None:
            model, _, _ = ckpt   # mean/std from checkpoint match; use data's for consistency
        else:
            model = fit_sequential(fresh_model, train_data, val_data, name)
            save_model(model, name, mean, std)

        pred_norm, pred_raw = evaluate_and_predict(
            model, train_data, val_data, mean, std, raw, name
        )
        predictions[name] = (pred_norm, pred_raw)

        queue_test(model, train_data, val_data, mean, std, raw_val,
                   noise_scale=1.75, name=name)

    print(f"\n{'='*60}")
    print("PREDICTION: 1,000,000th inter-arrival time in BCAUG89")
    print(f"{'='*60}")
    print(f"Last known value (999,999th):  {raw[-1]:.6e} s")
    print()
    for name, (pn, pr) in predictions.items():
        print(f"  {name:4s}  →  {pr:.6e} s  (log-norm: {pn:+.4f})")
    print()


def save_model(model, name: str, mean: float, std: float):
    path = f"{name.lower()}_checkpoint.pt"
    torch.save({
        "model_state": model.state_dict(),
        "model_class": type(model).__name__,
        "mean": mean,
        "std":  std,
    }, path)
    print(f"  Saved checkpoint → {path}")


def load_model(name: str, device: str = DEVICE):
    """Load checkpoint if it exists.  Returns (model, mean, std) or None."""
    path = f"{name.lower()}_checkpoint.pt"
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device)
    cls  = {"S4Predictor": S4Predictor, "S5Predictor": S5Predictor}[ckpt["model_class"]]
    model = cls().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded checkpoint ← {path}")
    return model, ckpt["mean"], ckpt["std"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true",
                        help="Ignore saved checkpoints and retrain from scratch")
    args = parser.parse_args()
    main(retrain=args.retrain)
