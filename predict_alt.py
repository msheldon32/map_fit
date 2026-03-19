"""
S4 and S5 for arrival-count prediction on the BCAUG89 network traffic trace.

Uniformization approach: time is divided into fixed-width bins and the model
predicts the number of arrivals in the next bin given a short sliding window
of past bin counts.  This contrasts with the embedded-chain approach
(predict.py), which operates at event epochs and predicts inter-arrival times.

Training: TBPTT – the SSM hidden state is carried across chunk boundaries,
giving the model a running view of the MAP hidden state over the full trace.

S4 (S4D-Lin): SISO – one independent diagonal SSM per model channel.
S5          : MIMO – one shared diagonal SSM across all H channels.
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Configuration ─────────────────────────────────────────────────────────────
BIN_WIDTH  = 0.010   # time-bin width (seconds); ~3 arrivals/bin for BCAUG89
WINDOW     = 8       # display-only: bins of context shown in last-20 table
CHUNK_SIZE = 512     # TBPTT truncation length (in bins)
D_MODEL    = 16#32
D_STATE    = 4#16
N_LAYERS   = 2
D_FF       = 64#128
DROPOUT    = 0.0
EPOCHS     = 20
LR         = 1e-3
TRAIN_FRAC = 0.80
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH  = "BCAUG89"


# ─── Data ──────────────────────────────────────────────────────────────────────

def make_counts(path: str, bin_width: float) -> np.ndarray:
    """
    Load IATs, convert to absolute arrival times, bin into a count sequence:
    counts[i] = number of arrivals in [i·Δt, (i+1)·Δt).
    """
    iats   = np.loadtxt(path, dtype=np.float64)
    times  = np.cumsum(iats)
    n_bins = int(np.ceil(times[-1] / bin_width))
    counts = np.zeros(n_bins, dtype=np.float32)
    idxs   = np.clip(np.floor(times / bin_width).astype(np.int64), 0, n_bins - 1)
    np.add.at(counts, idxs, 1)
    return counts


# ─── State utilities ───────────────────────────────────────────────────────────

def detach_states(states):
    if states is None:
        return None
    return [(sr.detach(), si.detach()) for sr, si in states]


# ─── S4 – S4D-Lin (SISO × H independent channels) ────────────────────────────

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
        """x: (B, L, H), state: (sr,si) each (B,H,N) or None → (B,L,H), new_state"""
        B, L, H = x.shape
        dt = torch.exp(self.log_dt)

        A_real = -torch.exp(self.log_A_real)
        Abar_r = torch.exp(A_real * dt) * torch.cos(self.A_imag * dt)
        Abar_i = torch.exp(A_real * dt) * torch.sin(self.A_imag * dt)
        Bbar_r = dt * self.B_real
        Bbar_i = dt * self.B_imag

        sr, si = (x.new_zeros(B, H, self.d_state), x.new_zeros(B, H, self.d_state)) \
                 if state is None else state

        outs = []
        for t in range(L):
            u    = x[:, t]
            Bu_r = Bbar_r * u.unsqueeze(-1)
            Bu_i = Bbar_i * u.unsqueeze(-1)
            sr, si = (Abar_r * sr - Abar_i * si + Bu_r,
                      Abar_r * si + Abar_i * sr + Bu_i)
            outs.append((self.C_real * sr - self.C_imag * si).sum(-1) + self.D * u)

        return self.drop(self.out_proj(torch.stack(outs, dim=1))), (sr, si)


class S4Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ssm   = S4DLayer(d_model, d_state, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, state=None):
        ssm_out, new_state = self.ssm(self.norm1(x), state)
        x = x + ssm_out
        x = x + self.ff(self.norm2(x))
        return x, new_state


class S4Predictor(nn.Module):
    def __init__(self, d_model=D_MODEL, d_state=D_STATE,
                 n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.enc  = nn.Linear(1, d_model)
        self.body = nn.ModuleList([
            S4Block(d_model, d_state, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1), nn.Softplus()
        )

    def forward(self, x: torch.Tensor, states=None):
        """x: (B, L, 1) → pred: (B, L), new_states"""
        x = self.enc(x)
        new_states = []
        for i, blk in enumerate(self.body):
            x, new_st = blk(x, states[i] if states is not None else None)
            new_states.append(new_st)
        return self.head(self.norm(x)).squeeze(-1), new_states


# ─── S5 – Simplified S5 (MIMO shared state across all channels) ───────────────

class S5Layer(nn.Module):
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
        """x: (B, L, H), state: (sr,si) each (B,N) or None → (B,L,H), new_state"""
        B, L, H = x.shape
        dt     = torch.exp(self.log_dt)
        Lam_r  = -torch.exp(self.log_Lam_real)
        Lbar_r = torch.exp(Lam_r * dt) * torch.cos(self.Lam_imag * dt)
        Lbar_i = torch.exp(Lam_r * dt) * torch.sin(self.Lam_imag * dt)
        Bbar_r = dt * self.B_real
        Bbar_i = dt * self.B_imag

        sr, si = (x.new_zeros(B, self.d_state), x.new_zeros(B, self.d_state)) \
                 if state is None else state

        outs = []
        for t in range(L):
            u  = x[:, t]
            sr, si = (Lbar_r * sr - Lbar_i * si + u @ Bbar_r.T,
                      Lbar_r * si + Lbar_i * sr + u @ Bbar_i.T)
            outs.append((sr @ self.C_real.T - si @ self.C_imag.T) + self.D * u)

        return self.drop(self.out_proj(torch.stack(outs, dim=1))), (sr, si)


class S5Block(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ssm   = S5Layer(d_model, d_state, dropout)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x, state=None):
        ssm_out, new_state = self.ssm(self.norm1(x), state)
        x = x + ssm_out
        x = x + self.ff(self.norm2(x))
        return x, new_state


class S5Predictor(nn.Module):
    def __init__(self, d_model=D_MODEL, d_state=D_STATE,
                 n_layers=N_LAYERS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.enc  = nn.Linear(1, d_model)
        self.body = nn.ModuleList([
            S5Block(d_model, d_state, d_ff, dropout) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1), nn.Softplus()
        )

    def forward(self, x: torch.Tensor, states=None):
        x = self.enc(x)
        new_states = []
        for i, blk in enumerate(self.body):
            x, new_st = blk(x, states[i] if states is not None else None)
            new_states.append(new_st)
        return self.head(self.norm(x)).squeeze(-1), new_states


# ─── Training ──────────────────────────────────────────────────────────────────

def fit_sequential(model, train_counts: np.ndarray, val_counts: np.ndarray,
                   name: str, device: str = DEVICE):
    """
    TBPTT over the full count sequence.  State is detached at each chunk
    boundary but otherwise carried forward, so the model tracks the MAP
    hidden state across the entire trace without shuffling.
    Validation continues from the final training state of each epoch.
    """
    model  = model.to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'─'*60}")
    print(f"{name}  ({n_params:,} parameters)")
    print(f"{'─'*60}")

    tr = torch.FloatTensor(train_counts).to(device)
    vl = torch.FloatTensor(val_counts).to(device)

    best_val, best_state_dict = float("inf"), None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        # ── Training ───────────────────────────────────────────────────────
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
            pred, states = model(chunk_in, states)
            n    = min(pred.shape[1], chunk_tgt.shape[1])
            loss = F.poisson_nll_loss(pred[:, :n], chunk_tgt[:, :n], log_input=False)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item()
            n_chunks   += 1

        sched.step()

        # ── Validation (state carried from end of training) ────────────────
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
                val_loss  += F.poisson_nll_loss(
                    pred[:, :n], chunk_tgt[:, :n], log_input=False
                ).item()
                val_chunks += 1

        tr_l = train_loss / max(n_chunks,   1)
        vl_l = val_loss   / max(val_chunks, 1)

        if vl_l < best_val:
            best_val = vl_l
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 4 == 0 or ep == EPOCHS:
            print(f"  ep {ep:3d}/{EPOCHS}  "
                  f"train={tr_l:.5f}  val={vl_l:.5f}  "
                  f"best={best_val:.5f}  [{time.time()-t0:.0f}s]")

    model.load_state_dict(best_state_dict)
    model.eval()
    return model


# ─── Evaluation & prediction ───────────────────────────────────────────────────

def evaluate_and_predict(model, train_counts: np.ndarray, val_counts: np.ndarray,
                         name: str, device: str = DEVICE):
    """
    Final sequential pass with the best model:
      1. Warm up state over the training portion.
      2. Run through validation, collecting per-step (pred, true) pairs.
      3. Print the last 20 validation predictions.
      4. Return the prediction for the bin immediately after the trace ends.
    """
    tr = torch.FloatTensor(train_counts).to(device)
    vl = torch.FloatTensor(val_counts).to(device)

    model.eval()
    with torch.no_grad():
        # Warm up over training data
        states = None
        for t in range(0, len(tr), CHUNK_SIZE):
            chunk_in = tr[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
            if chunk_in.shape[1] == 0:
                break
            _, states = model(chunk_in, states)

        # Sequential pass over validation data
        val_preds = []
        val_trues = []
        final_pred = None

        for t in range(0, len(vl), CHUNK_SIZE):
            chunk_in = vl[t : t + CHUNK_SIZE].unsqueeze(0).unsqueeze(-1)
            if chunk_in.shape[1] == 0:
                break
            pred, states = model(chunk_in, states)
            pred = pred.squeeze(0)          # (C,)

            n_known = min(len(pred), len(vl) - t - 1)
            val_preds.extend(pred[:n_known].cpu().tolist())
            val_trues.extend(vl[t+1 : t+1+n_known].cpu().tolist())

            if t + chunk_in.shape[1] >= len(vl):
                final_pred = pred[-1].item()

    val_preds = np.array(val_preds)
    val_trues = np.array(val_trues)

    print(f"\n{name} – last 20 validation predictions")
    print(f"  {'#':>4}  {'Predicted':>12}  {'Actual':>8}  {'Err %':>8}")
    print(f"  {'─'*4}  {'─'*12}  {'─'*8}  {'─'*8}")
    for i, (p, t_) in enumerate(zip(val_preds[-20:], val_trues[-20:])):
        err = 100.0 * (p - t_) / (t_ + 1e-6)
        print(f"  {i+1:4d}  {p:12.4f}  {t_:8.0f}  {err:+8.2f}%")

    return final_pred


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"BIN_WIDTH={BIN_WIDTH*1000:.1f} ms  CHUNK_SIZE={CHUNK_SIZE} bins")
    print(f"Loading and binning {DATA_PATH} …")

    counts = make_counts(DATA_PATH, BIN_WIDTH)
    N      = len(counts)
    print(f"  {N:,} bins  |  mean={counts.mean():.3f}  "
          f"std={counts.std():.3f}  max={int(counts.max())}")

    n_train    = int(N * TRAIN_FRAC)
    train_data = counts[:n_train]
    val_data   = counts[n_train:]
    print(f"  Train bins: {n_train:,}   Val bins: {N - n_train:,}")

    predictions = {}

    for name, model in [("S4", S4Predictor()), ("S5", S5Predictor())]:
        model = fit_sequential(model, train_data, val_data, name)
        next_count = evaluate_and_predict(model, train_data, val_data, name)
        predictions[name] = next_count

    print(f"\n{'='*60}")
    print(f"PREDICTION: arrivals in next bin  "
          f"(bin width = {BIN_WIDTH*1000:.1f} ms)")
    print(f"{'='*60}")
    print(f"Last known bin count: {int(val_data[-1])}")
    print(f"Mean over val set:    {val_data.mean():.3f}")
    print()
    for name, pred in predictions.items():
        print(f"  {name:4s}  →  {pred:.4f} arrivals")
    print()


if __name__ == "__main__":
    main()
