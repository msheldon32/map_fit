"""
predict_matrix.py – Direct MAP fitting via gradient descent through the forward algorithm.

D0 governs transitions without arrivals (hidden transitions during inter-arrival times).
D1 governs transitions with arrivals (visible transitions at event epochs).

The MAP log-likelihood over a sequence of IATs τ_1, …, τ_T is:

    log L = Σ_t  log( α_{t-1} @ expm(D0 · τ_t) @ D1 @ 1 )

where α_{t-1} is the posterior state distribution after the (t-1)-th arrival.
Backpropagating through this sum is equivalent to running the MAP backward
algorithm: the gradient w.r.t. D0 and D1 involves the backward variables
β_t = D1 @ expm(D0 · τ_{t+1}) @ … @ 1, exactly as in Baum-Welch.

Training uses TBPTT: α is detached at chunk boundaries but carried forward,
so the model maintains a running posterior over the MAP hidden state.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Configuration ─────────────────────────────────────────────────────────────
N_STATES   = 4       # MAP state space size
CHUNK_SIZE = 512     # TBPTT truncation length (inter-arrival times per chunk)
EPOCHS     = 20
LR         = 1e-3
TRAIN_FRAC = 0.80
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH  = "BCAUG89"


# ─── Data ──────────────────────────────────────────────────────────────────────

def load_data(path: str):
    raw      = np.loadtxt(path, dtype=np.float64).astype(np.float32)
    mean_iat = float(raw.mean())
    # Normalise so mean IAT = 1; rates in D0/D1 are then O(1)
    normed   = raw / mean_iat
    return normed, mean_iat


# ─── MAP Model ─────────────────────────────────────────────────────────────────

class MAPModel(nn.Module):
    """
    Learnable MAP(n_states) parametrised so that all constraints hold by
    construction:

      D1_ij  ≥ 0  (arrival transitions, softplus-parametrised)
      D0_ij  ≥ 0  for i ≠ j  (hidden transitions, softplus-parametrised)
      (D0 + D1) @ 1 = 0  (generator constraint)

    The diagonal of D0 is not a free parameter: it is set to
      D0_ii = -(Σ_{j≠i} D0_ij + Σ_j D1_ij)
    so rows of Q = D0 + D1 always sum to zero and D0_ii ≤ 0.
    """

    def __init__(self, n_states: int):
        super().__init__()
        self.n_states = n_states

        self.raw_D1         = nn.Parameter(torch.randn(n_states, n_states) * 0.5)
        self.raw_D0_offdiag = nn.Parameter(torch.randn(n_states, n_states) * 0.5)
        self.log_pi         = nn.Parameter(torch.zeros(n_states))

    def get_D0_D1(self):
        n      = self.n_states
        device = self.raw_D1.device

        D1          = F.softplus(self.raw_D1)
        D0_offdiag  = F.softplus(self.raw_D0_offdiag)
        D0_offdiag  = D0_offdiag * (1 - torch.eye(n, device=device))  # zero diagonal
        diag        = -(D0_offdiag.sum(-1) + D1.sum(-1))              # ≤ 0
        D0          = D0_offdiag + torch.diag(diag)

        return D0, D1

    def initial_alpha(self, device):
        return F.softmax(self.log_pi, dim=0).to(device)


# ─── Forward Algorithm ─────────────────────────────────────────────────────────

def map_forward_chunk(model: MAPModel, iats: torch.Tensor, alpha: torch.Tensor,
                      collect_preds: bool = False):
    """
    MAP forward algorithm over one chunk of inter-arrival times.

    For each IAT τ_t the update is:
        c_t    = α_{t-1} @ expm(D0 · τ_t) @ D1 @ 1   (scalar likelihood)
        α_t    = (α_{t-1} @ expm(D0 · τ_t) @ D1) / c_t

    Summing log c_t gives the chunk log-likelihood.  Autograd through this
    sum computes the same quantities as the MAP backward algorithm.

    When collect_preds=True the function also returns, for each step t,
    the predicted mean IAT *before* observing τ_t:
        E[τ_t | α_{t-1}] = α_{t-1} @ (-D0)^{-1} @ 1

    Args:
        model:         MAPModel with current parameters
        iats:          (T,) normalised IATs for this chunk
        alpha:         (n,) state distribution entering the chunk (detached)
        collect_preds: whether to compute per-step mean-IAT predictions

    Returns:
        nll:    per-IAT negative log-likelihood (scalar, differentiable)
        alpha:  (n,) state distribution after the chunk (detached)
        preds:  list of T predicted mean IATs in normalised time (or None)
    """
    D0, D1 = model.get_D0_D1()
    T      = len(iats)

    # Batch-compute expm(D0 · τ_t) for all t in the chunk in one call
    expm_D0 = torch.linalg.matrix_exp(
        D0.unsqueeze(0) * iats.view(-1, 1, 1)   # (T, n, n)
    )
    M = expm_D0 @ D1                             # (T, n, n): full per-step transition

    if collect_preds:
        # E[τ | state = i] = [(-D0)^{-1} @ 1]_i  (expected first-passage time)
        ones           = torch.ones(model.n_states, device=D0.device)
        mean_per_state = torch.linalg.solve(-D0, ones)   # (n,)

    log_ll = iats.new_zeros(1)
    preds  = [] if collect_preds else None

    for t in range(T):
        if collect_preds:
            # Predict before observing iats[t]
            preds.append((alpha @ mean_per_state).item())

        alpha  = (alpha @ M[t]).clamp(min=0)     # (n,) guard fp negatives
        c      = alpha.sum().clamp(min=1e-30)
        log_ll = log_ll + torch.log(c)
        alpha  = alpha / c

    return -log_ll / T, alpha.detach(), preds


# ─── Training ──────────────────────────────────────────────────────────────────

def fit(model: MAPModel, train_iats: np.ndarray, val_iats: np.ndarray,
        device: str = DEVICE):
    model    = model.to(device)
    opt      = torch.optim.Adam(model.parameters(), lr=LR)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nMAP({model.n_states})  ({n_params} parameters)")
    print(f"{'─'*60}")

    tr = torch.FloatTensor(train_iats).to(device)
    vl = torch.FloatTensor(val_iats).to(device)

    best_val, best_state_dict = float("inf"), None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        # ── Training ───────────────────────────────────────────────────────
        model.train()
        alpha      = model.initial_alpha(device)
        train_loss = 0.0
        n_chunks   = 0

        for t in range(0, len(tr) - 1, CHUNK_SIZE):
            chunk = tr[t : t + CHUNK_SIZE]
            if len(chunk) < 2:
                break

            alpha = alpha.detach()
            opt.zero_grad()
            nll, alpha, _ = map_forward_chunk(model, chunk, alpha)
            nll.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += nll.item()
            n_chunks   += 1

        sched.step()

        # ── Validation (state carried from end of training) ────────────────
        model.eval()
        val_alpha  = alpha.detach()
        val_loss   = 0.0
        val_chunks = 0

        with torch.no_grad():
            for t in range(0, len(vl) - 1, CHUNK_SIZE):
                chunk = vl[t : t + CHUNK_SIZE]
                if len(chunk) < 2:
                    break
                nll, val_alpha, _ = map_forward_chunk(model, chunk, val_alpha)
                val_loss  += nll.item()
                val_chunks += 1

        tr_l = train_loss / max(n_chunks,   1)
        vl_l = val_loss   / max(val_chunks, 1)

        if vl_l < best_val:
            best_val        = vl_l
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 4 == 0 or ep == EPOCHS:
            print(f"  ep {ep:3d}/{EPOCHS}  "
                  f"train={tr_l:.5f}  val={vl_l:.5f}  "
                  f"best={best_val:.5f}  [{time.time()-t0:.0f}s]")

    model.load_state_dict(best_state_dict)
    model.eval()
    return model


# ─── Evaluation & prediction ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_and_predict(model: MAPModel, train_iats: np.ndarray,
                         val_iats: np.ndarray, mean_iat: float,
                         device: str = DEVICE):
    """
    Final sequential pass with the best model.

    preds[t] = E[τ_t | α_{t-1}] = α_{t-1} @ (-D0)^{-1} @ 1  (before observing τ_t)
    so predictions and actuals align directly (no off-by-one shift needed).

    The prediction for the IAT *after* the last known arrival is computed from
    the final α after processing all of val_iats.
    """
    tr = torch.FloatTensor(train_iats).to(device)
    vl = torch.FloatTensor(val_iats).to(device)

    # Warm up over training data
    alpha = model.initial_alpha(device)
    for t in range(0, len(tr), CHUNK_SIZE):
        chunk = tr[t : t + CHUNK_SIZE]
        if len(chunk) == 0:
            break
        _, alpha, _ = map_forward_chunk(model, chunk, alpha)

    # Val pass collecting per-step predictions
    val_preds = []
    val_trues = []

    for t in range(0, len(vl), CHUNK_SIZE):
        chunk = vl[t : t + CHUNK_SIZE]
        if len(chunk) == 0:
            break
        _, alpha, preds = map_forward_chunk(model, chunk, alpha, collect_preds=True)
        # preds[k] predicts vl[t+k] (no shift: prediction made before observing iats[t+k])
        val_preds.extend(preds)
        val_trues.extend(chunk.cpu().tolist())

    val_preds = np.array(val_preds) * mean_iat
    val_trues = np.array(val_trues) * mean_iat

    print(f"\nLast 20 validation predictions")
    print(f"  {'#':>4}  {'Predicted':>14}  {'Actual':>14}  {'Err %':>8}")
    print(f"  {'─'*4}  {'─'*14}  {'─'*14}  {'─'*8}")
    for i, (p, t_) in enumerate(zip(val_preds[-20:], val_trues[-20:])):
        err = 100.0 * (p - t_) / (t_ + 1e-10)
        print(f"  {i+1:4d}  {p:14.6e}  {t_:14.6e}  {err:+8.2f}%")

    # Prediction for the IAT after the last known arrival
    D0, _ = model.get_D0_D1()
    ones           = torch.ones(model.n_states, device=D0.device)
    mean_per_state = torch.linalg.solve(-D0, ones)
    next_iat_norm  = (alpha @ mean_per_state).item()
    next_iat       = next_iat_norm * mean_iat

    return next_iat


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")
    print(f"Loading {DATA_PATH} …")

    iats, mean_iat = load_data(DATA_PATH)
    N = len(iats)

    print(f"  {N:,} inter-arrival times")
    print(f"  Raw mean IAT: {mean_iat:.4e} s  (normalised mean = 1.0)")

    n_train    = int(N * TRAIN_FRAC)
    train_iats = iats[:n_train]
    val_iats   = iats[n_train:]
    print(f"  Train: {n_train:,}   Val: {N - n_train:,}")

    model    = MAPModel(N_STATES)
    model    = fit(model, train_iats, val_iats)
    next_iat = evaluate_and_predict(model, train_iats, val_iats, mean_iat)

    # Report learned parameters
    D0, D1 = model.get_D0_D1()
    D0_s   = (D0.cpu().numpy() / mean_iat)   # convert back to 1/s
    D1_s   = (D1.cpu().numpy() / mean_iat)
    pi     = F.softmax(model.log_pi, dim=0).cpu().detach().numpy()

    print(f"\n{'='*60}")
    print(f"PREDICTION: 1,000,000th inter-arrival time")
    print(f"{'='*60}")
    print(f"  Last known IAT:        {val_iats[-1] * mean_iat:.6e} s")
    print(f"  MAP({N_STATES}) prediction:   {next_iat:.6e} s")
    print()
    print(f"Learned MAP({N_STATES}) (rates in 1/s):")
    print(f"  π = {pi.round(4)}")
    print(f"  Per-state arrival rates (row sums of D1): "
          f"{D1_s.sum(axis=1).round(1)}")
    print(f"  Per-state total rates   (diag of -D0):    "
          f"{(-D0_s.diagonal()).round(1)}")
    print()


if __name__ == "__main__":
    main()
