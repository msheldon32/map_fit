"""
predict_matrix_diagonal.py – MAP fitting with eigendecomposition of D0.

D0 is parametrised as:

    D0 = V · diag(λ) · V⁻¹,   λ_k = -softplus(raw_λ_k) < 0

which lets us compute the matrix exponential analytically:

    expm(D0 · τ) = V · diag(exp(λ · τ)) · V⁻¹

using only elementwise exp.  The gradient of expm w.r.t. λ_k is

    d/dλ_k  exp(λ_k · τ) = τ · exp(λ_k · τ)

— a simple scalar; no Fréchet derivative of matrix_exp is needed.

Relaxation
----------
The strict MAP sub-generator constraint (D0 off-diagonal ≥ 0) is NOT
enforced.  Only stability (λ_k < 0) and the generator row-sum property
(D0 + D1) @ 1 = 0 are encouraged; the latter via a soft penalty in the
training loss, controlled by GEN_LAMBDA.

Everything else (forward algorithm, TBPTT, prediction) is identical to
predict_matrix.py.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Configuration ─────────────────────────────────────────────────────────────
N_STATES   = 4       # MAP state space size
CHUNK_SIZE = 512     # TBPTT truncation length
EPOCHS     = 20
LR         = 1e-3
GEN_LAMBDA  = 0.1    # weight for generator row-sum penalty
COND_LAMBDA = 0.01   # weight for V condition-number penalty (log|det V|)
TRAIN_FRAC = 0.80
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
DATA_PATH  = "BCAUG89"


# ─── Data ──────────────────────────────────────────────────────────────────────

def load_data(path: str):
    raw      = np.loadtxt(path, dtype=np.float64).astype(np.float32)
    mean_iat = float(raw.mean())
    normed   = raw / mean_iat
    return normed, mean_iat


# ─── MAP Model (eigendecomposition) ────────────────────────────────────────────

class MAPModelDiag(nn.Module):
    """
    MAP(n_states) with D0 parametrised via its real eigendecomposition:

        D0 = V · diag(λ) · V⁻¹,   λ_k = -softplus(raw_λ_k) < 0

    V is an unconstrained n×n learnable matrix; V⁻¹ is computed via
    torch.linalg.inv each forward pass.

    The matrix exponential is computed as:

        expm(D0 · τ) = V · diag(exp(λ · τ)) · V⁻¹

    Gradient of expm w.r.t. λ_k: τ · exp(λ_k · τ)  (no Fréchet derivative).

    D1_ij ≥ 0 via softplus (same as predict_matrix.py).

    Relaxations:
      - D0 off-diagonal entries are not constrained ≥ 0.
      - Generator constraint (D0+D1)@1=0 is a soft penalty, not hard.
    """

    def __init__(self, n_states: int):
        super().__init__()
        self.n_states = n_states

        # Eigenvalues: λ_k = -softplus(raw_lambda_k), so λ_k < 0 always
        self.raw_lambda = nn.Parameter(torch.randn(n_states) * 0.5)

        # Eigenvector matrix — initialised near identity for a stable start
        self.V = nn.Parameter(
            torch.eye(n_states) + 0.1 * torch.randn(n_states, n_states)
        )

        # D1 (arrival transitions, non-negative via softplus)
        self.raw_D1 = nn.Parameter(torch.randn(n_states, n_states) * 0.5)

        # Initial state distribution
        self.log_pi = nn.Parameter(torch.zeros(n_states))

    def get_matrices(self):
        """
        Returns (D0, D1, V, V_inv, lam).

          lam   : (n,) negative eigenvalues of D0
          V     : (n,n) eigenvector matrix
          V_inv : (n,n) = torch.linalg.inv(V)
          D0    : (n,n) = V @ diag(lam) @ V_inv
          D1    : (n,n) non-negative arrival matrix
        """
        n     = self.n_states
        lam   = -F.softplus(self.raw_lambda)                    # (n,) all < 0
        # Small ridge prevents exact singularity during inv; COND_LAMBDA
        # penalty in the loss prevents V drifting toward near-singularity.
        V_stab = self.V + 1e-4 * torch.eye(n, device=self.V.device)
        V_inv  = torch.linalg.inv(V_stab)                      # (n,n)
        D0     = V_stab @ torch.diag(lam) @ V_inv              # (n,n)
        D1     = F.softplus(self.raw_D1)                        # (n,n) ≥ 0
        return D0, D1, V_stab, V_inv, lam

    def initial_alpha(self, device):
        return F.softmax(self.log_pi, dim=0).to(device)


# ─── Forward Algorithm ─────────────────────────────────────────────────────────

def map_forward_chunk(model: MAPModelDiag, iats: torch.Tensor,
                      alpha: torch.Tensor, collect_preds: bool = False):
    """
    MAP forward algorithm using eigendecomposition for expm.

    Batch-computes expm(D0 · τ_t) for all T steps at once:

        exp_lam[t, k] = exp(λ_k · τ_t)                     (T, n)
        expm_D0[t]    = (V * exp_lam[t].unsqueeze(0)) @ V⁻¹  (T, n, n)
                      = V · diag(exp(λ·τ_t)) · V⁻¹

    No call to torch.linalg.matrix_exp — gradients w.r.t. λ are trivial.

    Also returns gen_penalty = mean((D0+D1)@1)², which should → 0 if the
    generator constraint is satisfied.

    Returns:
        nll:         per-IAT negative log-likelihood (scalar, differentiable)
        alpha:       (n,) posterior state distribution (detached)
        preds:       list of T mean-IAT predictions (normalised), or None
        gen_penalty: scalar generator row-sum penalty
    """
    D0, D1, V, V_inv, lam = model.get_matrices()
    T = len(iats)

    # ── Batch expm via eigendecomposition ──────────────────────────────────────
    # exp_lam[t, k] = exp(λ_k · τ_t)
    exp_lam = torch.exp(lam.unsqueeze(0) * iats.unsqueeze(-1))          # (T, n)

    # (V.unsqueeze(0) * exp_lam.unsqueeze(1))[t, i, k] = V[i,k] * exp(λ_k·τ_t)
    expm_D0 = (V.unsqueeze(0) * exp_lam.unsqueeze(1)) @ V_inv.unsqueeze(0)
    #                                                                    (T, n, n)
    M = expm_D0 @ D1                                                  # (T, n, n)

    # Generator row-sum violation (soft penalty target)
    gen_penalty = ((D0 + D1).sum(dim=-1) ** 2).mean()

    if collect_preds:
        ones           = torch.ones(model.n_states, device=D0.device)
        mean_per_state = torch.linalg.solve(-D0, ones)                  # (n,)

    log_ll = iats.new_zeros(1)
    preds  = [] if collect_preds else None

    for t in range(T):
        if collect_preds:
            preds.append((alpha @ mean_per_state).item())

        alpha  = (alpha @ M[t]).clamp(min=0)     # guard fp negatives
        c      = alpha.sum().clamp(min=1e-30)
        log_ll = log_ll + torch.log(c)
        alpha  = alpha / c

    # V condition penalty: penalise small |det V| (near-singularity)
    # logdet of V^T V = 2 log|det V|; we maximise it, so add -logdet to loss
    V_cond_pen = -torch.logdet(V @ V.T).clamp(min=-20.0)

    return -log_ll / T, alpha.detach(), preds, gen_penalty, V_cond_pen


# ─── Training ──────────────────────────────────────────────────────────────────

def fit(model: MAPModelDiag, train_iats: np.ndarray, val_iats: np.ndarray,
        device: str = DEVICE):
    model    = model.to(device)
    opt      = torch.optim.Adam(model.parameters(), lr=LR)
    sched    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\nMAP-Diag({model.n_states})  ({n_params} parameters)")
    print(f"  GEN_LAMBDA = {GEN_LAMBDA}")
    print(f"{'─'*60}")

    tr = torch.FloatTensor(train_iats).to(device)
    vl = torch.FloatTensor(val_iats).to(device)

    best_val, best_state = float("inf"), None
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        # ── Training ───────────────────────────────────────────────────────────
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
            nll, alpha, _, gen_pen, cond_pen = map_forward_chunk(model, chunk, alpha)
            loss = nll + GEN_LAMBDA * gen_pen + COND_LAMBDA * cond_pen
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += nll.item()
            n_chunks   += 1

        sched.step()

        # ── Validation ─────────────────────────────────────────────────────────
        model.eval()
        val_alpha  = alpha.detach()
        val_loss   = 0.0
        val_chunks = 0

        with torch.no_grad():
            for t in range(0, len(vl) - 1, CHUNK_SIZE):
                chunk = vl[t : t + CHUNK_SIZE]
                if len(chunk) < 2:
                    break
                nll, val_alpha, _, _, _ = map_forward_chunk(model, chunk, val_alpha)
                val_loss   += nll.item()
                val_chunks += 1

        tr_l = train_loss / max(n_chunks,   1)
        vl_l = val_loss   / max(val_chunks, 1)

        if vl_l < best_val:
            best_val  = vl_l
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 4 == 0 or ep == EPOCHS:
            print(f"  ep {ep:3d}/{EPOCHS}  "
                  f"train={tr_l:.5f}  val={vl_l:.5f}  "
                  f"best={best_val:.5f}  [{time.time()-t0:.0f}s]")

    model.load_state_dict(best_state)
    model.eval()
    return model


# ─── Evaluation & prediction ───────────────────────────────────────────────────

@torch.no_grad()
def evaluate_and_predict(model: MAPModelDiag, train_iats: np.ndarray,
                         val_iats: np.ndarray, mean_iat: float,
                         device: str = DEVICE):
    tr = torch.FloatTensor(train_iats).to(device)
    vl = torch.FloatTensor(val_iats).to(device)

    # Warm-up over training data
    alpha = model.initial_alpha(device)
    for t in range(0, len(tr), CHUNK_SIZE):
        chunk = tr[t : t + CHUNK_SIZE]
        if len(chunk) == 0:
            break
        _, alpha, _, _, _ = map_forward_chunk(model, chunk, alpha)

    # Val pass collecting per-step predictions
    val_preds, val_trues = [], []
    for t in range(0, len(vl), CHUNK_SIZE):
        chunk = vl[t : t + CHUNK_SIZE]
        if len(chunk) == 0:
            break
        _, alpha, preds, _, _ = map_forward_chunk(model, chunk, alpha,
                                                  collect_preds=True)
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

    # Prediction for the next IAT after the last observed arrival
    D0, _, _, _, _ = model.get_matrices()
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

    model    = MAPModelDiag(N_STATES)
    model    = fit(model, train_iats, val_iats)
    next_iat = evaluate_and_predict(model, train_iats, val_iats, mean_iat)

    # Report learned parameters
    D0, D1, V, V_inv, lam = model.get_matrices()
    pi    = F.softmax(model.log_pi, dim=0).cpu().detach().numpy()
    lam_s = lam.cpu().detach().numpy() / mean_iat        # back to 1/s
    gen_viol = (D0 + D1).sum(dim=-1).abs().max().item()

    print(f"\n{'='*60}")
    print(f"PREDICTION: 1,000,000th inter-arrival time")
    print(f"{'='*60}")
    print(f"  Last known IAT:            {val_iats[-1] * mean_iat:.6e} s")
    print(f"  MAP-Diag({N_STATES}) prediction: {next_iat:.6e} s")
    print()
    print(f"Learned MAP-Diag({N_STATES}):")
    print(f"  π              = {pi.round(4)}")
    print(f"  D0 eigenvalues (1/s): {lam_s.round(2)}")
    print(f"  Max |generator row-sum|: {gen_viol:.4e}  "
          f"(0 = perfect constraint)")
    print()


if __name__ == "__main__":
    main()
