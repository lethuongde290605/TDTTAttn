import torch
from typing import List, Tuple, Optional

# -------------------------
# Utility helpers
# -------------------------
def unfold(tensor, mode):
    """Unfold tensor along `mode` -> shape (I_mode, -1)."""
    order = [mode] + [i for i in range(tensor.ndim) if i != mode]
    perm = tensor.permute(order)
    return perm.reshape(tensor.shape[mode], -1)

def fold(matrix, mode, shape):
    """Fold matrix (I_mode, prod(other dims)) back to tensor with `shape`."""
    full_order = [mode] + [i for i in range(len(shape)) if i != mode]
    perm_inv = [full_order.index(i) for i in range(len(shape))]
    tensor = matrix.reshape([shape[i] for i in full_order])
    return tensor.permute(perm_inv)

def n_mode_product(tensor, matrix, mode):
    """
    n-mode product: tensor ×_mode matrix
    matrix shape: (J, I_mode) if we want to reduce I_mode -> J
    or (I_mode, R) if we want to expand (we will use (R, I) for transposed multiplies).
    """
    res = torch.tensordot(tensor, matrix, dims=([mode], [1]))
    last = res.ndim - 1
    perm = list(range(res.ndim))
    desired = perm[:mode] + [last] + perm[mode:last]
    return res.permute(desired)

def compute_core(X, factors):
    """Compute core G = X ×_1 U1^T ×_2 U2^T ..."""
    G = X
    for mode, U in enumerate(factors):
        G = n_mode_product(G, U.T, mode)
    return G

def reconstruct(core, factors):
    """Reconstruct X_hat = G ×_1 U1 ×_2 U2 ..."""
    X_hat = core
    for mode, U in enumerate(factors):
        X_hat = n_mode_product(X_hat, U, mode)
    return X_hat

def rel_fro_loss(X, X_hat):
    return torch.norm(X - X_hat) / torch.norm(X)

# -------------------------
# HOSVD initializer
# -------------------------
def hosvd_init(X, ranks):
    """Return factors list computed by truncated SVD of each mode unfolding."""
    factors = []
    for mode, r in enumerate(ranks):
        Xn = unfold(X, mode)
        U, S, Vh = torch.linalg.svd(Xn, full_matrices=False)
        factors.append(U[:, :r].contiguous())
    core = compute_core(X, factors)
    return core, factors

# -------------------------
# HOOI (ALS) implementation
# -------------------------
def hooi(X, ranks, n_iter_max=10, tol=1e-6, verbose=False):
    """
    HOOI algorithm (ALS) to compute Tucker decomposition.
    """
    N = X.ndim - 1
    # initialize by HOSVD
    core, factors = hosvd_init(X, ranks)
    X_hat = reconstruct(core, factors)
    loss = float(rel_fro_loss(X, X_hat))
    history = [loss]
    if verbose:
        print(f"[init] loss={loss:.6f}")

    for it in range(n_iter_max):
        prev_factors = [U.clone() for U in factors]
        # update each mode
        for n in range(N):
            # build tensor projected on all modes except n
            Gtilde = X
            for k in range(N):
                if k == n:
                    continue
                Gtilde = n_mode_product(Gtilde, factors[k].T, k)
            # unfold along mode n and take top-r left singular vectors
            Gn = unfold(Gtilde, n)
            U, S, Vh = torch.linalg.svd(Gn, full_matrices=False)
            r_n = ranks[n]
            factors[n] = U[:, :r_n].contiguous()

        # recompute core and loss
        core = compute_core(X, factors)
        X_hat = reconstruct(core, factors)
        loss_new = float(rel_fro_loss(X, X_hat))
        history.append(loss_new)

        # compute relative improvement
        rel_change = abs(history[-2] - history[-1]) / (history[-2] + 1e-12)
        if verbose and it > 0 and n_iter_max % (it * 10) == 0:
            print(f"[iter {it+1}] loss={loss_new:.6e}, rel_change={rel_change:.3e}")
        if rel_change < tol:
            break

    return core, factors, history

def mode_matricize(tensor: torch.Tensor, mode: int) -> torch.Tensor:
    """Unfold / matricize a tensor along `mode` (0-based)."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")

    ndim = tensor.ndim
    if ndim == 0:
        raise ValueError("tensor must have at least 1 dimension")

    # normalize negative mode
    if mode < 0:
        mode = ndim + mode

    if mode < 0 or mode >= ndim:
        raise IndexError(f"mode out of range (got mode={mode} for tensor with ndim={ndim})")

    mode_dim = tensor.size(mode)

    # permutation: put `mode` first, keep the other axes in their original order
    perm = [mode] + [i for i in range(ndim) if i != mode]
    tensor_perm = tensor.permute(*perm).contiguous()

    mat = tensor_perm.reshape(mode_dim, -1)
    return mat

def choose_rank(S: torch.Tensor, eps: float = 0.9) -> int:
    """
    Chọn số chiều r nhỏ nhất sao cho năng lượng giữ lại ≥ eps.
    """
    S_sorted, _ = torch.sort(S, descending=True)
    singular_sq = S_sorted ** 2
    total_energy = singular_sq.sum()
    if total_energy == 0:
        return S.shape[0]
    energy_cumsum = torch.cumsum(singular_sq, dim=0)
    energy_ratio = energy_cumsum / total_energy

    idx = torch.where(energy_ratio >= eps)[0]
    if len(idx) == 0:
        return S.shape[0]
    return idx[0].item() + 1

def mps_decomposition(X: torch.Tensor, eps: float = 0.9):
    """MPS decomposition for 3D tensor."""
    dims = X.shape
    Bs = []
    X = mode_matricize(X, 0)

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    delta_1 = choose_rank(S, eps)

    Bs.append(U[:, :delta_1].reshape(1, dims[0], delta_1))
    S = S[:delta_1]
    S = torch.diag(S)
    U = U[:, :delta_1]
    Vh = Vh[:delta_1, :]

    X = torch.matmul(S, Vh)

    X = X.reshape(delta_1 * dims[1], -1)
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    delta_2 = choose_rank(S, eps)

    Cs = []
    Cs.append(Vh[:delta_2, :].reshape(delta_2, dims[2], 1))
    S = S[:delta_2]
    S = torch.diag(S)
    U = U[:, :delta_2]

    X = torch.matmul(U, S)

    X = X.reshape(delta_1, dims[1], delta_2)

    G = Bs + [X] + Cs

    return G

def combined_mps_hooi_compression(
    X: torch.Tensor,
    mps_eps: float = 0.99,
    hooi_ranks: Optional[List[int]] = None,
    hooi_input_shape: Tuple[int, int, int, int] = (8, 8, 12, 100),
    n_iter_max: int = 100,
    tol: float = 1e-7,
    verbose: bool = False
) -> torch.Tensor:
    """
    Perform combined MPS and HOOI compression on a square input tensor (768x768).
    
    Returns:
        Compressed representation reshaped back to (768, -1).
    """
    # Validation
    if X.ndim != 2:
        raise ValueError("Input tensor must have 2 dimensions.")
    if X.shape[0] != X.shape[1] or X.shape[0] != 768:
        raise ValueError("Input tensor must have shape (768, 768).")

    if hooi_ranks is None:
        hooi_ranks = [6, 6, 8]

    if verbose:
        print(f"[INFO] Input shape: {X.shape}")
        print(f"[INFO] MPS epsilon: {mps_eps}")
        print(f"[INFO] HOOI ranks: {hooi_ranks}")

    # 1. MPS Decomposition
    tensor = X.reshape(64, 64, 144)
    G = mps_decomposition(tensor, eps=mps_eps)
    if verbose:
        print(f"[MPS] Got {len(G)} cores with shapes: {[g.shape for g in G]}")

    # 2. HOOI Decomposition
    img_tensor_hooi = torch.randn(*hooi_input_shape, device=X.device, dtype=X.dtype)
    core_hooi, factors_hooi, hist = hooi(
        img_tensor_hooi,
        ranks=hooi_ranks,
        n_iter_max=n_iter_max,
        tol=tol,
        verbose=verbose
    )

    # 3. Reshape MPS cores
    delta_1 = G[0].shape[2]
    delta_2 = G[1].shape[2]

    G[0] = G[0].reshape(1, 8, 8, delta_1)
    G[1] = G[1].reshape(delta_1, 8, 8, delta_2)
    G[2] = G[2].reshape(delta_2, 12, 12, 1)

    if verbose:
        print(f"[MPS reshape] Core shapes: {[g.shape for g in G]}")

    # 4. Transpose HOOI factors
    factors_T = [f.T for f in factors_hooi]

    # 5. Tensor contractions
    result_G0 = torch.tensordot(G[0], factors_T[0], dims=([1], [1]))
    result_G1 = torch.tensordot(G[1], factors_T[1], dims=([1], [1]))
    result_G2 = torch.tensordot(G[2], factors_T[2], dims=([1], [1]))

    if verbose:
        print(f"[Contract] Shapes after tensordot: "
              f"G0={result_G0.shape}, G1={result_G1.shape}, G2={result_G2.shape}")

    # 6. Reconstruct combined tensor
    reconstructed = torch.tensordot(result_G0, result_G1, dims=([2], [0]))
    reconstructed = reconstructed.squeeze(0)
    reconstructed = torch.tensordot(reconstructed, result_G2, dims=([3], [0]))
    reconstructed = reconstructed.squeeze(-2)

    if verbose:
        print(f"[Reconstruct] Final tensor shape: {reconstructed.shape}")
        print(f"[Reconstruct] Expected to be 3D, will flatten to 2D matrix")

    # 7. Flatten back to matrix form
    # The reconstructed tensor has compressed structure
    # We need to reshape it properly to [out_features, in_features] for Linear layer
    out_dim = reconstructed.shape[0] * reconstructed.shape[1]  # compressed dimension
    in_dim = reconstructed.shape[2]  # should be related to input dimension
    
    result = reconstructed.reshape(out_dim, in_dim)
    
    if verbose:
        print(f"[Final] Reshaped to: {result.shape}")
    
    return result
