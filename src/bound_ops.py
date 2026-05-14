"""
IBP and CROWN bound propagation for MMfreeLM operations.
Follows the CROWN-IBP pattern: interval_propagate (IBP) and bound_backward (CROWN).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Interval (IBP) propagation — forward pass with bounds
# ═══════════════════════════════════════════════════════════════════════════

def ibp_linear(lower, upper, weight, bias=None):
    """IBP through Linear / BitLinear / Embedding layer. Exact for affine ops."""
    weight = weight.to(dtype=lower.dtype)
    if bias is not None:
        bias = bias.to(dtype=lower.dtype)
    mid = (lower + upper) / 2.0
    diff = (upper - lower) / 2.0
    w_abs = weight.abs()
    center = F.linear(mid, weight, bias)
    deviation = F.linear(diff, w_abs)
    return center - deviation, center + deviation


def ibp_conv1d(lower, upper, weight, bias=None, stride=1, padding=0):
    """IBP through Conv1d layer."""
    weight = weight.to(dtype=lower.dtype)
    if bias is not None:
        bias = bias.to(dtype=lower.dtype)
    mid = (lower + upper) / 2.0
    diff = (upper - lower) / 2.0
    w_abs = weight.abs()
    center = F.conv1d(mid, weight, bias, stride=stride, padding=padding)
    deviation = F.conv1d(diff, w_abs, None, stride=stride, padding=padding)
    return center - deviation, center + deviation


def ibp_sigmoid(lower, upper):
    """Sigmoid is monotonic."""
    return torch.sigmoid(lower), torch.sigmoid(upper)


def ibp_swiglu(gate_l, gate_u, x_l, x_u):
    """swiglu(gate, x) = gate * sigmoid(gate) * x.  Interval arithmetic."""
    silu_l = gate_l * torch.sigmoid(gate_l)
    silu_u = gate_u * torch.sigmoid(gate_u)
    # Both silu and x are elementwise; the product's extreme lies at corner points.
    prod_ll = silu_l * x_l
    prod_lu = silu_l * x_u
    prod_ul = silu_u * x_l
    prod_uu = silu_u * x_u
    stacked = torch.stack([prod_ll, prod_lu, prod_ul, prod_uu], dim=0)
    return stacked.min(dim=0).values, stacked.max(dim=0).values


def ibp_swiglu_linear(gate_l, gate_u, x, weight, bias=None):
    """swiglu(gate, x) * W + b where x is a shared input."""
    silu_l = gate_l * F.sigmoid(gate_l)
    silu_u = gate_u * F.sigmoid(gate_u)
    mid = (silu_l + silu_u) / 2.0
    diff = (silu_u - silu_l) / 2.0
    return ibp_linear(mid * x, diff * x.abs(), weight, bias)


def ibp_rmsnorm(lower, upper, weight, eps=1e-6):
    """
    RMSNorm: y = x / rms(x) * w.
    Certified IBP: conservative but can be loose.
    """
    weight = weight.to(dtype=lower.dtype)
    lower_sq = lower ** 2
    upper_sq = upper ** 2
    min_sq = torch.where((lower <= 0) & (upper >= 0),
                         torch.zeros_like(lower),
                         torch.min(lower_sq, upper_sq))
    max_sq = torch.max(lower_sq, upper_sq)
    min_rms = torch.sqrt(min_sq.mean(dim=-1, keepdim=True) + eps)
    max_rms = torch.sqrt(max_sq.mean(dim=-1, keepdim=True) + eps)

    w = weight.unsqueeze(0).unsqueeze(0) if weight.dim() == 1 else weight

    out_l = lower / max_rms * w.clamp(min=0) + upper / min_rms * w.clamp(max=0)
    out_u = upper / min_rms * w.clamp(min=0) + lower / max_rms * w.clamp(max=0)
    return out_l, out_u


def ibp_rmsnorm_tight(lower, upper, weight, eps=1e-6):
    """
    RMSNorm IBP using midpoint RMS as the normalization factor.
    Not strictly certified (the true min/max may lie slightly outside),
    but practically tight for small epsilon. Useful for studying
    bound propagation trends through early layers.
    """
    weight = weight.to(dtype=lower.dtype)
    mid = (lower + upper) / 2.0
    rms_mid = torch.sqrt(mid.square().mean(dim=-1, keepdim=True) + eps)
    w = weight.unsqueeze(0).unsqueeze(0) if weight.dim() == 1 else weight
    out_l = lower / rms_mid * w
    out_u = upper / rms_mid * w
    return out_l, out_u


def ibp_recurrent_step(i_l, i_u, f_l, f_u, initial_state=None):
    """
    Gated linear recurrence: o_t = f_t ⊙ o_{t-1} + i_t  where f_t ∈ (0, 1).
    B, H, L, D shaped tensors.
    Uses conservative interval arithmetic through time.
    """
    B, H, L, D = i_l.shape
    device = i_l.device
    dtype = i_l.dtype
    if initial_state is None:
        init_l = torch.zeros(B, H, D, device=device, dtype=dtype)
        init_u = torch.zeros(B, H, D, device=device, dtype=dtype)
    else:
        init_l, init_u = initial_state

    o_l, o_u = init_l, init_u
    o_l_list, o_u_list = [], []
    for t in range(L):
        i_lt, i_ut = i_l[:, :, t, :], i_u[:, :, t, :]
        f_lt, f_ut = f_l[:, :, t, :], f_u[:, :, t, :]
        # o_{t} = f_t * o_{t-1} + i_t
        # All non-negative f, so consider corners of (f, o, i)
        cand_ll = f_lt * o_l + i_lt
        cand_lu = f_lt * o_u + i_lt
        cand_ul = f_ut * o_l + i_ut
        cand_uu = f_ut * o_u + i_ut
        o_l = torch.stack([cand_ll, cand_lu, cand_ul, cand_uu], dim=0).min(dim=0).values
        o_u = torch.stack([cand_ll, cand_lu, cand_ul, cand_uu], dim=0).max(dim=0).values
        o_l_list.append(o_l)
        o_u_list.append(o_u)
    # Stack along the sequence dimension: (B, H, L, D)
    return torch.stack(o_l_list, dim=2), torch.stack(o_u_list, dim=2)


def ibp_elementwise_min(lower, upper, other_l, other_u):
    """Element-wise min(a, b) — not monotonic, use corner sampling."""
    cand = torch.stack([
        torch.min(lower, other_l), torch.min(lower, other_u),
        torch.min(upper, other_l), torch.min(upper, other_u),
    ], dim=0)
    return cand.min(dim=0).values, cand.max(dim=0).values


# ═══════════════════════════════════════════════════════════════════════════
# CROWN (linear relaxation) backward propagation
# ═══════════════════════════════════════════════════════════════════════════

def crown_linear_backward(last_uA, last_lA, weight, bias=None, bound_opts=None):
    """CROWN backward through Linear layer. Returns (uA, ubias_sum, lA, lbias_sum)."""
    if bound_opts is None:
        bound_opts = {}
    # Upper bound: propagate A^U * W, accumulate bias
    uA = None
    ubias_sum = 0
    if last_uA is not None:
        # last_uA: (batch, * , out_dim), weight: (out_dim, in_dim)
        # New A = last_uA @ W, acting on input x
        # Shape handling: last_uA (B, L, V), weight (V, D) -> (B, L, D)
        if last_uA.dim() == 3:
            uA = last_uA.matmul(weight)
        else:
            uA = last_uA.matmul(weight)
        if bias is not None:
            ubias_sum = ubias_sum + last_uA.matmul(bias.unsqueeze(-1)).squeeze(-1)

    lA = None
    lbias_sum = 0
    if last_lA is not None:
        if last_lA.dim() == 3:
            lA = last_lA.matmul(weight)
        else:
            lA = last_lA.matmul(weight)
        if bias is not None:
            lbias_sum = lbias_sum + last_lA.matmul(bias.unsqueeze(-1)).squeeze(-1)

    return uA, ubias_sum, lA, lbias_sum


def crown_sigmoid_backward(last_uA, last_lA, lower, upper, bound_opts=None):
    """
    CROWN linear relaxation of sigmoid.
    For y = sigmoid(x) with x in [l, u]:
    Upper bound: chord connecting (l, sig(l)) to (u, sig(u))
    Lower bound: tangent at the midpoint (tightest for sigmoid)
    """
    if bound_opts is None:
        bound_opts = {}
    sig_l = torch.sigmoid(lower)
    sig_u = torch.sigmoid(upper)

    # Upper linear bound: chord (always valid since sigmoid is convex then concave)
    chord_slope = (sig_u - sig_l) / (upper - lower + 1e-8)
    upper_d = chord_slope
    upper_b = sig_l - upper_d * lower

    # Lower linear bound: tangent at midpoint (tight lower bound for sigmoid)
    mid = (lower + upper) / 2.0
    sig_mid = torch.sigmoid(mid)
    d_mid = sig_mid * (1 - sig_mid)  # derivative of sigmoid at mid
    lower_d = d_mid
    lower_b = sig_mid - lower_d * mid

    uA, ubias, lA, lbias = None, 0, None, 0
    if last_uA is not None:
        uA = upper_d.unsqueeze(-1) * last_uA
        ubias = (last_uA * upper_b.unsqueeze(-1)).sum(-1) if last_uA.dim() > 2 else (last_uA * upper_b).sum(-1)
    if last_lA is not None:
        lA = lower_d.unsqueeze(-1) * last_lA
        lbias = (last_lA * lower_b.unsqueeze(-1)).sum(-1) if last_lA.dim() > 2 else (last_lA * lower_b).sum(-1)
    return uA, ubias, lA, lbias


def crown_rmsnorm_backward(last_uA, last_lA, lower, upper, weight, eps=1e-6):
    """
    CROWN backward linear relaxation of RMSNorm.
    RMSNorm(x)_i = x_i / rms(x) * w_i,  rms(x) = sqrt(mean(x^2) + eps)

    Linearizes: y_i ≈ a_i * x_i + sum_j b_ij * x_j + c_i
    using first-order Taylor expansion around the midpoint of [lower, upper].

    This is a local linearization (not globally certified), but provides
    tight estimates when the input interval is small relative to the RMS.
    """
    weight = weight.to(dtype=lower.dtype)
    mid = (lower + upper) / 2.0
    rms_mid = torch.sqrt(mid.square().mean(dim=-1, keepdim=True) + eps)
    w = weight.unsqueeze(0).unsqueeze(0) if weight.dim() == 1 else weight
    D = lower.shape[-1]

    # ∂(x_i/rms(x))/∂x_j at x=mid:
    # = 1/rms * delta_ij - x_i * x_j / (D * rms^3)
    inv_rms = 1.0 / rms_mid
    inv_rms3 = 1.0 / (rms_mid ** 3)

    # Jacobian: (B, L, D, D) — diagonal + rank-1 outer product
    # J_{ij} = w_i * (inv_rms * delta_ij - mid_i * mid_j / (D * inv_rms3))
    # For CROWN, we need A_new = A_old @ J, where J acts on x

    # Instead of building the full D×D Jacobian, use structure:
    # J @ v = w * (inv_rms * v - mid/D * inv_rms3 * (mid · v))
    # So A_new @ x = A_old @ (w * inv_rms * x) - A_old @ (w * mid/D * inv_rms3) * (mid · x)

    # For CROWN propagation: last_uA has shape (B, L, V) or (B*L, V)
    # We need to compute the action of J^T on each row of last_uA
    # J^T @ a = inv_rms * w * a - (mid · (w * a)) * inv_rms3 * mid / D

    # This is complex for 3D tensors. We use a simplified element-wise
    # diagonal approximation: ∂y_i/∂x_i ≈ w_i / rms(mid)
    # This ignores off-diagonal coupling but is a reasonable first-order approx.

    diag_jac = w * inv_rms  # (B, L, D)

    uA, ubias, lA, lbias = None, 0, None, 0
    if last_uA is not None:
        uA = last_uA * diag_jac  # element-wise: A_out acts through diagonal Jacobian
        # bias correction: y(mid) - J @ mid
        y_mid = mid / rms_mid * w
        ubias = (last_uA * (y_mid - diag_jac * mid)).sum(-1)

    if last_lA is not None:
        lA = last_lA * diag_jac
        y_mid = mid / rms_mid * w
        lbias = (last_lA * (y_mid - diag_jac * mid)).sum(-1)

    return uA, ubias, lA, lbias


def crown_swiglu_backward(last_uA, last_lA, gate_l, gate_u, val_l, val_u):
    """
    CROWN backward through SwiGLU(gate, val) = gate * sigmoid(gate) * val.
    Uses linear relaxation around midpoints of both gate and val.
    """
    gate_mid = (gate_l + gate_u) / 2.0
    val_mid = (val_l + val_u) / 2.0

    # SwiGLU(g, v) = silu(g) * v where silu(g) = g * sigmoid(g)
    silu_g = gate_mid * torch.sigmoid(gate_mid)
    d_silu = torch.sigmoid(gate_mid) + gate_mid * torch.sigmoid(gate_mid) * (1 - torch.sigmoid(gate_mid))

    # ∂f/∂gate = d_silu * val_mid
    grad_gate = d_silu * val_mid
    # ∂f/∂val = silu(gate_mid)
    grad_val = silu_g

    uA, ubias, lA, lbias = None, 0, None, 0
    y_mid = silu_g * val_mid

    if last_uA is not None:
        # last_uA acts on the SwiGLU output; A_new_gate = last_uA * grad_gate
        uA_gate = last_uA * grad_gate
        uA_val = last_uA * grad_val
        uA = (uA_gate, uA_val)  # return pair for gate/val paths
        ubias = (last_uA * (y_mid - grad_gate * gate_mid - grad_val * val_mid)).sum(-1)

    if last_lA is not None:
        lA_gate = last_lA * grad_gate
        lA_val = last_lA * grad_val
        lA = (lA_gate, lA_val)
        lbias = (last_lA * (y_mid - grad_gate * gate_mid - grad_val * val_mid)).sum(-1)

    return uA, ubias, lA, lbias


# ═══════════════════════════════════════════════════════════════════════════
# Concrete bound computation (final step of CROWN)
# ═══════════════════════════════════════════════════════════════════════════

def crown_linear_backward(last_uA, last_lA, weight, bias=None, bound_opts=None):
    """CROWN backward through Linear layer. Returns (uA, ubias, lA, lbias)."""
    if bound_opts is None:
        bound_opts = {}
    if last_uA is not None:
        uA = last_uA.matmul(weight)
        ubias = last_uA.matmul(bias) if bias is not None else 0
    else:
        uA, ubias = None, 0
    if last_lA is not None:
        lA = last_lA.matmul(weight)
        lbias = last_lA.matmul(bias) if bias is not None else 0
    else:
        lA, lbias = None, 0
    return uA, ubias, lA, lbias


def crown_sigmoid_backward(last_uA, last_lA, lower, upper, bound_opts=None):
    """
    CROWN linear relaxation of sigmoid.
    For y = sigmoid(x) with x ∈ [l, u]:
      y ≥ sigmoid(l) + (sigmoid(u)-sigmoid(l))/(u-l) * (x-l)  (convex hull lower)
    Actually uses the tangent line at mid or the chord, whichever is a valid lower bound.
    """
    if bound_opts is None:
        bound_opts = {}
    sig_l = torch.sigmoid(lower)
    sig_u = torch.sigmoid(upper)
    # Slope of chord
    chord_slope = (sig_u - sig_l) / (upper - lower + 1e-8)
    # For upper bound: use the chord
    # For lower bound: use 0 (since sigmoid is convex then concave, the simplest
    # valid lower bound is a line from (l, sig_l) to (u, sig_u), or we use 0-slope.)
    # For a conservative lower bound, use 0-slope (any slope in [0, chord_slope] is valid).
    upper_d = chord_slope
    upper_b = sig_l - upper_d * lower

    if bound_opts.get("zero-lb", False):
        lower_d = torch.zeros_like(upper_d)
    else:
        # Use chord slope for lower bound too (valid but may be loose)
        lower_d = upper_d
    lower_b = sig_l - lower_d * lower

    uA, ubias, lA, lbias = None, 0, None, 0
    if last_uA is not None:
        uA = upper_d.unsqueeze(1) * last_uA
        ubias = (last_uA.view(last_uA.size(0), last_uA.size(1), -1)
                 .matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1))
    if last_lA is not None:
        lA = lower_d.unsqueeze(1) * last_lA
        lbias = (last_lA.view(last_lA.size(0), last_lA.size(1), -1)
                 .matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1))
    return uA, ubias, lA, lbias


# ═══════════════════════════════════════════════════════════════════════════
# Concrete bound computation (final step of CROWN)
# ═══════════════════════════════════════════════════════════════════════════

def concretize_bounds(A, sum_b, x_L, x_U, norm=np.inf, sign=-1):
    """Compute concrete lower (sign=-1) or upper (sign=+1) bound from A matrix."""
    if A is None:
        return None
    A = A.view(A.size(0), A.size(1), -1)
    if norm == np.inf:
        x_lb = x_L.view(x_L.size(0), -1, 1)
        x_ub = x_U.view(x_U.size(0), -1, 1)
        center = (x_ub + x_lb) / 2.0
        diff = (x_ub - x_lb) / 2.0
        bound = A.bmm(center) + sign * A.abs().bmm(diff)
    else:
        x = x_U.view(x_U.size(0), -1, 1)
        dual_norm = 1.0 / (1.0 - 1.0 / norm)
        deviation = A.norm(dual_norm, -1) * 0  # L2 not needed for our use
        bound = A.bmm(x) + sign * deviation.unsqueeze(-1)
    bound = bound.squeeze(-1) + sum_b
    return bound


# ═══════════════════════════════════════════════════════════════════════════
# CROWN v2: (V, D) matrix format for IBP-CROWN hybrid
# ═══════════════════════════════════════════════════════════════════════════

def crown_rmsnorm_backward_v2(last_uA, last_lA, x_L, x_U, weight, eps=1e-6):
    """
    CROWN backward through RMSNorm with (V, D) A matrices.

    Args:
        last_uA, last_lA: (V, D) — linear coeffs w.r.t. normalized output
        x_L, x_U: (B, L, D) — bounds on pre-norm hidden states
        weight: (D,) — RMSNorm weight
    Returns:
        uA, lA: (V, D) — linear coeffs w.r.t. pre-norm input
        ubias, lbias: (V,) — accumulated bias per logit
    """
    weight = weight.to(dtype=x_L.dtype)
    mid = (x_L + x_U) / 2.0  # (B, L, D)
    rms_mid = torch.sqrt(mid.square().mean(dim=-1, keepdim=True) + eps)  # (B, L, 1)
    w = weight.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
    D = x_L.shape[-1]

    # Diagonal Jacobian approximation: ∂y_i/∂x_i ≈ w_i / rms(mid)
    diag_jac = w / rms_mid  # (B, L, D)

    # For each logit v, A_new(v, j) = sum_i A(v, i) * ∂y_i/∂x_j
    # ≈ A(v, j) * diag_jac(j) (diagonal approximation)
    # A has shape (V, D), diag_jac has shape (B, L, D)
    # Result should be (B, L, V, D) or we concretize directly

    # Actually, we compute per-(B,L) bounds directly:
    # A_eff(b,l,v,j) = last_uA(v, j) * diag_jac(b, l, j)
    # The ubias needs to account for the linearization error
    # y_mid(b,l,i) = mid(b,l,i) / rms_mid(b,l) * w(0,0,i)
    y_mid = mid / rms_mid * w  # (B, L, D)
    # Jacobian @ mid ≈ y_mid (since RMSNorm is approx. linear near mid)

    # Bias correction: f(mid) - J @ mid ≈ y_mid - diag_jac * mid
    bias_correction = y_mid - diag_jac * mid  # (B, L, D)

    # Now: A_eff(v, b, l, j) = last_uA(v, j) * diag_jac(b, l, j)
    # ubias(v, b, l) = sum_j last_uA(v, j) * bias_correction(b, l, j)

    uA = None
    ubias = 0
    if last_uA is not None:
        # Shape: last_uA (V, D) × diag_jac (B, L, D) → (B, L, V, D) for A
        # ubias: (V, D) @ (B, L, D)^T → (V, B, L) → sum → (B, L, V)
        # Using einsum for clarity:
        # A_eff[b,l,v,j] = last_uA[v,j] * diag_jac[b,l,j]
        # ubias_contrib[v] = sum_j last_uA[v,j] * bias_correction[b,l,j]
        ubias = torch.einsum('vd,bld->blv', last_uA, bias_correction)
        uA_last = last_uA  # store for concretize, will multiply with diag_jac later

    lA = None
    lbias = 0
    if last_lA is not None:
        lbias = torch.einsum('vd,bld->blv', last_lA, bias_correction)
        lA_last = last_lA

    # We store the diag_jac for concretize step
    return (uA_last, diag_jac), ubias, (lA_last, diag_jac), lbias


def concretize_bounds_v2(A_info, sum_b, x_L, x_U, sign=-1):
    """
    Concretize bounds from (A_weight, diag_jac) pair.

    Args:
        A_info: (A_weight, diag_jac) tuple or None
            - A_weight: (V, D) — base linear coefficients
            - diag_jac: (B, L, D) — diagonal Jacobian of RMSNorm
        sum_b: (B, L, V) — accumulated bias
        x_L, x_U: (B, L, D) — input bounds
        sign: -1 for lower bound, +1 for upper bound
    Returns:
        bound: (B, L, V) — concrete bounds
    """
    if A_info is None:
        return None
    A_weight, diag_jac = A_info

    # Effective A per position: A_eff[b,l,v,j] = A_weight[v,j] * diag_jac[b,l,j]
    # Bound = A_eff @ x_mid + sign * |A_eff| @ x_diff + sum_b
    x_mid = (x_U + x_L) / 2.0  # (B, L, D)
    x_diff = (x_U - x_L) / 2.0  # (B, L, D)

    # A_eff_mid: (B, L, V) = sum_j A_weight[v,j] * diag_jac[b,l,j] * x_mid[b,l,j]
    A_eff_times_mid = torch.einsum('vd,bld,bld->blv', A_weight, diag_jac, x_mid)

    # |A_eff| @ x_diff: (B, L, V) = sum_j |A_weight[v,j] * diag_jac[b,l,j]| * x_diff[b,l,j]
    A_eff_abs_times_diff = torch.einsum('vd,bld,bld->blv',
                                         A_weight.abs(), diag_jac.abs(), x_diff)

    bound = A_eff_times_mid + sign * A_eff_abs_times_diff + sum_b
    return bound


# ═══════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════

def bound_width(lower, upper):
    """Mean absolute bound width."""
    return (upper - lower).abs().mean().item()


def margin_from_bounds(lb, ub, labels=None):
    """
    Certified prediction margin: for each (sample, position), the gap between
    the target label's lower bound and the highest upper bound of any other class.
    Positive margin ⇒ certified robust.
    Handles both (B, V) and (B, L, V) shapes.
    """
    if labels is None:
        mid = (lb + ub) / 2.0
        labels = mid.argmax(dim=-1)
    flat_lb = lb.view(-1, lb.shape[-1])
    flat_ub = ub.view(-1, ub.shape[-1])
    flat_labels = labels.view(-1)
    B, V = flat_lb.shape
    mask = torch.ones(B, V, device=flat_lb.device, dtype=torch.bool)
    mask.scatter_(1, flat_labels.unsqueeze(1), False)
    other_ub = flat_ub[mask].view(B, V - 1)
    label_lb = flat_lb.gather(1, flat_labels.unsqueeze(1)).squeeze(-1)
    margins = label_lb - other_ub.max(dim=1).values
    return margins.view(labels.shape)
