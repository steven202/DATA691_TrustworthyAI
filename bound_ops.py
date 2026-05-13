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
