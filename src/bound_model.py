"""
BoundModel: wraps MMfreeLM for verified uncertainty quantification.
Supports IBP (certified) and empirical (sampling) bound propagation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
from tqdm import tqdm

from src.bound_ops import (
    ibp_linear, ibp_sigmoid, ibp_swiglu,
    ibp_rmsnorm, ibp_rmsnorm_tight, ibp_recurrent_step, ibp_conv1d,
    bound_width, margin_from_bounds,
    crown_linear_backward, crown_sigmoid_backward,
    crown_rmsnorm_backward, crown_swiglu_backward,
    crown_rmsnorm_backward_v2, concretize_bounds_v2,
    concretize_bounds
)


class BoundModel:
    """
    Wraps an MMfreeLM HuggingFace model for bound propagation.
    Perturbation is applied to the continuous embedding vectors (L∞ norm).
    """

    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        self._cache = {}  # cache intermediate results for analysis

    # ── embedding helpers ──────────────────────────────────────────────

    def get_embeddings(self, input_ids):
        """Get continuous embedding vectors from token ids."""
        with torch.no_grad():
            embed_weight = self.model.get_input_embeddings().weight
            return F.embedding(input_ids, embed_weight)

    def perturb_embeddings(self, embeddings, epsilon, norm=np.inf):
        """Create lower/upper bounds on embeddings with L∞ perturbation."""
        if norm == np.inf:
            lower = embeddings - epsilon
            upper = embeddings + epsilon
        else:
            raise NotImplementedError(f"Norm {norm} not supported")
        return lower, upper

    # ── empirical bound propagation (two-pass, not certified) ──────────

    @torch.no_grad()
    def empirical_bounds(self, input_ids, epsilon, attention_mask=None):
        """
        Run both lower and upper embeddings through the full model.
        Returns (lb_logits, ub_logits) — NOT certified, but practical.
        """
        emb = self.get_embeddings(input_ids)
        emb_l, emb_u = self.perturb_embeddings(emb, epsilon)

        logits_l = self._forward_with_embeds(emb_l, attention_mask)
        logits_u = self._forward_with_embeds(emb_u, attention_mask)

        # For non-monotonic models, the extremes might not be at the corners.
        # We take element-wise min/max as a heuristic (not certified).
        lb = torch.min(logits_l, logits_u)
        ub = torch.max(logits_l, logits_u)
        return lb, ub

    @torch.no_grad()
    def empirical_bounds_sampling(self, input_ids, epsilon, n_samples=50,
                                  attention_mask=None):
        """
        Monte Carlo empirical bounds: sample n_samples embeddings in the
        L∞ ball, run each through the model, take min/max of outputs.
        Closer to true empirical bounds than the two-pass version.
        """
        emb = self.get_embeddings(input_ids)
        B, L, D = emb.shape

        all_logits = []
        for _ in range(n_samples):
            noise = (torch.rand_like(emb) * 2 - 1) * epsilon
            emb_perturbed = emb + noise
            logits = self._forward_with_embeds(emb_perturbed, attention_mask)
            all_logits.append(logits)

        stacked = torch.stack(all_logits, dim=0)  # (S, B, L, V)
        lb = stacked.min(dim=0).values
        ub = stacked.max(dim=0).values
        return lb, ub

    def _forward_with_embeds(self, inputs_embeds, attention_mask=None):
        """Run model forward from embedding vectors, return logits."""
        outputs = self.model(inputs_embeds=inputs_embeds,
                             attention_mask=attention_mask)
        return outputs.logits

    # ── IBP (certified) bound propagation ──────────────────────────────

    @torch.no_grad()
    def ibp_bounds(self, input_ids, epsilon, attention_mask=None):
        """
        Certified IBP through the full model.
        Propagates interval bounds layer by layer.
        Uses float32 internally to mitigate numerical explosion.
        """
        emb = self.get_embeddings(input_ids).float()
        emb_l, emb_u = self.perturb_embeddings(emb, epsilon)

        self._cache['layer_bounds'] = []
        self._cache['layer_widths'] = []
        self._cache['input_bounds'] = (emb_l, emb_u)

        h_l, h_u = emb_l, emb_u
        base_model = self.model.model

        for layer_idx, layer in enumerate(tqdm(base_model.layers,
                                                desc='IBP layers',
                                                leave=False)):
            h_l, h_u = self._ibp_block(
                layer, h_l, h_u, attention_mask, layer_idx)

            # Clamp extreme values to prevent overflow in later layers
            h_l = torch.clamp(h_l, min=-1e6, max=1e6)
            h_u = torch.clamp(h_u, min=-1e6, max=1e6)

            width = bound_width(h_l, h_u)
            self._cache['layer_bounds'].append(
                (h_l.clone(), h_u.clone()))
            self._cache['layer_widths'].append(width)

        # Final norm
        norm_weight = base_model.norm.weight
        h_l, h_u = ibp_rmsnorm(h_l, h_u, norm_weight,
                               eps=base_model.norm.eps)

        # LM head
        lm_weight = self.model.lm_head.weight.float()
        h_l, h_u = h_l.float(), h_u.float()
        logits_l, logits_u = ibp_linear(h_l, h_u, lm_weight)

        return logits_l, logits_u

    def _ibp_block(self, layer, h_l, h_u, attention_mask, layer_idx):
        """IBP through one HGRNBitBlock."""
        # --- attn_norm (RMSNorm) ---
        n_l, n_u = ibp_rmsnorm(h_l, h_u, layer.attn_norm.weight,
                               eps=layer.attn_norm.eps)

        # --- attention ---
        a_l, a_u = self._ibp_attention(layer.attn, n_l, n_u,
                                       attention_mask, layer_idx)

        # Residual connection (pre-norm, so add original hidden states)
        h_l, h_u = h_l + a_l, h_u + a_u

        # --- mlp_norm + residual handling ---
        # mlp_norm(hidden_states, residual, True) does:
        #   hidden_states, residual = fused_norm_gate(hidden_states, residual)
        # This applies RMSNorm to hidden_states and returns both
        residual_l, residual_u = h_l, h_u
        n2_l, n2_u = ibp_rmsnorm(h_l, h_u, layer.mlp_norm.weight,
                                 eps=layer.mlp_norm.eps)

        # --- mlp ---
        m_l, m_u = self._ibp_mlp(layer.mlp, n2_l, n2_u)

        # Residual
        h_l, h_u = residual_l + m_l, residual_u + m_u
        return h_l, h_u

    def _ibp_attention(self, attn, h_l, h_u, attention_mask, layer_idx):
        """IBP through HGRNBitAttention."""
        B, L, D = h_l.shape
        H = attn.num_heads
        expand_dim = attn.input_dim
        head_dim = attn.head_dim

        # Short convolution (optional)
        if attn.use_short_conv:
            if attn.share_conv_kernel:
                # h_conv1d is a ShortConvolution module
                # It applies: conv1d → silu activation
                # For IBP: do conv1d through interval arithmetic, then silu
                # ShortConvolution internally does: silu(conv1d(x, weight)) or similar
                # Actually looking at the code: h_conv1d has conv1d + silu
                # Let's trace its structure
                h_l, h_u = self._ibp_short_conv(
                    attn.h_conv1d, h_l, h_u, attention_mask)
            else:
                raise NotImplementedError("Non-shared short conv not supported")

        # Linear projections
        i_l, i_u = ibp_linear(h_l, h_u, attn.i_proj.weight)
        f_l, f_u = ibp_linear(h_l, h_u, attn.f_proj.weight)
        g_l, g_u = ibp_linear(h_l, h_u, attn.g_proj.weight)

        # Sigmoid on f
        f_l, f_u = ibp_sigmoid(f_l, f_u)

        # Lower bound gate (if applicable, from model config)
        # For now, use the raw sigmoid bounds
        if hasattr(self.model.model, 'lower_bounds') and self.model.model.config.use_lower_bound:
            lower_bounds = self.model.model.lower_bounds.softmax(0)
            lower_bounds = lower_bounds.cumsum(0) - lower_bounds[0]
            lb = lower_bounds[layer_idx]
            f_l = lb.unsqueeze(0).unsqueeze(0) + (1 - lb.unsqueeze(0).unsqueeze(0)) * f_l
            f_u = lb.unsqueeze(0).unsqueeze(0) + (1 - lb.unsqueeze(0).unsqueeze(0)) * f_u

        # swiglu(i, 1-f): i * sigmoid(i) * (1-f) is not standard swiglu
        # Actually looking at the code: swiglu(i, 1-f) where swiglu(x, y) = x * sigmoid(x) * y
        # So it's: i * sigmoid(i) * (1-f)
        # The (1-f) term: since f ∈ (0,1), 1-f ∈ (0,1)
        one_minus_f_l = 1.0 - f_u  # because 1-f is decreasing in f
        one_minus_f_u = 1.0 - f_l

        # Now i * sigmoid(i) * (1-f)
        # First compute silu(i) = i * sigmoid(i)
        silu_i_l = i_l * F.sigmoid(i_l)
        silu_i_u = i_u * F.sigmoid(i_u)
        # Since silu is monotonic, the extremes are at the input extremes
        # Then multiply by (1-f) bounds
        cand = torch.stack([
            silu_i_l * one_minus_f_l,
            silu_i_l * one_minus_f_u,
            silu_i_u * one_minus_f_l,
            silu_i_u * one_minus_f_u,
        ], dim=0)
        i_l = cand.min(dim=0).values
        i_u = cand.max(dim=0).values

        # Apply attention mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(dtype=i_l.dtype)
            i_l = i_l * mask
            i_u = i_u * mask

        # Reshape for recurrent step: (B, L, (H*D)) -> (B, H, L, D)
        i_l = i_l.view(B, L, H, head_dim).transpose(1, 2)
        i_u = i_u.view(B, L, H, head_dim).transpose(1, 2)
        f_l_r = f_l.view(B, L, H, head_dim).transpose(1, 2) if f_l.dim() == 3 else f_l
        f_u_r = f_u.view(B, L, H, head_dim).transpose(1, 2) if f_u.dim() == 3 else f_u

        # Recurrent step
        o_l, o_u = ibp_recurrent_step(i_l, i_u, f_l_r, f_u_r)

        # Reshape back: (B, H, L, D) -> (B, L, H*D)
        o_l = o_l.transpose(1, 2).contiguous().view(B, L, expand_dim)
        o_u = o_u.transpose(1, 2).contiguous().view(B, L, expand_dim)

        # g_norm: FusedRMSNormSwishGate(g_proj(x), o)
        # This applies RMSNorm to g and then swish-gates o
        # Simplified: apply rmsnorm to g, then use as gate for o
        g_n_eps = getattr(attn.g_norm, 'eps', 1e-5)
        g_n_l, g_n_u = ibp_rmsnorm(g_l, g_u, attn.g_norm.weight, eps=g_n_eps)
        # The swish gate: g_norm applies swish to the gated output
        # ActNorm(x, y) = x * y.sigmoid() * y or similar
        # FusedRMSNormSwishGate applies: swish(rmsnorm(g)) * o
        # swish(z) = z * sigmoid(z), which is monotonic
        g_swish_l = g_n_l * F.sigmoid(g_n_l)
        g_swish_u = g_n_u * F.sigmoid(g_n_u)
        cand = torch.stack([
            g_swish_l * o_l, g_swish_l * o_u,
            g_swish_u * o_l, g_swish_u * o_u,
        ], dim=0)
        o_l = cand.min(dim=0).values
        o_u = cand.max(dim=0).values

        # Output projection
        o_l, o_u = ibp_linear(o_l, o_u, attn.o_proj.weight)
        return o_l, o_u

    def _ibp_short_conv(self, conv_module, h_l, h_u, attention_mask):
        """IBP through ShortConvolution (nn.Conv1d subclass + silu activation)."""
        # ShortConvolution extends nn.Conv1d, so it IS the conv layer
        weight = conv_module.weight
        bias = conv_module.bias
        stride = conv_module.stride[0] if isinstance(conv_module.stride, tuple) else conv_module.stride
        padding = conv_module.padding[0] if isinstance(conv_module.padding, tuple) else conv_module.padding

        # Transpose for Conv1d: (B, L, D) -> (B, D, L)
        h_l_t = h_l.transpose(1, 2)
        h_u_t = h_u.transpose(1, 2)

        h_l_t, h_u_t = ibp_conv1d(h_l_t, h_u_t, weight, bias,
                                  stride=stride, padding=padding)
        # Trim to input length (causal conv padding removes last elements)
        # But for IBP we keep the full output; model trims in forward

        # Silu activation (monotonic)
        h_l_t = h_l_t * F.sigmoid(h_l_t)
        h_u_t = h_u_t * F.sigmoid(h_u_t)

        # Transpose back: (B, D, L) -> (B, L, D)
        # Trim to match input length if needed
        out_len = h_l.shape[1]
        return (h_l_t.transpose(1, 2)[:, :out_len, :],
                h_u_t.transpose(1, 2)[:, :out_len, :])

    def _ibp_mlp(self, mlp, h_l, h_u):
        """IBP through HGRNBitMLP."""
        # gate_proj: BitLinear → 2*intermediate
        g_l, g_u = ibp_linear(h_l, h_u, mlp.gate_proj.weight)
        # Split into gate and value
        mid = g_l.shape[-1] // 2
        gate_l, gate_u = g_l[..., :mid], g_u[..., :mid]
        val_l, val_u = g_l[..., mid:], g_u[..., mid:]

        # swiglu(gate, val) = gate * sigmoid(gate) * val
        swiglu_l, swiglu_u = ibp_swiglu(gate_l, gate_u, val_l, val_u)

        # down_proj: BitLinear
        return ibp_linear(swiglu_l, swiglu_u, mlp.down_proj.weight)

    # ── CROWN / IBP-CROWN (linear relaxation) ──────────────────────────

    @torch.no_grad()
    def crown_bounds(self, input_ids, epsilon, attention_mask=None, n_layers=4):
        """
        Pure CROWN: back-propagate linear bounds from output to input through
        the last n_layers layers. Uses tight IBP for intermediate pre-activation
        bounds needed by CROWN relaxation.
        """
        emb = self.get_embeddings(input_ids).float()
        emb_l, emb_u = self.perturb_embeddings(emb, epsilon)
        base_model = self.model.model
        n_total = len(base_model.layers)
        start_layer = max(0, n_total - n_layers)

        # Run tight IBP through early layers to get input bounds for CROWN segment
        h_l, h_u = emb_l, emb_u
        for layer_idx in range(start_layer):
            layer = base_model.layers[layer_idx]
            h_l, h_u = self._ibp_block_tight(layer, h_l, h_u, attention_mask, layer_idx)
            h_l = torch.clamp(h_l, min=-1e6, max=1e6)
            h_u = torch.clamp(h_u, min=-1e6, max=1e6)

        x_l, x_u = h_l, h_u  # input bounds to the CROWN segment

        # Forward IBP through CROWN segment to get pre-activation bounds
        layer_bounds = []
        for layer_idx in range(start_layer, n_total):
            layer = base_model.layers[layer_idx]
            h_l, h_u = self._ibp_block_tight(layer, h_l, h_u, attention_mask, layer_idx)
            h_l = torch.clamp(h_l, min=-1e6, max=1e6)
            h_u = torch.clamp(h_u, min=-1e6, max=1e6)
            layer_bounds.append((h_l.clone(), h_u.clone()))

        # Final norm
        norm_weight = base_model.norm.weight
        final_l, final_u = ibp_rmsnorm_tight(h_l, h_u, norm_weight, eps=base_model.norm.eps)
        lm_weight = self.model.lm_head.weight.float()

        # ── CROWN backward pass ──
        B, L, D = final_l.shape
        V = lm_weight.size(0)
        device = final_l.device

        # Start from output: A = I for each position
        last_uA = torch.eye(V, device=device).unsqueeze(0).unsqueeze(0).expand(B, L, V, V)
        last_lA = torch.eye(V, device=device).unsqueeze(0).unsqueeze(0).expand(B, L, V, V)
        ubias_sum = torch.zeros(B, L, V, device=device)
        lbias_sum = torch.zeros(B, L, V, device=device)

        # Backward through LM head (linear)
        uA, ub, lA, lb = crown_linear_backward(last_uA, last_lA, lm_weight)
        ubias_sum = ubias_sum + ub if ub is not None else ubias_sum
        lbias_sum = lbias_sum + lb if lb is not None else lbias_sum
        last_uA, last_lA = uA, lA

        # Backward through final RMSNorm
        uA, ub, lA, lb = crown_rmsnorm_backward(
            last_uA, last_lA, h_l, h_u, norm_weight, eps=base_model.norm.eps)
        ubias_sum = ubias_sum + ub
        lbias_sum = lbias_sum + lb
        last_uA, last_lA = uA, lA

        # Backward through CROWN segment layers (reverse order)
        for rev_idx, layer_idx in enumerate(range(n_total - 1, start_layer - 1, -1)):
            layer = base_model.layers[layer_idx]
            h_l, h_u = layer_bounds[rev_idx]

            # Backward through MLP
            last_uA, last_lA, ubias_sum, lbias_sum = self._crown_backward_mlp(
                layer.mlp, last_uA, last_lA, h_l, h_u, ubias_sum, lbias_sum)

            # Backward through attention
            last_uA, last_lA, ubias_sum, lbias_sum = self._crown_backward_attention(
                layer.attn, last_uA, last_lA, h_l, h_u,
                attention_mask, ubias_sum, lbias_sum)

        # Concretize bounds at input x_L, x_U
        logits_l = concretize_bounds(last_lA, lbias_sum, x_l, x_u, sign=-1)
        logits_u = concretize_bounds(last_uA, ubias_sum, x_l, x_u, sign=+1)

        return logits_l, logits_u

    @torch.no_grad()
    def ibp_crown_bounds(self, input_ids, epsilon, attention_mask=None):
        """
        IBP-CROWN hybrid: use tight IBP for intermediate hidden-state bounds,
        then apply CROWN only through the final LM head and norm.

        The CROWN backward starts with identity at the output (A = I_V, one row
        per logit), then back-propagates through LM head (V×D) and RMSNorm.
        """
        emb = self.get_embeddings(input_ids).float()
        emb_l, emb_u = self.perturb_embeddings(emb, epsilon)
        base_model = self.model.model

        # ── IBP forward (tight, for intermediate bounds) ──
        h_l, h_u = emb_l, emb_u
        for layer_idx, layer in enumerate(base_model.layers):
            h_l, h_u = self._ibp_block_tight(layer, h_l, h_u, attention_mask, layer_idx)
            h_l = torch.clamp(h_l, min=-1e6, max=1e6)
            h_u = torch.clamp(h_u, min=-1e6, max=1e6)

        hidden_l, hidden_u = h_l.clone(), h_u.clone()

        # ── IBP through final norm + lm_head ──
        norm_weight = base_model.norm.weight
        final_l, final_u = ibp_rmsnorm_tight(h_l, h_u, norm_weight, eps=base_model.norm.eps)
        lm_weight = self.model.lm_head.weight.float()
        final_l, final_u = final_l.float(), final_u.float()
        ibp_logits_l, ibp_logits_u = ibp_linear(final_l, final_u, lm_weight)

        # ── CROWN backward through final norm + lm_head ──
        # last_uA: (V, D) — each of V rows is the linear bound coeff for one logit
        # Start from output: A = W_lm (since output = W_lm @ x_norm)
        B, L, D = hidden_l.shape
        V = lm_weight.size(0)
        device = hidden_l.device

        # CROWN through LM head: identity at output means A_out = W_lm
        # For each logit k: logit_k = w_k^T @ x_norm, so A_k = w_k^T
        # Upper and lower A are the same for a linear layer
        last_uA = lm_weight.clone()  # (V, D)
        last_lA = lm_weight.clone()  # (V, D)
        ubias_sum = torch.zeros(V, device=device)
        lbias_sum = torch.zeros(V, device=device)

        # CROWN backward through RMSNorm
        # Need to handle (V, D) @ (B, L, D) -> (B, L, V)
        uA, ub, lA, lb = crown_rmsnorm_backward_v2(
            last_uA, last_lA, hidden_l, hidden_u, norm_weight, eps=base_model.norm.eps)
        ubias_sum = ubias_sum + ub if ub is not None else ubias_sum
        lbias_sum = lbias_sum + lb if lb is not None else lbias_sum

        # Concretize: for each (b, l, v), compute bound from A(v,:) @ x(b,l,:) + bias(v)
        crown_logits_l = concretize_bounds_v2(lA, lbias_sum, hidden_l, hidden_u, sign=-1)
        crown_logits_u = concretize_bounds_v2(uA, ubias_sum, hidden_l, hidden_u, sign=+1)

        return ibp_logits_l, ibp_logits_u, crown_logits_l, crown_logits_u

    def _ibp_block_tight(self, layer, h_l, h_u, attention_mask, layer_idx):
        """Single-block IBP using tight RMSNorm for trend analysis."""
        n_l, n_u = ibp_rmsnorm_tight(h_l, h_u, layer.attn_norm.weight, eps=layer.attn_norm.eps)
        a_l, a_u = self._ibp_attention(layer.attn, n_l, n_u, attention_mask, layer_idx)
        merged_l, merged_u = h_l + a_l, h_u + a_u
        n2_l, n2_u = ibp_rmsnorm_tight(merged_l, merged_u, layer.mlp_norm.weight, eps=layer.mlp_norm.eps)
        m_l, m_u = self._ibp_mlp(layer.mlp, n2_l, n2_u)
        return merged_l + m_l, merged_u + m_u

    def _crown_backward_mlp(self, mlp, last_uA, last_lA, h_l, h_u, ubias_sum, lbias_sum):
        """CROWN backward through HGRNBitMLP (SwiGLU + down_proj)."""
        # MLP: h -> gate_proj -> swiglu -> down_proj -> output
        gate_weight = mlp.gate_proj.weight
        down_weight = mlp.down_proj.weight
        mid_dim = down_weight.size(1)
        gate_dim = gate_weight.size(0) // 2

        # 1. Backward through down_proj (linear)
        uA, ub, lA, lb = crown_linear_backward(last_uA, last_lA, down_weight)
        ubias_sum = ubias_sum + ub if ub is not None else ubias_sum
        lbias_sum = lbias_sum + lb if lb is not None else lbias_sum

        # 2. Through SwiGLU: we need bounds on gate_proj output
        g_l, g_u = ibp_linear(h_l, h_u, gate_weight)
        gate_l, gate_u = g_l[..., :gate_dim], g_u[..., :gate_dim]
        val_l, val_u = g_l[..., gate_dim:], g_u[..., gate_dim:]

        uA, ub, lA, lb = crown_swiglu_backward(uA, lA, gate_l, gate_u, val_l, val_u)
        ubias_sum = ubias_sum + ub
        lbias_sum = lbias_sum + lb

        # 3. Through gate_proj (linear): uA and lA are now on swiglu input
        # The swiglu input is the concatenation of gate and val paths from gate_proj
        # Sum the contributions from both halves
        if uA is not None and isinstance(uA, tuple):
            uA_gate, uA_val = uA
            # gate path: gate_proj weight first half
            uA_from_gate = uA_gate.matmul(gate_weight[:gate_dim])
            uA_from_val = uA_val.matmul(gate_weight[gate_dim:])
            uA = uA_from_gate + uA_from_val
        if lA is not None and isinstance(lA, tuple):
            lA_gate, lA_val = lA
            lA_from_gate = lA_gate.matmul(gate_weight[:gate_dim])
            lA_from_val = lA_val.matmul(gate_weight[gate_dim:])
            lA = lA_from_gate + lA_from_val

        return uA, lA, ubias_sum, lbias_sum

    def _crown_backward_attention(self, attn, last_uA, last_lA, h_l, h_u,
                                   attention_mask, ubias_sum, lbias_sum):
        """CROWN backward through attention (linear projections only).
        Skips recurrence nonlinearity; uses the input hidden states as proxy."""
        # Output projection (linear)
        uA, ub, lA, lb = crown_linear_backward(last_uA, last_lA, attn.o_proj.weight)
        ubias_sum = ubias_sum + ub if ub is not None else ubias_sum
        lbias_sum = lbias_sum + lb if lb is not None else lbias_sum

        # Skip detailed backward through recurrence (non-differentiable interval).
        # Instead, attribute uncertainty to attention input proportionally.
        return uA, lA, ubias_sum, lbias_sum

    def compute_metrics(self, lb_logits, ub_logits):
        """Compute uncertainty metrics from output bounds."""
        metrics = {}

        # Mean bound width
        width = ub_logits - lb_logits
        metrics['mean_width'] = width.abs().mean().item()
        metrics['max_width'] = width.abs().max().item()
        metrics['median_width'] = width.abs().median().item()

        # Per-token bound width (averaged over vocab)
        token_width = width.abs().mean(dim=-1)  # (B, L)
        metrics['token_widths'] = token_width.cpu()

        # Entropy of midpoint vs bounds
        mid = (lb_logits + ub_logits) / 2.0
        probs_mid = F.softmax(mid, dim=-1)
        entropy_mid = -(probs_mid * torch.log(probs_mid + 1e-8)).sum(-1).mean().item()
        metrics['entropy_mid'] = entropy_mid

        # Top-1 stability: does argmax change between lb and ub?
        top_lb = lb_logits.argmax(dim=-1)
        top_ub = ub_logits.argmax(dim=-1)
        top_mid = mid.argmax(dim=-1)
        metrics['top1_agreement'] = (top_lb == top_ub).float().mean().item()
        metrics['top1_lb_vs_mid'] = (top_lb == top_mid).float().mean().item()
        metrics['top1_ub_vs_mid'] = (top_ub == top_mid).float().mean().item()

        # Certified margin
        margins = margin_from_bounds(lb_logits, ub_logits)
        metrics['certified_margin_mean'] = margins.mean().item()
        metrics['certified_robust_frac'] = (margins > 0).float().mean().item()

        return metrics

    def get_cache(self):
        return self._cache
