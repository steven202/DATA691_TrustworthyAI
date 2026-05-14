"""
Main experiment script: verified uncertainty quantification for MMfreeLM.
Compares IBP (certified), layer-limited IBP, and empirical (sampling) bounds.
"""
import os
import sys
import json
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/guo/Data691TrustworthyAI/matmulfreellm')
sys.path.insert(0, '/home/guo/Data691TrustworthyAI/CROWN-IBP')

import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.bound_model import BoundModel
from src.bound_ops import (
    ibp_linear, ibp_rmsnorm, ibp_rmsnorm_tight, ibp_sigmoid, ibp_swiglu,
    ibp_recurrent_step, bound_width
)
from src.visualize import (
    plot_bound_width_vs_epsilon,
    plot_token_uncertainty_heatmap,
    plot_bound_growth_summary,
    plot_ibp_vs_empirical,
    plot_cross_model_comparison,
    plot_prediction_stability,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MODEL_NAMES = [
    "ridger/MMfreeLM-370M",
    "ridger/MMfreeLM-1.3B",
    "ridger/MMfreeLM-2.7B",
]

PROMPTS = [
    "In a shocking finding, scientist discovered a herd of unicorns living in a remote, ",
    "The capital of France is Paris. The capital of Germany is",
    "Artificial intelligence will transform society by",
    "The main ethical concern about large language models is",
    "To solve climate change, we need to",
]

DEFAULT_EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1]
DEFAULT_N_SAMPLES = 50


def load_model_and_tokenizer(model_name, device='cuda'):
    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map='auto'
    )
    model.eval()
    return model, tokenizer


def ibp_safe(bound_model, input_ids, epsilon, attention_mask, max_layers=None):
    """
    IBP with NaN detection. If max_layers is set, only propagate through
    the first max_layers layers to study bound growth.
    """
    emb = bound_model.get_embeddings(input_ids).float()
    emb_l, emb_u = bound_model.perturb_embeddings(emb, epsilon)

    h_l, h_u = emb_l, emb_u
    base_model = bound_model.model.model
    n_layers = len(base_model.layers) if max_layers is None else min(max_layers, len(base_model.layers))
    layer_widths = []

    for layer_idx in range(n_layers):
        layer = base_model.layers[layer_idx]
        h_l, h_u = bound_model._ibp_block(layer, h_l, h_u, attention_mask, layer_idx)

        # NaN detection
        if torch.isnan(h_l).any() or torch.isnan(h_u).any():
            logger.warning(f"  IBP NaN at layer {layer_idx}, stopping.")
            return None, None, layer_widths, f'NaN at layer {layer_idx}'

        h_l = torch.clamp(h_l, min=-1e6, max=1e6)
        h_u = torch.clamp(h_u, min=-1e6, max=1e6)

        width = bound_width(h_l, h_u)
        layer_widths.append(width)

    # Final norm + LM head
    norm_weight = base_model.norm.weight
    h_l, h_u = ibp_rmsnorm(h_l, h_u, norm_weight, eps=base_model.norm.eps)
    lm_weight = bound_model.model.lm_head.weight.float()
    h_l, h_u = h_l.float(), h_u.float()
    logits_l, logits_u = ibp_linear(h_l, h_u, lm_weight)

    if torch.isnan(logits_l).any() or torch.isnan(logits_u).any():
        return None, None, layer_widths, 'NaN at output'

    return logits_l, logits_u, layer_widths, None


def ibp_tight_block(bound_model, h_l, h_u, layer, attention_mask, layer_idx):
    """Single-block IBP using tight RMSNorm (not certified, for trend analysis)."""
    # attn_norm
    n_l, n_u = ibp_rmsnorm_tight(h_l, h_u, layer.attn_norm.weight, eps=layer.attn_norm.eps)
    # attention
    a_l, a_u = bound_model._ibp_attention(layer.attn, n_l, n_u, attention_mask, layer_idx)
    # residual merge
    merged_l, merged_u = h_l + a_l, h_u + a_u
    # mlp_norm
    n2_l, n2_u = ibp_rmsnorm_tight(merged_l, merged_u, layer.mlp_norm.weight, eps=layer.mlp_norm.eps)
    # mlp
    m_l, m_u = bound_model._ibp_mlp(layer.mlp, n2_l, n2_u)
    # final residual
    return merged_l + m_l, merged_u + m_u


def ibp_tight_layers(bound_model, input_ids, epsilon, attention_mask, max_layers):
    """Heuristic IBP through first max_layers using tight RMSNorm."""
    emb = bound_model.get_embeddings(input_ids).float()
    emb_l, emb_u = bound_model.perturb_embeddings(emb, epsilon)
    h_l, h_u = emb_l, emb_u
    base_model = bound_model.model.model
    widths = []

    for layer_idx in range(min(max_layers, len(base_model.layers))):
        layer = base_model.layers[layer_idx]
        h_l, h_u = ibp_tight_block(bound_model, h_l, h_u, layer, attention_mask, layer_idx)
        if torch.isnan(h_l).any() or torch.isnan(h_u).any():
            return None, None, widths, f'NaN at layer {layer_idx}'
        h_l = torch.clamp(h_l, min=-1e6, max=1e6)
        h_u = torch.clamp(h_u, min=-1e6, max=1e6)
        widths.append(bound_width(h_l, h_u))

    norm_weight = base_model.norm.weight
    h_l, h_u = ibp_rmsnorm_tight(h_l, h_u, norm_weight, eps=base_model.norm.eps)
    h_l, h_u = h_l.float(), h_u.float()
    logits_l, logits_u = ibp_linear(h_l, h_u, bound_model.model.lm_head.weight.float())
    return logits_l, logits_u, widths, None



def run_experiment(bound_model, input_ids, epsilon, n_samples, attention_mask=None):
    """Run all bound methods for one (prompt, epsilon) pair."""
    results = {'epsilon': epsilon}

    # ── Tight IBP (heuristic, first 4 layers) ──
    lb_t, ub_t, wt, errt = ibp_tight_layers(bound_model, input_ids, epsilon, attention_mask, 4)
    if not errt:
        mt = bound_model.compute_metrics(lb_t, ub_t)
        results['ibp_tight_4'] = _serialize_metrics(mt)
        results['ibp_tight_4']['layer_widths'] = wt

    # ── IBP (certified, full) ──
    t0 = time.time()
    lb_i, ub_i, widths, err = ibp_safe(bound_model, input_ids, epsilon, attention_mask)
    if err:
        logger.warning(f"  ε={epsilon:.4f} IBP: {err}")
        results['ibp'] = {'error': err, 'layer_widths_partial': widths}
    else:
        metrics_i = bound_model.compute_metrics(lb_i, ub_i)
        results['ibp'] = _serialize_metrics(metrics_i)
        results['ibp']['time'] = time.time() - t0
        results['ibp']['layer_widths'] = widths
        logger.info(f"  ε={epsilon:.4f} IBP: width={metrics_i['mean_width']:.2e}, "
                    f"layers={len(widths)}")

    # ── IBP-CROWN hybrid (tight IBP + CROWN final layers) ──
    t0 = time.time()
    try:
        ibp_l, ibp_u, crown_l, crown_u = bound_model.ibp_crown_bounds(
            input_ids, epsilon, attention_mask)
        metrics_ibp = bound_model.compute_metrics(ibp_l, ibp_u)
        metrics_crown = bound_model.compute_metrics(crown_l, crown_u)
        results['ibp_crown_hybrid'] = {
            'ibp_width': metrics_ibp['mean_width'],
            'crown_width': metrics_crown['mean_width'],
            'tightening_ratio': metrics_ibp['mean_width'] / max(metrics_crown['mean_width'], 1e-8),
            'ibp_top1': metrics_ibp['top1_agreement'],
            'crown_top1': metrics_crown['top1_agreement'],
            'time': time.time() - t0,
        }
        logger.info(f"  ε={epsilon:.4f} IBP-CROWN: IBP={metrics_ibp['mean_width']:.4f}, "
                    f"CROWN={metrics_crown['mean_width']:.4f}, "
                    f"ratio={metrics_ibp['mean_width']/max(metrics_crown['mean_width'],1e-8):.1f}x")
    except Exception as e:
        logger.warning(f"  IBP-CROWN failed: {e}")
        results['ibp_crown_hybrid'] = {'error': str(e)}

    # ── Empirical (two-pass) ──
    t0 = time.time()
    try:
        lb_e, ub_e = bound_model.empirical_bounds(input_ids, epsilon, attention_mask)
        metrics_e = bound_model.compute_metrics(lb_e, ub_e)
        results['empirical'] = _serialize_metrics(metrics_e)
        results['empirical']['time'] = time.time() - t0
        logger.info(f"  ε={epsilon:.4f} Emp: width={metrics_e['mean_width']:.4f}")
    except Exception as e:
        results['empirical'] = {'error': str(e)}

    # ── Monte Carlo empirical ──
    if n_samples > 0:
        t0 = time.time()
        try:
            lb_mc, ub_mc = bound_model.empirical_bounds_sampling(
                input_ids, epsilon, n_samples=n_samples, attention_mask=attention_mask)
            metrics_mc = bound_model.compute_metrics(lb_mc, ub_mc)
            results['mc_empirical'] = _serialize_metrics(metrics_mc)
            results['mc_empirical']['time'] = time.time() - t0
            results['mc_empirical']['n_samples'] = n_samples
            logger.info(f"  ε={epsilon:.4f} MC:  width={metrics_mc['mean_width']:.4f}")
        except Exception as e:
            results['mc_empirical'] = {'error': str(e)}

    return results


def _serialize_metrics(metrics):
    """Convert metrics dict to JSON-serializable values."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
            if v.numel() == 1:
                out[k] = float(v.item())
            else:
                out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()
    return obj


def main():
    parser = argparse.ArgumentParser(description='Verified Uncertainty Quantification for MMfreeLM')
    parser.add_argument('--models', nargs='+', default=None)
    parser.add_argument('--epsilons', nargs='+', type=float, default=DEFAULT_EPSILONS)
    parser.add_argument('--prompts', nargs='+', default=None)
    parser.add_argument('--n-samples', type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument('--output-dir', type=str, default='results')
    parser.add_argument('--max-length', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip-mc', action='store_true', help='Skip Monte Carlo (faster)')
    parser.add_argument('--single-prompt', action='store_true')
    parser.add_argument('--small', action='store_true', help='Only 370M model, 2 prompts, fewer eps')
    args = parser.parse_args()

    models_to_run = args.models if args.models else (MODEL_NAMES[:1] if args.small else MODEL_NAMES)
    prompts = args.prompts if args.prompts else (PROMPTS[:2] if args.small else PROMPTS)
    epsilons = args.epsilons[:3] if args.small else args.epsilons
    n_samples = min(args.n_samples, 20) if args.small else args.n_samples

    if args.single_prompt:
        prompts = [prompts[0]]

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)

    all_results = {}

    for model_name in models_to_run:
        model_key = model_name.split('/')[-1]
        logger.info(f"{'='*60}\n  Model: {model_key}\n{'='*60}")

        model, tokenizer = load_model_and_tokenizer(model_name, args.device)
        bound_model = BoundModel(model, tokenizer, args.device)
        model_results = {}

        for prompt_idx, prompt in enumerate(prompts):
            logger.info(f"  Prompt {prompt_idx+1}/{len(prompts)}: {prompt[:60]}...")
            inputs = tokenizer(prompt, return_tensors='pt',
                              max_length=args.max_length, truncation=True)
            input_ids = inputs.input_ids.to(args.device)
            attn_mask = inputs.attention_mask.to(args.device) if 'attention_mask' in inputs else None

            prompt_results = {}
            for eps in tqdm(epsilons, desc='  ε', leave=False):
                prompt_results[f'eps_{eps}'] = run_experiment(
                    bound_model, input_ids, eps, n_samples, attn_mask)

            model_results[f'prompt_{prompt_idx}'] = {
                'text': prompt,
                'input_ids': input_ids.cpu().tolist(),
                'results': prompt_results,
            }

        all_results[model_key] = model_results

        # Per-model plots
        plot_dir = os.path.join(args.output_dir, 'plots', model_key)
        os.makedirs(plot_dir, exist_ok=True)
        try:
            plot_bound_width_vs_epsilon(model_results, epsilons,
                                        os.path.join(plot_dir, 'bound_width_vs_eps.png'))
            plot_token_uncertainty_heatmap(model_results, epsilons,
                                           os.path.join(plot_dir, 'token_uncertainty.png'))
            plot_bound_growth_summary(model_results, epsilons,
                                      os.path.join(plot_dir, 'bound_growth_summary.png'))
            plot_prediction_stability(model_results, epsilons,
                                      os.path.join(plot_dir, 'prediction_stability.png'))
        except Exception as e:
            logger.warning(f"  Plot failed: {e}")

        del model, bound_model
        torch.cuda.empty_cache()

    # Cross-model comparison
    if len(models_to_run) > 1:
        try:
            plot_dir = os.path.join(args.output_dir, 'plots')
            plot_cross_model_comparison(all_results, epsilons,
                                        os.path.join(plot_dir, 'cross_model.png'))
        except Exception as e:
            logger.warning(f"  Cross-model plot failed: {e}")

    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    logger.info(f"\nResults saved to {results_path}")
    return all_results


if __name__ == '__main__':
    main()
