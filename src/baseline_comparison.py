"""
Baseline comparison: MMfreeLM vs standard Transformer (GPT-2 Medium).
Uses MC empirical bounds (model-agnostic) for fair comparison.
"""
import sys, os, json, time, argparse, logging
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/home/guo/Data691TrustworthyAI/matmulfreellm')
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# MMfreeLM (matmul-free, no dense matmuls) vs GPT-2 (standard Transformer)
MODELS = {
    'MMfreeLM-370M': 'ridger/MMfreeLM-370M',
    'GPT-2-Medium':  'openai-community/gpt2-medium',  # 355M params
}

PROMPTS = [
    "In a shocking finding, scientist discovered a herd of unicorns living in a remote, ",
    "The capital of France is Paris. The capital of Germany is",
    "Artificial intelligence will transform society by",
    "The main ethical concern about large language models is",
    "To solve climate change, we need to",
]
EPSILONS = [0.001, 0.005, 0.01, 0.05, 0.1]
N_SAMPLES = 50


@torch.no_grad()
def mc_empirical_bounds(model, input_ids, epsilon, n_samples, attention_mask=None):
    """MC empirical bounds: sample in embedding L∞ ball. Handles both
    MMfreeLM (inputs_embeds) and GPT-2 (hook-based) models."""
    embed_layer = model.get_input_embeddings()
    emb = embed_layer(input_ids)
    B, L, D = emb.shape

    # Check if model supports inputs_embeds directly
    import inspect
    sig = inspect.signature(model.forward)
    use_embeds = 'inputs_embeds' in sig.parameters

    all_logits = []
    for _ in range(n_samples):
        noise = (torch.rand(B, L, D, device=emb.device, dtype=emb.dtype) * 2 - 1) * epsilon
        outputs = model(inputs_embeds=emb + noise, attention_mask=attention_mask)
        all_logits.append(outputs.logits.cpu())

    stacked = torch.stack(all_logits, dim=0)
    return stacked.min(dim=0).values, stacked.max(dim=0).values


@torch.no_grad()
def two_pass_empirical_bounds(model, input_ids, epsilon, attention_mask=None):
    """Two-pass corner evaluation: run with emb±ε, take element-wise min/max."""
    embed_layer = model.get_input_embeddings()
    emb = embed_layer(input_ids)

    results = []
    for sign in [-1, 1]:
        emb_pert = emb + sign * epsilon
        outputs = model(inputs_embeds=emb_pert, attention_mask=attention_mask)
        results.append(outputs.logits.cpu())

    lb = torch.min(results[0], results[1])
    ub = torch.max(results[0], results[1])
    return lb, ub


def compute_metrics(lb, ub):
    width = ub - lb
    mid = (lb + ub) / 2.0
    probs_mid = F.softmax(mid, dim=-1)
    entropy = -(probs_mid * torch.log(probs_mid + 1e-8)).sum(-1)
    top_lb, top_ub = lb.argmax(-1), ub.argmax(-1)
    return {
        'mean_width': width.abs().mean().item(),
        'median_width': width.abs().median().item(),
        'max_width': width.abs().max().item(),
        'top1_agreement': (top_lb == top_ub).float().mean().item(),
        'mean_entropy': entropy.mean().item(),
        'token_widths': width.abs().mean(-1).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='results/baseline')
    parser.add_argument('--n-samples', type=int, default=N_SAMPLES)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}

    for label, model_name in MODELS.items():
        logger.info(f'=== {label} ({model_name}) ===')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map='auto')
        model.eval()

        model_results = {}
        for pi, prompt in enumerate(PROMPTS):
            inputs = tokenizer(prompt, return_tensors='pt', max_length=32, truncation=True)
            input_ids = inputs.input_ids.to(args.device)
            attn = inputs.attention_mask.to(args.device) if 'attention_mask' in inputs else None

            pr = {}
            for eps in tqdm(EPSILONS, desc=f'  {label} epsilons', leave=False):
                t0 = time.time()
                lb, ub = mc_empirical_bounds(model, input_ids, eps, args.n_samples, attn)
                m = compute_metrics(lb, ub)
                m['time'] = time.time() - t0
                pr[f'eps_{eps}'] = {'epsilon': eps, 'mc_empirical': m}

                # Two-pass empirical
                lb2, ub2 = two_pass_empirical_bounds(model, input_ids, eps, attn)
                m2 = compute_metrics(lb2, ub2)
                pr[f'eps_{eps}']['empirical'] = m2

                logger.info(f'  eps={eps:.3f}: MC width={m["mean_width"]:.4f}, '
                           f'2-pass width={m2["mean_width"]:.4f}, '
                           f'top1={m["top1_agreement"]:.3f}, time={m["time"]:.1f}s')

            model_results[f'prompt_{pi}'] = {'text': prompt, 'results': pr}

        all_results[label] = model_results
        del model; torch.cuda.empty_cache()

    # Save
    def ser(o):
        if isinstance(o, dict): return {k: ser(v) for k, v in o.items()}
        if isinstance(o, list): return [ser(v) for v in o]
        if isinstance(o, (np.floating, np.integer)): return float(o)
        if isinstance(o, torch.Tensor): return o.cpu().tolist()
        return o

    with open(f'{args.output_dir}/baseline_results.json', 'w') as f:
        json.dump(ser(all_results), f, indent=2)
    logger.info(f'Saved to {args.output_dir}/baseline_results.json')

    # Generate comparison plots
    from src.visualize import plot_cross_model_comparison, plot_bound_width_vs_epsilon
    os.makedirs(f'{args.output_dir}/plots', exist_ok=True)
    plot_cross_model_comparison(all_results, EPSILONS,
                                f'{args.output_dir}/plots/baseline_comparison.png')
    for label in all_results:
        plot_bound_width_vs_epsilon(all_results[label], EPSILONS,
                                    f'{args.output_dir}/plots/{label}_width.png')
    logger.info('Plots done')


if __name__ == '__main__':
    main()
