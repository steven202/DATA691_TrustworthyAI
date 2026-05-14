"""
CROWN vs IBP comparison on RMSNorm: demonstrates why linear relaxation
is dramatically tighter than interval arithmetic for division-based ops.
"""
import torch, sys, os, json, argparse, logging
import numpy as np

sys.path.insert(0, '/home/guo/Data691TrustworthyAI/uncertainty_verification/src')
from bound_ops import ibp_rmsnorm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def crown_rmsnorm_diag(x_l, x_u, weight, eps=1e-6):
    """CROWN diagonal linear relaxation of RMSNorm."""
    weight = weight.to(dtype=x_l.dtype)
    mid = (x_l + x_u) / 2.0
    rms_mid = torch.sqrt(mid.square().mean(dim=-1, keepdim=True) + eps)
    w = weight.unsqueeze(0).unsqueeze(0) if weight.dim() == 1 else weight
    diag_jac = w / rms_mid
    y_mid = mid / rms_mid * w
    bias_correction = y_mid - diag_jac * mid
    x_mid = (x_u + x_l) / 2.0
    x_diff = (x_u - x_l) / 2.0
    crown_u = diag_jac * x_mid + diag_jac.abs() * x_diff + bias_correction
    crown_l = diag_jac * x_mid - diag_jac.abs() * x_diff + bias_correction
    return crown_l, crown_u


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='results/crown')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    D = 1024  # hidden dimension
    B, L = 1, 8
    weight = torch.ones(D, device=args.device)

    # Vary the bound width to see the IBP-CROWN gap
    bound_widths = [0.1, 1.0, 10.0, 100.0, 1000.0, 5000.0, 10000.0]
    results = []

    for bw in bound_widths:
        torch.manual_seed(42)
        x = torch.randn(B, L, D, device=args.device) * bw * 0.1
        x_l = x - bw / 2.0
        x_u = x + bw / 2.0

        # IBP certified
        ibp_l, ibp_u = ibp_rmsnorm(x_l, x_u, weight, eps=1e-6)
        ibp_w = (ibp_u - ibp_l).abs().mean().item()

        # CROWN diagonal
        crown_l, crown_u = crown_rmsnorm_diag(x_l, x_u, weight, eps=1e-6)
        crown_w = (crown_u - crown_l).abs().mean().item()

        ratio = ibp_w / max(crown_w, 1e-8)
        frac_zero = ((x_l <= 0) & (x_u >= 0)).float().mean().item()

        results.append({
            'bound_width': bw,
            'ibp_width': ibp_w,
            'crown_width': crown_w,
            'tightening_ratio': ratio,
            'frac_cross_zero': frac_zero,
        })
        logger.info(f'bw={bw:.0f}: IBP={ibp_w:.2e}, CROWN={crown_w:.2e}, '
                    f'ratio={ratio:.1f}x, zero_frac={frac_zero:.3f}')

    with open(f'{args.output_dir}/rmsnorm_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    # ASCII table for paper
    print('\n' + '='*70)
    print('RMSNorm: IBP vs CROWN Bound Width Comparison')
    print('='*70)
    print(f'{"Bound Range":>12s}  {"IBP Width":>12s}  {"CROWN Width":>12s}  {"Ratio":>8s}')
    print('-'*50)
    for r in results:
        print(f'{r["bound_width"]:>12.1f}  {r["ibp_width"]:>12.2e}  {r["crown_width"]:>12.2e}  {r["tightening_ratio"]:>7.0f}x')
    print('='*70)


if __name__ == '__main__':
    main()
