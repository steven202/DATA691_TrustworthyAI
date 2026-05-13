# Certified and Empirical Uncertainty Quantification in Matmul-Free Language Models

DATA 691 Trustworthy AI -- Final Project

## Overview

This project investigates uncertainty quantification for [MMfreeLM](https://github.com/ridgerchu/MatmulFreeLLM) (a matmul-free recurrent language model) using formal verification techniques. We apply:

- **Interval Bound Propagation (IBP)**: Certified bounds following the CROWN-IBP pattern
- **Monte Carlo Empirical Bounds**: Practical uncertainty estimates via sampling
- **Two-pass Empirical Bounds**: Fast heuristic using corner-point evaluation

## Structure

```
.
├── bound_ops.py         # IBP/CROWN operations for model layers
├── bound_model.py       # Model wrapper with bound propagation
├── experiments.py       # Main experiment script
├── visualize.py         # Publication-quality plotting
├── report.tex           # LaTeX report
└── results/
    └── full/
        └── plots/       # Generated figures
```

## Setup

Requires the `uncertainty-verif` conda environment (clone of `alpha-beta-crown` with `mmfreelm`):

```bash
conda activate uncertainty-verif
pip install -e /path/to/matmulfreellm  # if not already installed
```

## Usage

```bash
# Quick test (370M model, 2 prompts, 3 epsilons)
python experiments.py --small

# Full experiment (all 3 models, 5 prompts, 5 epsilons)
python experiments.py --models "ridger/MMfreeLM-370M" "ridger/MMfreeLM-1.3B" "ridger/MMfreeLM-2.7B" \
    --epsilons 0.001 0.005 0.01 0.05 0.1 --n-samples 50 --output-dir results/full
```

## Key Findings

1. **Certified IBP bounds explode** within 1 layer due to RMSNorm and recurrent operations
2. **Monte Carlo bounds** provide stable UQ: output widths of 0.5-10 under ε ∈ [0.001, 0.1]
3. **Model size is not monotonic** with robustness: 1.3B model is 2-3× more sensitive than 370M and 2.7B
4. Two-pass corner evaluation **underestimates** uncertainty by 2-6×

## References

- Zhang et al., "Towards Stable and Efficient Training of Verifiably Robust Neural Networks", ICLR 2020
- Gowal et al., "On the effectiveness of interval bound propagation", arXiv:1810.12715
- Qin et al., "HGRN2: Gated Linear RNNs with State Expansion", arXiv:2404.07904
