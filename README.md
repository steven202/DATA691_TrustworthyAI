# Certified and Empirical Uncertainty Quantification in Matmul-Free Language Models

DATA 691 Trustworthy AI — Final Project

## Overview

This project applies formal verification techniques to quantify uncertainty in [MMfreeLM](https://github.com/ridgerchu/MatmulFreeLLM), a matmul-free recurrent language model family (370M–2.7B parameters). We compare certified and empirical bound propagation methods, and benchmark against a standard GPT-2 Medium (355M) Transformer baseline.

Methods implemented:

- **Interval Bound Propagation (IBP)** — certified forward interval arithmetic
- **CROWN linear relaxation** — backward diagonal-Jacobian relaxation for tight bounds
- **IBP-CROWN hybrid** — IBP for intermediate layers + CROWN for final projection
- **Monte Carlo empirical bounds** — model-agnostic uncertainty via L∞ embedding sampling
- **Two-pass corner evaluation** — fast heuristic using emb ± ε

## Setup

```bash
# 1. Create the conda environment
conda env create -f environment.yml

# 2. Activate the environment
conda activate uncertainty-verif

# 3. Verify installation
python -c "import torch; import mmfreelm; import transformers; print('OK')"
```

`environment.yml` pins Python 3.11, PyTorch 2.2.1 with CUDA 12.1, and all required packages (transformers, accelerate, mmfreelm, etc.). No other repositories or dependencies need to be installed — the IBP/CROWN bound propagation is implemented from scratch in `src/bound_ops.py` and does not depend on alpha-beta-CROWN or auto_LiRPA.

## Project Structure

```
.
├── environment.yml                    # Conda environment specification
├── README.md
├── src/
│   ├── bound_ops.py                   # IBP and CROWN operations (RMSNorm, SwiGLU, BitLinear)
│   ├── bound_model.py                 # Model wrapper with bound propagation methods
│   ├── experiments.py                 # Main experiment: certified + empirical bounds
│   ├── baseline_comparison.py         # GPT-2 vs MMfreeLM baseline comparison
│   ├── crown_experiment.py            # IBP vs CROWN single-op RMSNorm comparison
│   └── visualize.py                   # Publication-quality plotting
├── paper/
│   ├── report.tex                     # LaTeX report (7 pages)
│   ├── report.pdf                     # Compiled PDF
│   └── ref.bib                        # BibTeX references
└── results/
    ├── baseline/                      # Baseline comparison (GPT-2 vs MMfreeLM-370M)
    │   ├── baseline_results.json
    │   └── plots/
    ├── crown/                         # IBP vs CROWN RMSNorm comparison
    │   └── rmsnorm_comparison.json
    └── full/                          # Full experiment results
        └── plots/
```

## Usage

### Main experiment (certified IBP + IBP-CROWN + empirical bounds)

```bash
# Quick test (370M model, 2 prompts, 3 epsilons)
python src/experiments.py --small --output-dir results/test

# Full experiment (all 3 models, 5 prompts, 5 epsilons, 50 MC samples)
python src/experiments.py \
    --models "ridger/MMfreeLM-370M" "ridger/MMfreeLM-1.3B" "ridger/MMfreeLM-2.7B" \
    --epsilons 0.001 0.005 0.01 0.05 0.1 \
    --n-samples 50 \
    --output-dir results/full

# Single prompt (faster iteration)
python src/experiments.py --single-prompt --output-dir results/debug
```

### Baseline comparison (MMfreeLM vs GPT-2 Medium)

```bash
python src/baseline_comparison.py --output-dir results/baseline --n-samples 50
```

### CROWN vs IBP standalone (single RMSNorm operation)

```bash
python src/crown_experiment.py --output-dir results/crown
```

## Key Findings

1. **Certified IBP explodes on matmul-free architectures**: RMSNorm allows the denominator to approach zero under adversarial input ranges, causing bound width to grow 500,000× within a single layer. The gated recurrence then amplifies this across 24 layers.

2. **CROWN linear relaxation prevents RMSNorm explosion**: By linearizing RMSNorm around the midpoint with a diagonal Jacobian, CROWN keeps the denominator fixed and achieves a tightening ratio of up to 495,220× over certified IBP on a single RMSNorm operation.

3. **IBP-CROWN hybrid**: Using tight IBP for intermediate layers and CROWN for the final projection provides practical certified bounds while inheriting CROWN's tightness benefit at the output.

4. **Monte Carlo empirical bounds provide stable uncertainty quantification**: Model-agnostic MC sampling yields output logit widths of 0.5–10 under ε ∈ [0.001, 0.1], providing a practical alternative where certified bounds are infeasible.

5. **MMfreeLM-370M is more robust than GPT-2 Medium (355M)** under the same L∞ embedding perturbations, despite having a comparable parameter count and no dense matrix multiplications.

6. **Model size is not monotonic with robustness**: The 1.3B model exhibits 2–3× higher output sensitivity than both the 370M and 2.7B variants.

## References

- Zhu et al., "Scalable matmul-free language modeling", arXiv:2406.02528, 2024
- Wang et al., "Beta-CROWN: Efficient Bound Propagation with Per-Neuron Split Constraints", NeurIPS 2021
- Zhang et al., "Towards Stable and Efficient Training of Verifiably Robust Neural Networks", ICLR 2020
- Gowal et al., "On the effectiveness of interval bound propagation for training verifiably robust models", ICCV 2019
- Zhang et al., "Efficient Neural Network Robustness Certification with General Activation Functions", NeurIPS 2018
