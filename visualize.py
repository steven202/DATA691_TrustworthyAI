"""
Publication-quality visualizations for verified UQ report.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

COLORS = {'ibp': '#2196F3', 'empirical': '#FF9800', 'mc_empirical': '#4CAF50',
          'ibp_tight_4': '#9C27B0'}


def _extract_metric(model_results, epsilons, method, metric_key):
    all_values = {eps: [] for eps in epsilons}
    for prompt_key, prompt_data in model_results.items():
        results = prompt_data['results']
        for eps in epsilons:
            eps_key = f'eps_{eps}'
            if eps_key in results and method in results[eps_key]:
                m_data = results[eps_key][method]
                if isinstance(m_data, dict) and 'error' not in m_data and metric_key in m_data:
                    v = m_data[metric_key]
                    all_values[eps].append(float(v) if not isinstance(v, list) else np.mean(v))
    means = [np.mean(all_values[e]) if all_values[e] else np.nan for e in epsilons]
    stds = [np.std(all_values[e]) if all_values[e] else np.nan for e in epsilons]
    return means, stds


def plot_bound_width_vs_epsilon(model_results, epsilons, save_path, title=None):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, label, color, marker in [
        ('mc_empirical', 'Monte Carlo ($K{=}50$)', COLORS['mc_empirical'], 'o'),
        ('empirical', 'Two-pass empirical', COLORS['empirical'], 's'),
    ]:
        means, stds = _extract_metric(model_results, epsilons, method, 'mean_width')
        valid = ~np.isnan(means)
        if valid.sum() >= 2:
            ax.errorbar(np.array(epsilons)[valid], np.array(means)[valid],
                        yerr=np.array(stds)[valid], label=label,
                        color=color, marker=marker, capsize=3, linewidth=1.8, markersize=6)

    ax.set_xlabel('Perturbation $\\varepsilon$ ($L_\\infty$)')
    ax.set_ylabel('Mean Output Bound Width')
    if title:
        ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_token_uncertainty_heatmap(model_results, epsilons, save_path, method='mc_empirical'):
    first_prompt = list(model_results.keys())[0]
    results = model_results[first_prompt]['results']
    token_data = {}
    for eps in epsilons:
        eps_key = f'eps_{eps}'
        if eps_key not in results or method not in results[eps_key]:
            continue
        m_data = results[eps_key][method]
        if 'token_widths' not in m_data:
            continue
        tw = m_data['token_widths']
        if isinstance(tw, list):
            tw = np.array(tw).flatten()
        elif hasattr(tw, 'flatten'):
            tw = tw.flatten()
        token_data[eps] = tw

    if not token_data:
        return

    eps_list = sorted(token_data.keys())
    max_len = max(len(v) for v in token_data.values())
    matrix = np.zeros((len(eps_list), max_len))
    for i, eps in enumerate(eps_list):
        v = token_data[eps]
        matrix[i, :len(v)] = v

    fig, ax = plt.subplots(figsize=(max(8, max_len * 0.35), 3.5))
    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                   norm=LogNorm(vmin=max(matrix[matrix > 0].min(), 1e-8), vmax=matrix.max()))
    ax.set_yticks(range(len(eps_list)))
    ax.set_yticklabels([f'$\\varepsilon={e}$' for e in eps_list])
    ax.set_xlabel('Token Position')
    ax.set_title('Per-Token Uncertainty (Monte Carlo Bound Width)')
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label('Mean Bound Width')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_bound_growth_summary(model_results, epsilons, save_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods_to_try = ['ibp_tight_4', 'ibp']
    for eps in epsilons:
        eps_key = f'eps_{eps}'
        for pk, pv in model_results.items():
            if eps_key in pv['results']:
                for method in methods_to_try:
                    md = pv['results'][eps_key].get(method, {})
                    widths = md.get('layer_widths', [])
                    if widths:
                        ax.semilogy(range(len(widths)), widths, marker='.',
                                    label=f'$\\varepsilon={eps}$', linewidth=1.5, alpha=0.7)
                        break  # only plot first prompt
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Mean Bound Width (log scale)')
    ax.set_title('Bound Propagation Through Layers')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25, linestyle='--')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_ibp_vs_empirical(model_results, epsilons, save_path):
    ibp_vals, emp_vals, eps_labels = [], [], []
    for eps in epsilons:
        eps_key = f'eps_{eps}'
        for _, pv in model_results.items():
            results = pv['results']
            if eps_key not in results:
                continue
            for m_ibp, m_emp in [('ibp_tight_4', 'mc_empirical'), ('ibp', 'empirical')]:
                if m_ibp in results[eps_key] and m_emp in results[eps_key]:
                    i = results[eps_key][m_ibp].get('mean_width')
                    e = results[eps_key][m_emp].get('mean_width')
                    if i and e and not np.isnan(i) and not np.isnan(e):
                        ibp_vals.append(i)
                        emp_vals.append(e)
                        eps_labels.append(eps)
                    break

    if not ibp_vals:
        return

    fig, ax = plt.subplots(figsize=(5.5, 5))
    unique_eps = sorted(set(eps_labels))
    for eps in unique_eps:
        idx = [j for j, e in enumerate(eps_labels) if e == eps]
        ax.scatter(np.array(emp_vals)[idx], np.array(ibp_vals)[idx],
                   label=f'$\\varepsilon={eps}$', s=30, alpha=0.7)
    all_vals = ibp_vals + emp_vals
    max_val = max(all_vals) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.2, linewidth=1)
    ax.set_xlabel('Empirical Bound Width')
    ax.set_ylabel('IBP Bound Width')
    ax.set_title('Certified vs.\ Empirical Bounds')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, linestyle='--')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_cross_model_comparison(all_results, epsilons, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, method, method_key in zip(
        axes, ['Monte Carlo ($K{=}50$)', 'Empirical (2-pass)'],
        ['mc_empirical', 'empirical']
    ):
        for model_name, model_results in all_results.items():
            label = model_name.replace('MMfreeLM-', '')
            means, stds = _extract_metric(model_results, epsilons, method_key, 'mean_width')
            valid = ~np.isnan(means)
            if valid.sum() >= 1:
                ax.errorbar(np.array(epsilons)[valid], np.array(means)[valid],
                            yerr=np.array(stds)[valid], label=label,
                            marker='o', capsize=3, linewidth=1.8, alpha=0.8)
        ax.set_xlabel('$\\varepsilon$')
        ax.set_ylabel('Mean Bound Width')
        ax.set_title(method)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xscale('log')
    fig.suptitle('Uncertainty Across Model Sizes', fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_prediction_stability(model_results, epsilons, save_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, label, color, marker in [
        ('mc_empirical', 'Monte Carlo', COLORS['mc_empirical'], 'o'),
        ('empirical', 'Two-pass', COLORS['empirical'], 's'),
    ]:
        means, stds = _extract_metric(model_results, epsilons, method, 'top1_agreement')
        valid = ~np.isnan(means)
        if valid.sum() >= 2:
            ax.errorbar(np.array(epsilons)[valid], np.array(means)[valid],
                        yerr=np.array(stds)[valid], label=label,
                        color=color, marker=marker, capsize=3, linewidth=1.8)
    ax.set_xlabel('Perturbation $\\varepsilon$')
    ax.set_ylabel('Top-1 Agreement')
    ax.set_title('Prediction Stability Under Perturbation')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_ylim(0, 1.05)
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
