import numpy as np
import itertools
import matplotlib.pyplot as plt
from pathlib import Path
# import seaborn as sns
import pickle
try:
    import torch
    from torchvision.utils import make_grid
except ImportError:
    pass

import matplotlib
font = {'family' : "serif",
        'weight' : 'normal',
        'size'   : 9}
TITLE_FONTSIZE = 9
TICK_FONTSIZE = 8
matplotlib.rc('font', **font)
# sns.set_palette('colorblind')

# Specify result and figure directories
result_root = Path("../../logs/opt")
assert result_root.exists()
fig_root = Path("../../logs/figures")
fig_root.mkdir(exist_ok=True)

# Specify benchmarks and weight types
benchmarks = ["chem"]
weight_types = ["rank"]
titles = {
    "chem": "Chemical Design"
}
LINESTYLES = ['-', '--', '-.', ':'] * 2

# Specify k and r values for rank weighting
k_rank = ["1e-3"]
r_rank = [50]
k_dict = dict(chem=k_rank)
r_dict = dict(chem=r_rank)

results = {}
for benchmark in benchmarks:
    results[benchmark] = {}
    for weight_type in weight_types:
        results[benchmark][weight_type] = {}

        # Specify random seeds
        seeds = [1, 2, 3, 4, 5]

        # Load rank weighting results
        if weight_type == "rank":
            for k, r in zip(k_dict[benchmark], r_dict[benchmark]):
                results[benchmark][weight_type][f"k_{k}-r_{r}"] = {}
                for seed in seeds:
                    res_file = result_root / benchmark / weight_type / f"k_{k}" / f"r_{r}" / f"seed{seed}" / "results.npz"
                    if res_file.is_file():
                        results[benchmark][weight_type][f"k_{k}-r_{r}"][str(seed)] = np.load(res_file,
                                                                                             allow_pickle=True)

# # Also load train data
# train_data = dict()
#
# # Chem
# with open("../../data/chem/qm9/alpha.pkl", "rb") as f:
#     chem_props = pickle.load(f)
# with open("../../data/chem/qm9/train.txt") as f:
#     smiles_list = [s.strip() for s in f.readlines()]
# train_data["chem"] = np.array([chem_props[s] for s in smiles_list])
# del smiles_list, chem_props


def make_optimization_plot(ax, data, n_queries, labels, benchmark, titles, to_plot, metric="top1"):
    query_idx = range(1, n_queries + 1)
    idx = 0
    for method in data["rank"]:
        if method not in to_plot[benchmark]:
            continue
        scores = []
        for seed in data["rank"][method]:
            all_scores = np.array(data["rank"][method][seed]["opt_point_properties"])

            # Top1 scores
            scores.append(np.array([np.max(all_scores[:q]) for q in query_idx]))

        scores_mean = np.nanmean(np.array(scores), axis=0)

        scores_std = np.nanstd(np.array(scores), axis=0)

        ax.plot(query_idx, scores_mean, label=labels[method], linewidth=1, linestyle=LINESTYLES[idx])
        ax.fill_between(query_idx, scores_mean - scores_std, scores_mean + scores_std, alpha=0.3)
        idx += 1

    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    ax.tick_params(axis='both', which='minor', labelsize=TICK_FONTSIZE)
    plt.sca(ax)
    plt.title(titles[benchmark], fontsize=TITLE_FONTSIZE)
    plt.xlabel(r"Num. eval. of $f$")
    plt.xlim([0, n_queries])


labels = {
    "k_1e-3-r_50": r"$k=10^{-3}$, $r=r_{\mathrm{low}}$",
}

to_plot = {
    "chem": ["k_1e-3-r_50"],
}


benchmarks_to_plot = ["chem"]

for metric in "top1".split():
    print(metric)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.5, 1.75))
    for ax, benchmark in zip([ax1, ax2, ax3], benchmarks_to_plot):
        make_optimization_plot(ax, results[benchmark], n_queries=500, labels=labels,
                               benchmark=benchmark, titles=titles, metric=metric, to_plot=to_plot)

    handles, legend_labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(legend_labels, handles))
    ncol = len(by_label.keys())

    fig.legend(list(by_label.values()), list(by_label.keys()), ncol=ncol, loc="upper center",
               bbox_to_anchor=(0.5, 0.11), columnspacing=1.7, borderpad=0.1, labelspacing=0.05,
               borderaxespad=0.01, handletextpad=0.5, handlelength=1.5)
    plt.tight_layout()
    plt.subplots_adjust(left=0.105, right=0.99, top=0.88, bottom=0.35, wspace=0.22)
    plt.savefig(fig_root / f"optimization-{metric}.pdf")
    plt.show()
    plt.clf()
