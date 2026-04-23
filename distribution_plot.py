"""
Feature distribution plots: TX, RX, ΔTX/ΔRX — Normal (label=0) vs Attack (label=1)

Each CSV is an independent experiment. KDE is computed per CSV, then averaged
across all CSVs in the domain to produce the aggregated distribution.

Output folder structure:
  analysis_output/
    local_repair/
      Node_5/
        base/   → dist_domain1.png
        gc/     → dist_domain2.png
        oo/     → dist_domain3.png
      Node_10/
        ...
    blackhole/
      ...
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = os.path.join(os.path.dirname(__file__), "attack_data")
XLSX    = os.path.join(os.path.dirname(__file__), "domain_details.xlsx")
OUT_DIR = os.path.join(os.path.dirname(__file__), "analysis_output")

# ── Load domain metadata ───────────────────────────────────────────────────────
meta = pd.read_excel(XLSX)
meta.columns = meta.columns.str.strip()
meta["Domain Name"] = meta["Domain Name"].str.strip().str.lower()
meta["Attack Type"] = meta["Attack Type"].str.strip().str.lower().str.replace(" ", "_")
meta["Version"]     = meta["Version"].str.strip().str.lower()
meta["Node"]        = meta["Node"].astype(int)

# ── Styling ────────────────────────────────────────────────────────────────────
COLORS     = {0: "#2196F3", 1: "#F44336"}
LABEL_NAME = {0: "Normal",  1: "Attack"}
FEATURES   = ["tx", "rx", "tx.1", "rx.1"]
FEAT_TITLE = {
    "tx":   "TX (mean)",
    "rx":   "RX (mean)",
    "tx.1": "TX (std)",
    "rx.1": "RX (std)",
}

ATTACK_DISPLAY = {
    "blackhole":    "Blackhole",
    "dis_flooding": "DIS-Flooding",
    "failing_node": "Failing Node",
    "local_repair": "Local Repair",
    "worst_parent": "Worst Parent",
}

# ── Helper: compute features for a single CSV ─────────────────────────────────
def compute_features(path):
    df = pd.read_csv(path, index_col=0)
    # tx   = mean TX across nodes,  rx   = mean RX across nodes
    # tx.1 = std  TX across nodes,  rx.1 = std  RX across nodes
    df = df[["tx", "rx", "tx.1", "rx.1", "label"]].copy()
    return df

# ── Helper: averaged KDE across independent experiments ───────────────────────
def averaged_kde(csv_list, feat, lbl, xs):
    """
    For each CSV, compute a KDE on label=lbl samples, evaluate on xs.
    Return the mean KDE across all CSVs (equal weight per experiment).
    """
    kde_curves = []
    all_values = []   # collected only for KS test later
    for path in csv_list:
        df     = compute_features(path)
        values = df.loc[df["label"] == lbl, feat].dropna()
        all_values.append(values)
        if len(values) < 10:
            continue
        lo, hi = np.percentile(values, 1), np.percentile(values, 99)
        vc = values[(values >= lo) & (values <= hi)]
        if len(vc) < 10:
            continue
        try:
            kde = stats.gaussian_kde(vc, bw_method=0.3)
            kde_curves.append(kde(xs))
        except Exception:
            continue

    if not kde_curves:
        return None, pd.concat(all_values) if all_values else pd.Series(dtype=float)

    mean_kde = np.mean(kde_curves, axis=0)
    return mean_kde, pd.concat(all_values) if all_values else pd.Series(dtype=float)

# ── Helper: shared x-axis range across all CSVs for a feature ─────────────────
def global_range(csv_list, feat):
    lo_vals, hi_vals = [], []
    for path in csv_list:
        df     = compute_features(path)
        values = df[feat].dropna()
        if len(values) < 10:
            continue
        lo_vals.append(np.percentile(values, 1))
        hi_vals.append(np.percentile(values, 99))
    if not lo_vals:
        return 0, 1
    # use median to avoid range being skewed by outlier experiments
    return np.median(lo_vals), np.median(hi_vals)

# ── Helper: plot aggregated distributions for one domain ──────────────────────
def plot_domain(csv_list, domain, attack_type, node, version, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes = axes.flatten()
    # fig.suptitle(
    #    f"{ATTACK_DISPLAY.get(attack_type, attack_type)} |  "
    #    f"Node {node}  |  {version}\n"
    #    f"Feature Distributions: Normal vs Attack"
    #    f"(averaged over {len(csv_list)} experiments)",
    #    fontsize=18, fontweight="bold", y=1.02
    #)

    for ax, feat in zip(axes, FEATURES):
        lo, hi = global_range(csv_list, feat)
        xs = np.linspace(lo, hi, 400)

        all_vals = {}
        for lbl in [0, 1]:
            mean_ys, all_v = averaged_kde(csv_list, feat, lbl, xs)
            all_vals[lbl]  = all_v
            if mean_ys is None:
                continue
            n_total = len(all_v)
            ax.plot(xs, mean_ys, color=COLORS[lbl], linewidth=2.5,
                    label=f"{LABEL_NAME[lbl]})")
            ax.fill_between(xs, mean_ys, alpha=0.15, color=COLORS[lbl])

        # KS test on pooled values (for significance annotation only)
        v0, v1 = all_vals.get(0, pd.Series()), all_vals.get(1, pd.Series())
        if len(v0) > 1 and len(v1) > 1:
            ks_stat, ks_p = stats.ks_2samp(v0.dropna(), v1.dropna())
            ax.set_title(f"{FEAT_TITLE[feat]}\nKS = {ks_stat:.3f})",
                         fontsize=24, fontweight="bold")
        else:
            ax.set_title(FEAT_TITLE[feat], fontsize=24, fontweight="bold")

        ax.set_xlabel("Value", fontsize=28)
        ax.set_ylabel("Density", fontsize=28)
        ax.tick_params(axis="both", which="major", labelsize=24)
        ax.legend(fontsize=20)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

# ── Main loop: one plot per domain ────────────────────────────────────────────
subset = meta[meta["Attack Type"] != "failing_node"]
print(f"Processing {len(subset)} domains …\n")

for _, row in subset.iterrows():
    domain      = row["Domain Name"]
    attack_type = row["Attack Type"]
    node        = int(row["Node"])
    version     = row["Version"]

    folder = os.path.join(BASE, attack_type, domain)
    csv_list = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csv_list:
        print(f"  [SKIP] No CSVs found: {folder}")
        continue

    out_folder = os.path.join(OUT_DIR, attack_type, f"Node_{node}", version)
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, f"dist_{domain}.png")

    plot_domain(csv_list, domain, attack_type, node, version, out_path)
    print(f"  [{attack_type}/{domain}]  Node={node}  ver={version}  "
          f"experiments={len(csv_list)}  → {out_path.replace(OUT_DIR+'/', '')}")

print("\nDone.")
