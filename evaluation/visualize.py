#!/usr/bin/env python3
"""
FoodMedKG - Evaluation Visualisations
======================================
Generates paper-ready figures for all three evaluation parts.

Usage
-----
    cd evaluation
    python visualize.py                        # reads results/, writes figures/
    python visualize.py --results results/     # explicit input dir
    python visualize.py --format pdf           # pdf or png (default png)
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

# Acceptance threshold of the ingredient matcher (Sec. 5.2; see linking/threshold_curve.py)
LINKING_TAU = 0.68

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.titlesize": 19,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

PALETTE   = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
CORRECT   = "#4CAF50"
INCORRECT = "#F44336"
NEUTRAL   = "#90A4AE"


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  → {path.name}  +  {pdf_path.name}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PART A
# ══════════════════════════════════════════════════════════════════════════════

def fig_a_metric_distributions(dfs: "dict[str, pd.DataFrame]", out: Path) -> None:
    """5 metric subplots with one overlapping KDE line per model."""
    has_soft = any("f1_soft" in df.columns and df["f1_soft"].gt(0).any() for df in dfs.values())
    metrics = [
        ("f1",              "F1 Strict"),
        ("f1_flexible",     "F1 Flexible"),
    ] + ([("f1_soft", "F1 Soft")] if has_soft else []) + [
        ("precision",       "Precision"),
        ("recall",          "Recall"),
        ("prep_valid_rate", "Prep Valid"),
    ]
    n_metrics = len(metrics)
    model_names = list(dfs.keys())
    colors = PALETTE

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.2 * n_metrics, 4), sharey=False)
    if n_metrics == 1:
        axes = [axes]
    fig.suptitle("Metric Distributions Across Models",
                 fontsize=18, fontweight="bold", y=1.03)

    for ax, (col, label) in zip(axes, metrics):
        ax.set_title(label, fontsize=15)
        ax.set_xlim(-0.05, 1.05)
        for i, (model, df) in enumerate(dfs.items()):
            if col not in df.columns:
                continue
            vals  = df[col].dropna().values
            color = colors[i % len(colors)]
            short = model.split(":")[0]
            if len(set(vals)) > 1:
                kde = gaussian_kde(vals, bw_method=0.4)
                x   = np.linspace(0, 1, 200)
                ax.fill_between(x, kde(x), alpha=0.10, color=color)
                ax.plot(x, kde(x), color=color, linewidth=2, label=short)
            else:
                ax.axvline(vals.mean(), color=color, linewidth=2,
                           linestyle="--", label=short)
            ax.axvline(vals.mean(), color=color, linestyle=":",
                       linewidth=1.2, alpha=0.7)
        ax.set_xlabel("Score", fontsize=14)
        ax.set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    seen, h2, l2 = set(), [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l); h2.append(h); l2.append(l)
    fig.legend(h2, l2, loc="upper center", ncol=len(model_names),
               fontsize=14, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout()
    _save(fig, out)


def fig_a_f1_comparison(dfs: "dict[str, pd.DataFrame]", out: Path) -> None:
    """
    Publication-quality F1 comparison: Strict / Flexible / Soft.

    Layout: one violin per (F1-type × model) group showing the full
    distribution across recipes, with per-model mean trajectories
    overlaid as connected dots and ± 1 SD error bars.
    """
    # Determine which F1 columns are available and non-trivial
    has_soft = any(
        "f1_soft" in df.columns and df["f1_soft"].gt(0).any()
        for df in dfs.values()
    )
    f1_cols   = ["f1", "f1_flexible"] + (["f1_soft"] if has_soft else [])
    x_labels  = ["Strict", "Flexible"] + (["Soft (τ=0.75)"] if has_soft else [])
    col_colors = {"f1": "#2196F3", "f1_flexible": "#FF9800", "f1_soft": "#E91E63"}

    models  = list(dfs.keys())
    n_types = len(f1_cols)
    x_pos   = np.arange(n_types)          # 0, 1, [2]

    fig, ax = plt.subplots(figsize=(4 + 2 * n_types, 5))

    # ── Violins (pooled across all models per F1 type) ────────────────────────
    pool = [
        pd.concat([df[c] for df in dfs.values() if c in df.columns]).dropna().values
        for c in f1_cols
    ]
    vp = ax.violinplot(pool, positions=x_pos, widths=0.55,
                       showmedians=False, showextrema=False)
    for i, (pc, col) in enumerate(zip(vp["bodies"], f1_cols)):
        pc.set_facecolor(col_colors[col])
        pc.set_alpha(0.18)
        pc.set_edgecolor(col_colors[col])
        pc.set_linewidth(0.8)
        # Median tick
        med = float(np.median(pool[i]))
        ax.plot([x_pos[i] - 0.18, x_pos[i] + 0.18], [med, med],
                color=col_colors[col], linewidth=2, zorder=4)

    # ── Per-model connected means ± SD ────────────────────────────────────────
    for j, (model, df) in enumerate(dfs.items()):
        short  = model.split(":")[0]
        color  = PALETTE[j % len(PALETTE)]
        means, stds, xs = [], [], []
        for i, col in enumerate(f1_cols):
            if col not in df.columns:
                continue
            v = df[col].dropna()
            means.append(v.mean())
            stds.append(v.std())
            xs.append(x_pos[i])

        ax.errorbar(xs, means, yerr=stds, fmt="o-",
                    color=color, linewidth=1.8, markersize=7,
                    capsize=4, capthick=1.2, zorder=5,
                    label=short, alpha=0.9)

        # Annotate final mean value
        ax.annotate(f"{means[-1]:.2f}",
                    xy=(xs[-1], means[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=11, color=color, va="center")

    # ── Axes & style ──────────────────────────────────────────────────────────
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=16)
    ax.set_ylabel("Ingredient F1", fontsize=16)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.55, n_types - 0.3)
    ax.set_title("F1 Score under different matching criteria",
                 fontsize=18, fontweight="bold", pad=12)
    ax.legend(fontsize=14, loc="lower right", framealpha=0.9,
              edgecolor="#DDDDDD", ncol=max(1, len(models) // 3))

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, out)


def fig_a_scatter_extraction(df: pd.DataFrame, out: Path, model: str) -> None:
    """Scatter: extracted vs expected ingredient count, coloured by F1."""
    fig, ax = plt.subplots(figsize=(6, 5))

    sc = ax.scatter(df["n_expected"], df["avg_n_extracted"],
                    c=df["f1"], cmap="RdYlGn", vmin=0, vmax=1,
                    s=90, edgecolors="gray", linewidths=0.4, zorder=3)

    lim = max(df["n_expected"].max(), df["avg_n_extracted"].max()) + 1
    ax.plot([0, lim], [0, lim], "--", color="gray", linewidth=1, alpha=0.5, label="1:1 line")
    ax.fill_between([0, lim], [0, lim], [0, lim * 1.5],
                    alpha=0.06, color="orange", label="Over-extraction zone")

    for _, row in df.iterrows():
        ax.annotate(row["recipe"][:14], (row["n_expected"], row["avg_n_extracted"]),
                    fontsize=10, alpha=0.75,
                    xytext=(3, 3), textcoords="offset points")

    plt.colorbar(sc, ax=ax, label="F1 Score", fraction=0.04, pad=0.02)
    ax.set_xlabel("Expected ingredients (ground truth)")
    ax.set_ylabel("Extracted ingredients (LLM)")
    ax.set_title(f"Part A - Extraction Count vs Ground Truth  [{model}]",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, frameon=False)
    fig.tight_layout()
    _save(fig, out)


def fig_a_model_comparison(dfs: dict[str, pd.DataFrame], out: Path) -> None:
    """Grouped bar chart comparing models across all metrics."""
    has_soft = any("f1_soft" in df.columns and df["f1_soft"].gt(0).any() for df in dfs.values())
    metrics = ["json_valid_rate", "fields_complete", "prep_valid_rate", "precision", "recall", "f1", "f1_flexible"] + (["f1_soft"] if has_soft else [])
    labels  = ["JSON Valid", "Complete", "Prep OK", "Precision", "Recall", "F1 Strict", "F1 Flexible"] + (["F1 Soft"] if has_soft else [])

    models   = list(dfs.keys())
    x        = np.arange(len(metrics))
    width    = 0.75 / len(models)

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (model, df) in enumerate(dfs.items()):
        offset = (i - len(models) / 2 + 0.5) * width
        means  = [df[m].mean() * 100 if m in df.columns else 0 for m in metrics]
        bars   = ax.bar(x + offset, means, width,
                        label=model, color=PALETTE[i % len(PALETTE)], alpha=0.85)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=11)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 115)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.set_title("Part A - LLM Model Comparison", fontsize=18, fontweight="bold")
    ax.legend(fontsize=14, loc="lower right")
    fig.tight_layout()
    _save(fig, out)


def fig_a_f1_boxplot(dfs: dict[str, pd.DataFrame], out: Path) -> None:
    """Box plot of F1 distribution per model."""
    fig, ax = plt.subplots(figsize=(7, 4))

    data   = [df["f1"].values for df in dfs.values()]
    labels = list(dfs.keys())

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([l.split(":")[0] for l in labels], fontsize=15)
    ax.set_ylabel("Ingredient F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1 Distribution per model", fontsize=17, fontweight="bold")
    ax.axhline(np.mean([df["f1"].mean() for df in dfs.values()]),
               color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# PART B
# ══════════════════════════════════════════════════════════════════════════════

def fig_b_score_distribution(df: pd.DataFrame, out: Path, top_k: int) -> None:
    """KDE of matching scores split by correct / incorrect."""
    correct   = df[df["top1_correct"]]["top1_score"].values
    incorrect = df[~df["top1_correct"]]["top1_score"].values

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.linspace(0, 1, 300)

    for vals, color, label in [
        (correct,   CORRECT,   f"Correct (n={len(correct)})"),
        (incorrect, INCORRECT, f"Incorrect (n={len(incorrect)})"),
    ]:
        if len(vals) > 1:
            kde = gaussian_kde(vals, bw_method=0.25)
            ax.fill_between(x, kde(x), alpha=0.2, color=color)
            ax.plot(x, kde(x), color=color, linewidth=2, label=label)
            ax.axvline(vals.mean(), color=color, linestyle="--",
                       linewidth=1.2, alpha=0.7)
        elif len(vals) == 1:
            ax.axvline(vals[0], color=color, linewidth=2, label=label)

    ax.set_xlabel("Matching score")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1.05)
    ax.set_title("Score distribution: correct vs incorrect matches",
                 fontsize=16, fontweight="bold")
    ax.legend(fontsize=14, frameon=False)
    fig.tight_layout()
    _save(fig, out)


def fig_b_score_violin_by_group(df: pd.DataFrame, out: Path) -> None:
    """Violin plot of matching scores per food group."""
    groups = sorted(df["group"].unique())
    data   = [df[df["group"] == g]["top1_score"].values for g in groups]

    fig, ax = plt.subplots(figsize=(13, 5))
    parts = ax.violinplot(data, positions=range(len(groups)),
                          showmedians=True, showextrema=True, widths=0.7)

    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(PALETTE[i % len(PALETTE)])
        pc.set_alpha(0.45)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=38, ha="right", fontsize=14)
    ax.set_ylabel("Matching score")
    ax.set_ylim(0.3, 1.05)
    ax.axhline(LINKING_TAU, color="gray", linestyle="--", linewidth=0.8,
               alpha=0.5, label=f"τ = {LINKING_TAU}")
    ax.set_title("Part B - Matching score Spread per Food Group",
                 fontsize=17, fontweight="bold")
    ax.legend(fontsize=14, frameon=False)
    fig.tight_layout()
    _save(fig, out)


def fig_b_group_accuracy(df_groups: pd.DataFrame, out: Path, top_k: int) -> None:
    """Horizontal grouped bar: Top-1 / Top-k / MRR per group, sorted."""
    top_k_col = f"top{top_k}"
    df = df_groups.sort_values("top1").reset_index(drop=True)
    y  = np.arange(len(df))
    h  = 0.25

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(y - h, df["top1"]      * 100, h, color=PALETTE[0], alpha=0.85, label="Top-1 Accuracy")
    ax.barh(y,     df[top_k_col]   * 100, h, color=PALETTE[1], alpha=0.85, label=f"Top-{top_k} Accuracy")
    ax.barh(y + h, df["mrr"]       * 100, h, color=PALETTE[2], alpha=0.85, label="MRR × 100")

    ax.set_yticks(y)
    ax.set_yticklabels(df["group"], fontsize=15)
    ax.set_xlabel("Score (%)")
    ax.set_xlim(0, 118)
    ax.axvline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(102, i, f"n={int(row['n'])}", va="center", fontsize=12, color="gray")
    ax.set_title("Ingredient matching accuracy by food group",
                 fontsize=17, fontweight="bold")
    ax.legend(loc="lower right", fontsize=14)
    fig.tight_layout()
    _save(fig, out)


def fig_b_group_lollipop(df_groups: pd.DataFrame, out: Path, top_k: int) -> None:
    """Lollipop chart - Top-1 dot + Top-k recovery, coloured by performance tier."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    top_k_col  = f"top{top_k}"
    sort_col   = "score_std" if "score_std" in df_groups.columns else "top1"
    df = df_groups.sort_values(sort_col, ascending=True).reset_index(drop=True)
    y  = np.arange(len(df))

    # Colour each group by its Top-1 tier
    cmap = LinearSegmentedColormap.from_list("perf", ["#E53935", "#FB8C00", "#43A047"])
    colors = [cmap(v) for v in df["top1"].values]

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(left=0.22)          # room for long group names
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Subtle alternating row bands
    for i in range(len(df)):
        ax.axhspan(i - 0.5, i + 0.5, color="white" if i % 2 == 0 else "#F5F5F5", zorder=0)

    for i, (_, row) in enumerate(df.iterrows()):
        t1  = row["top1"]  * 100
        tk  = row[top_k_col] * 100
        mrr = row["mrr"]   * 100
        c   = colors[i]

        # Stem from 60 to Top-1
        ax.plot([60, t1], [i, i], color=c, linewidth=2.2, solid_capstyle="round", zorder=2)

        # Dashed extension Top-1 → Top-k (always drawn)
        ax.plot([t1, tk], [i, i], color="#888888", linewidth=1.5,
                linestyle="--", alpha=0.7, zorder=2)

        # Top-1 filled circle
        ax.scatter(t1, i, s=160, color=c, zorder=4, edgecolors="white", linewidths=1.2)

        # Top-k hollow diamond (always shown, gray)
        ax.scatter(tk, i, s=90, marker="D", color="none",
                   edgecolors="#888888", linewidths=1.8, zorder=4)

        # MRR tick mark
        ax.plot([mrr, mrr], [i - 0.28, i + 0.28], color=c,
                linewidth=1.5, alpha=0.6, zorder=3)

        # Value label - in the blank space after 100%
        ax.text(101.5, i, f"{t1:.0f}%", va="center", fontsize=13,
                color=c, fontweight="bold")

    # 100% reference line
    ax.axvline(100, color="#BDBDBD", linestyle="--", linewidth=1, zorder=1)

    ax.set_yticks(y)
    ax.set_yticklabels(df["group"], fontsize=15)
    ax.set_xlabel("Accuracy (%)", fontsize=16)
    ax.set_xlim(57, 106)
    # Hide tick labels beyond 100
    ax.set_xticks([60, 70, 80, 90, 100])
    ax.set_ylim(-0.7, len(df) - 0.3)
    ax.set_title("Ingredient matching accuracy by food group",
                 fontsize=18, fontweight="bold", pad=14)

    # Legend - far bottom left
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#43A047",
               markersize=10, label="Top-1 Accuracy"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="none",
               markeredgecolor="#888888", markersize=8, label=f"Top-{top_k} Accuracy"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=2,
              fontsize=14, framealpha=0.95, edgecolor="#DDDDDD")

    # Colour scale bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=60, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.018, pad=0.02)
    cbar.set_label("Top-1 Score (%)", fontsize=14)
    cbar.ax.tick_params(labelsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(left=False)
    ax.grid(axis="x", color="#E0E0E0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    fig.tight_layout()
    _save(fig, out)


def fig_b_score_buckets(df: pd.DataFrame, out: Path) -> None:
    """Grouped bar histogram: how many queries fall in each score bucket, correct vs incorrect."""
    bins   = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    labels = ["0.50–0.60", "0.60–0.70", "0.70–0.75",
              "0.75–0.80", "0.80–0.85", "0.85–0.90", "0.90–0.95", "0.95–1.00"]

    correct   = df[df["top1_correct"]]["top1_score"]
    incorrect = df[~df["top1_correct"]]["top1_score"]

    c_counts = [((correct   >= bins[i]) & (correct   < bins[i+1])).sum() for i in range(len(labels))]
    w_counts = [((incorrect >= bins[i]) & (incorrect < bins[i+1])).sum() for i in range(len(labels))]

    x     = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - width/2, c_counts, width, color=CORRECT,   alpha=0.85,
           label=f"Correct match (n={len(correct)})")
    ax.bar(x + width/2, w_counts, width, color=INCORRECT, alpha=0.85,
           label=f"Incorrect match (n={len(incorrect)})")

    for xi, (c, w) in enumerate(zip(c_counts, w_counts)):
        if c > 0:
            ax.text(xi - width/2, c + 0.15, str(c), ha="center", va="bottom", fontsize=12)
        if w > 0:
            ax.text(xi + width/2, w + 0.15, str(w), ha="center", va="bottom",
                    fontsize=12, color=INCORRECT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=14)
    ax.set_xlabel("Matching score range")
    ax.set_ylabel("Number of queries")
    ax.set_title("Matching score Distribution: correct vs incorrect matches",
                 fontsize=17, fontweight="bold")
    ax.legend(fontsize=14, frameon=False)
    fig.tight_layout()
    _save(fig, out)


def fig_b_group_variety(df: pd.DataFrame, out: Path) -> None:
    """Horizontal box + jitter per group, sorted by intra-group score std."""
    group_std    = df.groupby("group")["top1_score"].std().sort_values(ascending=True)
    groups_sorted = group_std.index.tolist()
    n = len(groups_sorted)

    cmap   = plt.get_cmap("tab20", n)
    colors = {g: cmap(i) for i, g in enumerate(groups_sorted)}

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.55)))

    for i, grp in enumerate(groups_sorted):
        sub = df[df["group"] == grp]["top1_score"].values
        c   = colors[grp]
        ax.boxplot(
            sub, positions=[i], vert=False, widths=0.5,
            patch_artist=True, notch=False, showfliers=False,
            boxprops=dict(facecolor=(*c[:3], 0.25), edgecolor=c),
            medianprops=dict(color=c, linewidth=2.2),
            whiskerprops=dict(color=c, linewidth=1.2),
            capprops=dict(color=c, linewidth=1.5),
        )
        rng = np.random.default_rng(seed=42)
        jit = rng.normal(i, 0.10, size=len(sub))
        correct_mask = df[df["group"] == grp]["top1_correct"].astype(bool).values
        ax.scatter(sub[correct_mask],  jit[correct_mask],
                   s=18, color=c, alpha=0.65, zorder=3)
        ax.scatter(sub[~correct_mask], jit[~correct_mask],
                   s=160, color=INCORRECT, marker="X", zorder=6)

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{g}  (σ={group_std[g]:.3f})" for g in groups_sorted], fontsize=13
    )
    ax.set_xlabel("Matching score", fontsize=15)
    ax.set_xlim(0.35, 1.08)
    ax.set_title("Score spread per food group  (sorted by within-group variability)",
                 fontsize=17, fontweight="bold", pad=12)
    ax.axvline(LINKING_TAU, color="#BDBDBD", linestyle="--", linewidth=1)

    handles = [
        plt.Line2D([0], [0], marker="X", color=INCORRECT, linestyle="none",
                   markersize=12, label="Incorrect match"),
        plt.Line2D([0], [0], color="#BDBDBD", linestyle="--", label=f"Threshold τ={LINKING_TAU}"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.07), ncol=3,
              fontsize=13, framealpha=0.92, edgecolor="#DDDDDD")

    ax.grid(axis="x", alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out)


def fig_b_combined(df_q: pd.DataFrame, df_groups: pd.DataFrame, out: Path) -> None:
    """Side-by-side: score spread (b6) + Top-1 accuracy lollipop (b5), shared y-axis."""
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    # Shared group order: ascending score std
    group_std    = df_q.groupby("group")["top1_score"].std().sort_values(ascending=True)
    groups_sorted = group_std.index.tolist()
    n = len(groups_sorted)

    cmap_grp  = plt.get_cmap("tab20", n)
    gcolors   = {g: cmap_grp(i) for i, g in enumerate(groups_sorted)}
    perf_cmap = LinearSegmentedColormap.from_list("perf", ["#E53935", "#FB8C00", "#43A047"])

    # Wider left panel (score spread), narrower right (lollipop)
    fig, (ax_box, ax_lol) = plt.subplots(
        1, 2, figsize=(26, max(10, n * 0.72)), sharey=True,
        gridspec_kw={"width_ratios": [5, 1]}
    )
    fig.subplots_adjust(wspace=0.04, left=0.17, right=0.97, top=0.93, bottom=0.13)

    # ── LEFT panel: score spread ───────────────────────────────────────────────
    for i, grp in enumerate(groups_sorted):
        sub = df_q[df_q["group"] == grp]["top1_score"].values
        c   = gcolors[grp]
        ax_box.boxplot(
            sub, positions=[i], vert=False, widths=0.52,
            patch_artist=True, notch=False, showfliers=False,
            boxprops=dict(facecolor=(*c[:3], 0.22), edgecolor=c),
            medianprops=dict(color=c, linewidth=2.5),
            whiskerprops=dict(color=c, linewidth=1.3),
            capprops=dict(color=c, linewidth=1.5),
        )
        rng  = np.random.default_rng(seed=42)
        jit  = rng.normal(i, 0.10, size=len(sub))
        mask = df_q[df_q["group"] == grp]["top1_correct"].astype(bool).values
        ax_box.scatter(sub[mask],  jit[mask],  s=28, color=c, alpha=0.6, zorder=3)
        ax_box.scatter(sub[~mask], jit[~mask], s=120,
                       color=INCORRECT, marker="x", linewidths=1.8, zorder=6)

    ax_box.set_xlabel("Matching score", fontsize=22)
    ax_box.set_xlim(0.44, 1.04)
    ax_box.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_box.axvline(LINKING_TAU, color="#BDBDBD", linestyle="--", linewidth=1.2)
    ax_box.set_title("(A)  Score spread per group", fontsize=22, fontweight="bold", pad=14)
    ax_box.set_yticks(range(n))
    ax_box.set_yticklabels(
        [f"{g}  (σ={group_std[g]:.3f})" for g in groups_sorted], fontsize=19
    )
    ax_box.set_ylim(-0.7, n - 0.3)
    ax_box.tick_params(axis="x", labelsize=19)
    ax_box.grid(axis="x", alpha=0.22, linewidth=0.7)
    ax_box.set_axisbelow(True)

    handles_a = [
        Line2D([0], [0], marker="x", color=INCORRECT, linestyle="none",
               markersize=14, markeredgewidth=2.2, label="Incorrect match"),
        Line2D([0], [0], color="#BDBDBD", linestyle="--", linewidth=1.5, label=f"τ = {LINKING_TAU}"),
    ]
    ax_box.legend(handles=handles_a, loc="upper center",
                  bbox_to_anchor=(0.5, -0.07), ncol=3,
                  fontsize=17, framealpha=0.95, edgecolor="#DDDDDD")

    # ── RIGHT panel: accuracy lollipop (compact) ─────────────────────────────
    df_lol = df_groups.set_index("group").reindex(groups_sorted).reset_index()
    perf_colors = [perf_cmap(float(v)) for v in df_lol["top1"].values]

    # Only Top-1 is shown: every correct match is ranked first, so Top-k and MRR equal Top-1
    for i, (_, row) in enumerate(df_lol.iterrows()):
        t1 = float(row["top1"]) * 100
        c  = perf_colors[i]
        ax_lol.plot([75, t1], [i, i], color=c, linewidth=1.5, solid_capstyle="round", zorder=2)
        ax_lol.scatter(t1, i, s=80, color=c, zorder=4, edgecolors="white", linewidths=0.8)
        ax_lol.text(101.5, i, f"{t1:.0f}%", va="center",
                    fontsize=19, color=c, fontweight="bold", clip_on=False)

    ax_lol.set_xlabel("Accuracy (%)", fontsize=22)
    ax_lol.set_xlim(73, 108)
    ax_lol.set_xticks([75, 85, 95])
    ax_lol.axvline(100, color="#BDBDBD", linestyle="--", linewidth=0.8, zorder=1)
    ax_lol.set_title("(B)  Top-1 accuracy",
                     fontsize=22, fontweight="bold", pad=14)
    ax_lol.tick_params(left=False, axis="x", labelsize=19)
    ax_lol.grid(axis="x", alpha=0.15, linewidth=0.5)
    ax_lol.set_axisbelow(True)

    fig.suptitle(
        "Ingredient matching: score spread and retrieval accuracy by food group",
        fontsize=22, fontweight="bold", y=1.01
    )
    _save(fig, out)


def fig_b_intra_variability(df: pd.DataFrame, out: Path) -> None:
    """Dumbbell range chart: min–max score per food concept, sorted by std."""
    if "primary_kw" not in df.columns:
        print("  [b7] skipping: 'primary_kw' column not in dataframe - run with --benchmark")
        return

    intra = (
        df.groupby(["group", "primary_kw"])["top1_score"]
        .agg(mean="mean", std="std", lo="min", hi="max", n="count")
        .reset_index()
        .dropna(subset=["std"])
    )
    intra = intra[intra["n"] > 1].sort_values("std", ascending=True).reset_index(drop=True)

    n_groups = df["group"].nunique()
    cmap     = plt.get_cmap("tab20", n_groups)
    grp_list = sorted(df["group"].unique())
    gcolor   = {g: cmap(i) for i, g in enumerate(grp_list)}

    n_rows = len(intra)
    fig, ax = plt.subplots(figsize=(10, max(8, n_rows * 0.38)))

    for i, row in intra.iterrows():
        c = gcolor[row["group"]]
        ax.plot([row["lo"], row["hi"]], [i, i], color=c, linewidth=2.2, alpha=0.7)
        ax.scatter(row["mean"], i, color=c, s=55, zorder=4, edgecolors="white", linewidths=0.8)
        ax.text(row["hi"] + 0.006, i,
                f"σ={row['std']:.3f}  n={int(row['n'])}",
                va="center", fontsize=9, color="#444")

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(intra["primary_kw"], fontsize=11)
    ax.set_xlabel("Top-1 matching score", fontsize=15)
    ax.set_xlim(0.35, 1.12)
    ax.set_title("Intra-element score variability across query variants",
                 fontsize=17, fontweight="bold", pad=12)

    legend_handles = [
        mpatches.Patch(facecolor=gcolor[g], label=g, alpha=0.8)
        for g in grp_list
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10,
              ncol=2, framealpha=0.92, edgecolor="#DDDDDD")

    ax.axvline(LINKING_TAU, color="#BDBDBD", linestyle="--", linewidth=1, alpha=0.7)
    ax.grid(axis="x", alpha=0.2, linewidth=0.7)
    ax.set_axisbelow(True)
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# PART C
# ══════════════════════════════════════════════════════════════════════════════

def fig_c_kg_summary(df: pd.DataFrame, out: Path) -> None:
    """Stacked bar per food showing found / correct breakdown."""
    foods   = df["food_name"].unique()
    found   = [df[df["food_name"] == f]["found"].mean()   for f in foods]
    correct = [df[df["food_name"] == f]["correct"].mean() for f in foods]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: per-food found & correct rate
    y = np.arange(len(foods))
    axes[0].barh(y, [c * 100 for c in correct], 0.4,
                 color=CORRECT, alpha=0.85, label="Suitability Correct")
    axes[0].barh(y, [f * 100 for f in found], 0.4,
                 color=NEUTRAL, alpha=0.4, label="Found in KG")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(foods, fontsize=14)
    axes[0].set_xlabel("Rate (%)")
    axes[0].set_xlim(0, 118)
    axes[0].axvline(100, color="gray", linestyle="--", linewidth=0.8, alpha=0.4)
    axes[0].set_title("Per-Food Correctness", fontsize=16, fontweight="bold")
    axes[0].legend(fontsize=14, frameon=False)

    # Right: overall pie
    total    = len(df)
    n_correct = df["correct"].sum()
    n_found   = df["found"].sum()
    n_wrong   = n_found - n_correct
    n_missing = total - n_found

    sizes  = [n_correct, n_wrong, n_missing]
    labels = [f"Correct\n({n_correct}/{total})",
              f"Wrong suitability\n({n_wrong})",
              f"Not found\n({n_missing})"]
    colors = [CORRECT, INCORRECT, NEUTRAL]
    wedges, texts, autotexts = axes[1].pie(
        sizes, labels=labels, colors=colors,
        autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)
    axes[1].set_title("Overall KG Correctness", fontsize=16, fontweight="bold")

    fig.suptitle("Part C - Knowledge Graph Disease Association Correctness",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# PART A - Extra ingredient analysis
# ══════════════════════════════════════════════════════════════════════════════

def _collect_extras(dfs: "dict[str, pd.DataFrame]") -> "dict[str, list[str]]":
    """Return {model: [extra_ingredient, ...]} from the extra_ingredients column."""
    result = {}
    for model, df in dfs.items():
        if "extra_ingredients" not in df.columns:
            continue
        extras = []
        for cell in df["extra_ingredients"].dropna():
            if cell:
                extras.extend(cell.split("|"))
        result[model] = [e.strip() for e in extras if e.strip()]
    return result


def fig_a_extra_wordclouds(dfs: "dict[str, pd.DataFrame]", out: Path) -> None:
    """Word cloud of extra (unmatched) ingredients - one panel per model + combined."""
    from wordcloud import WordCloud
    from collections import Counter

    extras_by_model = _collect_extras(dfs)
    if not extras_by_model:
        return

    all_extras = [e for lst in extras_by_model.values() for e in lst]
    models     = list(extras_by_model.keys())
    n_panels   = len(models) + 1          # +1 for combined

    cols = min(n_panels, 3)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(6 * cols, 4.5 * rows),
                             facecolor="#FAFAFA")
    axes_flat = np.array(axes).flatten()

    def _make_cloud(words: list[str], ax, title: str, color: str) -> None:
        if not words:
            ax.set_visible(False)
            return
        freq = Counter(words)
        wc = WordCloud(
            width=600, height=380,
            background_color="white",
            colormap="RdYlGn_r",
            max_words=60,
            prefer_horizontal=0.85,
            collocations=False,
        ).generate_from_frequencies(freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=16, fontweight="bold", pad=8, color=color)
        ax.axis("off")

    for i, model in enumerate(models):
        short = model.split(":")[0]
        _make_cloud(extras_by_model[model], axes_flat[i],
                    f"{short}\n({len(extras_by_model[model])} extras)", "#37474F")

    # Combined panel
    _make_cloud(all_extras, axes_flat[len(models)],
                f"All models combined\n({len(all_extras)} extras)", "#B71C1C")

    # Hide unused axes
    for ax in axes_flat[n_panels:]:
        ax.set_visible(False)

    fig.suptitle("Part A - Ingredients Extracted by LLMs Not in Ground Truth",
                 fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out)


def fig_a_extra_bar(dfs: "dict[str, pd.DataFrame]", out: Path, top_n: int = 20) -> None:
    """Horizontal bar chart of top extra ingredients across all models."""
    from collections import Counter

    extras_by_model = _collect_extras(dfs)
    if not extras_by_model:
        return

    all_extras = [e for lst in extras_by_model.values() for e in lst]
    freq = Counter(all_extras).most_common(top_n)
    if not freq:
        return

    labels = [f[0] for f in reversed(freq)]
    counts = [f[1] for f in reversed(freq)]

    # Colour by per-model agreement (how many models mentioned it)
    n_models   = len(extras_by_model)
    model_sets = [set(v) for v in extras_by_model.values()]
    agreement  = [sum(1 for s in model_sets if lbl in s) / n_models for lbl in labels]
    cmap       = plt.cm.RdYlGn
    colors     = [cmap(a) for a in agreement]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    bars = ax.barh(labels, counts, color=colors, edgecolor="white", linewidth=0.5)

    for bar, cnt, agr in zip(bars, counts, agreement):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{cnt}  ({agr:.0%})", va="center", fontsize=12, color="#555")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Model agreement", fontsize=14)
    cbar.ax.tick_params(labelsize=8)

    ax.set_xlabel("Times extracted across all models & recipes")
    ax.set_title(f"Part A - Top {top_n} Extra Ingredients Not in Ground Truth",
                 fontsize=17, fontweight="bold")
    ax.set_xlim(0, max(counts) * 1.25)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    _save(fig, out)


# ── Model-agreement analysis ──────────────────────────────────────────────────

def _build_agreement_df(dfs: "dict[str, pd.DataFrame]") -> "tuple[pd.DataFrame, int]":
    """
    For each (recipe, ingredient) pair found in any model's extras, count how
    many models extracted it.  Returns a DataFrame with columns
    [ingredient, recipe, n_models] and the total number of models.
    """
    model_recipe_extras: dict[str, dict[str, set]] = {}
    for model, df in dfs.items():
        if "extra_ingredients" not in df.columns:
            continue
        model_recipe_extras[model] = {}
        for _, row in df.iterrows():
            recipe = row["recipe"]
            cell   = row.get("extra_ingredients", "")
            extras: set[str] = set()
            if cell and not (isinstance(cell, float)):
                extras = {e.strip() for e in str(cell).split("|") if e.strip()}
            model_recipe_extras[model][recipe] = extras

    n_models = len(model_recipe_extras)
    rows = []
    all_recipes = {r for m_data in model_recipe_extras.values() for r in m_data}
    for recipe in all_recipes:
        ingr_count: dict[str, int] = {}
        for m_data in model_recipe_extras.values():
            for ingr in m_data.get(recipe, set()):
                ingr_count[ingr] = ingr_count.get(ingr, 0) + 1
        for ingr, cnt in ingr_count.items():
            rows.append({"ingredient": ingr, "recipe": recipe, "n_models": cnt})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ingredient", "recipe", "n_models"]), n_models


def fig_a_extra_agreement(dfs: "dict[str, pd.DataFrame]", out: Path) -> None:
    """
    Agreement analysis: distribution bar (top) + wide ranked pill grid (bottom).
    Rows = agreement levels top-to-bottom (universal → subjective).
    Pills sorted left-to-right by extraction count.
    """
    from collections import Counter

    agree_df, n_models = _build_agreement_df(dfs)
    if agree_df.empty:
        return

    cmap   = plt.cm.RdYlGn
    levels = list(range(1, n_models + 1))

    ingr_stats = (
        agree_df.groupby("ingredient")
        .agg(total_count=("n_models", "sum"),
             max_agreement=("n_models", "max"),
             n_recipes=("recipe", "nunique"))
        .reset_index()
    )

    # ── Layout: thin bar row on top, tall grid canvas below ──────────────────
    max_cols  = 8
    cell_w    = 2.6
    cell_h    = 1.1
    grid_w    = max_cols * cell_w
    grid_h    = n_models * cell_h

    fig = plt.figure(figsize=(grid_w + 1.8, grid_h + 4.5))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[2, 3.5], hspace=0.45)
    ax_bar  = fig.add_subplot(gs[0])
    ax_grid = fig.add_subplot(gs[1])

    fig.suptitle("Model agreement analysis in over-extraction of ingredients",
                 fontsize=18, fontweight="bold")

    # ── Top: distribution bar ─────────────────────────────────────────────────
    level_counts = Counter(agree_df["n_models"].values)
    counts     = [level_counts.get(lv, 0) for lv in levels]
    bar_colors = [cmap(lv / n_models) for lv in levels]
    bars = ax_bar.bar(levels, counts, color=bar_colors,
                      edgecolor="white", linewidth=0.8, zorder=3)

    total_pairs = len(agree_df)
    for lv, bar, cnt in zip(levels, bars, counts):
        if cnt == 0:
            continue
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                    str(cnt), ha="center", va="bottom", fontsize=16, fontweight="bold")

    tier_xlabels = {}
    for lv in levels:
        cnt  = level_counts.get(lv, 0)
        pct  = cnt / total_pairs * 100
        tier = {1: "subjective", n_models: "universal"}.get(lv, "")
        n_str = f"{lv} model" + ("s" if lv > 1 else "")
        pct_str = f"{pct:.0f}%"
        tier_xlabels[lv] = f"{n_str}\n{pct_str}" + (f"\n({tier})" if tier else "")

    ax_bar.set_xticks(levels)
    ax_bar.set_xticklabels([tier_xlabels[lv] for lv in levels], fontsize=16)
    ax_bar.set_ylabel("Pairs (recipe × ingredient)", fontsize=16)
    ax_bar.set_title("Agreement level distribution", fontsize=18, fontweight="bold")
    ax_bar.set_axisbelow(True)
    ax_bar.grid(axis="y", alpha=0.25)
    for spine in ["top", "right"]:
        ax_bar.spines[spine].set_visible(False)

    # ── Bottom: wide pill grid ────────────────────────────────────────────────
    ax_grid.set_xlim(0, grid_w)
    ax_grid.set_ylim(-0.15, grid_h + 0.15)
    ax_grid.axis("off")

    max_count = ingr_stats["total_count"].max()

    # Draw rows top-to-bottom: universal (n_models) first
    for row_idx, lv in enumerate(reversed(levels)):
        y0     = (n_models - 1 - row_idx) * cell_h
        color  = cmap(lv / n_models)
        subset = (ingr_stats[ingr_stats["max_agreement"] == lv]
                  .sort_values("total_count", ascending=False)
                  .head(max_cols))

        # Row header strip on the far left
        tier_label = {1: "subjective", n_models: "universal"}.get(lv, "")
        header_color = "#E53935" if lv == 1 else ("#43A047" if lv == n_models else "#888")
        ax_grid.text(
            -0.06, y0 + cell_h * 0.5,
            f"{lv}{'★' if tier_label else ''}\n{tier_label}" if tier_label
            else str(lv),
            ha="right", va="center", fontsize=16,
            color=header_color, fontweight="bold",
            transform=ax_grid.transData
        )

        # Light row band
        ax_grid.add_patch(mpatches.FancyBboxPatch(
            (0, y0 + 0.03), grid_w, cell_h - 0.06,
            boxstyle="round,pad=0.0",
            facecolor=color, alpha=0.07,
            edgecolor="none", zorder=0
        ))

        for rank, (_, ingr_row) in enumerate(subset.iterrows()):
            x0    = rank * cell_w
            alpha = 0.22 + 0.68 * (ingr_row["total_count"] / max_count)

            pill = mpatches.FancyBboxPatch(
                (x0 + 0.10, y0 + 0.12), cell_w - 0.20, cell_h - 0.24,
                boxstyle="round,pad=0.12",
                facecolor=color, alpha=alpha,
                edgecolor="white", linewidth=1.5, zorder=2
            )
            ax_grid.add_patch(pill)

            # Use white text on dark pills, dark text on light ones
            r, g, b, _ = plt.cm.colors.to_rgba(color) if hasattr(plt.cm, "colors") else (*color[:3], 1)
            r, g, b = color[0], color[1], color[2]
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            effective_darkness = 1 - (luminance * (1 - alpha) + alpha * luminance)
            txt_color  = "white" if alpha * (1 - luminance) > 0.35 else "#111"
            txt_color2 = "white" if alpha * (1 - luminance) > 0.35 else "#444"

            ax_grid.text(
                x0 + cell_w * 0.5, y0 + cell_h * 0.62,
                ingr_row["ingredient"],
                ha="center", va="center", fontsize=16,
                fontweight="bold", color=txt_color, zorder=3
            )
            ax_grid.text(
                x0 + cell_w * 0.5, y0 + cell_h * 0.25,
                f"×{int(ingr_row['total_count'])}",
                ha="center", va="center", fontsize=14,
                color=txt_color2, zorder=3
            )

    ax_grid.set_title("Top extras per agreement level  (sorted by count)",
                      fontsize=18, fontweight="bold", pad=8)

    _save(fig, out)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def _load_and_merge_llm_csvs(result_dirs: list[Path]) -> "dict[str, pd.DataFrame]":
    """Load part_a_llm_*.csv from all dirs, merge rows for same model."""
    from collections import defaultdict
    frames: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for res in result_dirs:
        for path in sorted(res.glob("part_a_llm_*.csv")):
            model = path.stem.replace("part_a_llm_", "").replace("_", ":")
            if model == "part_a_llm":
                model = "llama3"
            frames[model].append(pd.read_csv(path))
    return {m: pd.concat(dfs, ignore_index=True) for m, dfs in frames.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", default=["results/batch1", "results/batch2"], metavar="DIR",
                        help="One or more results directories to merge (Part A is combined, "
                             "Part B/C taken from first dir that has them)")
    parser.add_argument("--benchmark", default="benchmark.json", metavar="FILE",
                        help="Path to benchmark.json used for Part B (needed for intra-variability figure)")
    parser.add_argument("--format",  default="png", choices=["png", "pdf"])
    parser.add_argument("--out",     default=None,  metavar="DIR",
                        help="Output directory for figures (default: figures/ next to first --results dir)")
    args = parser.parse_args()

    result_dirs = [Path(r) for r in args.results]
    out = Path(args.out) if args.out else result_dirs[0].parent / "figures"
    out.mkdir(exist_ok=True)
    fmt = args.format

    print(f"\nReading from : {', '.join(str(r) for r in result_dirs)}")
    print(f"Writing to   : {out}/\n")

    # ── Part A - merge across all result dirs ─────────────────────────────────
    llm_dfs = _load_and_merge_llm_csvs(result_dirs)

    for model, df in llm_dfs.items():
        safe = re.sub(r"[^a-z0-9]", "_", model.lower())
        fig_a_scatter_extraction(df, out / f"a2_extraction_scatter_{safe}.{fmt}", model)
        n = len(df)
        print(f"  Loaded {model}: {n} recipes")

    if llm_dfs:
        fig_a_metric_distributions(llm_dfs,  out / f"a1_metric_distributions.{fmt}")
        fig_a_f1_comparison(llm_dfs,         out / f"a5_f1_comparison.{fmt}")
        fig_a_extra_wordclouds(llm_dfs,      out / f"a6_extra_wordclouds.{fmt}")
        fig_a_extra_bar(llm_dfs,             out / f"a7_extra_ingredients_bar.{fmt}")
        fig_a_extra_agreement(llm_dfs,       out / f"a8_extra_agreement.{fmt}")

    if len(llm_dfs) > 1:
        fig_a_model_comparison(llm_dfs, out / f"a3_model_comparison.{fmt}")
        fig_a_f1_boxplot(llm_dfs,       out / f"a4_f1_boxplot.{fmt}")
    elif len(llm_dfs) == 1:
        print("  (only 1 model - skipping model comparison charts)")

    # ── Part B/C - first dir that has the files ───────────────────────────────
    def _find(filename: str) -> "Path | None":
        for d in result_dirs:
            p = d / filename
            if p.exists():
                return p
        return None

    query_csv = _find("part_b_matching.csv")
    group_csv = _find("part_b_by_group.csv")
    kg_csv    = _find("part_c_kg.csv")

    if query_csv:
        df_q  = pd.read_csv(query_csv)
        top_k = int(
            [c for c in df_q.columns if c.startswith("top") and "correct" in c][0]
            .replace("top","").replace("_correct","")
        )
        # Enrich with primary_kw from benchmark for intra-variability analysis
        bench_path = Path(args.benchmark)
        if bench_path.exists():
            import json as _json
            with open(bench_path) as _f:
                _bench = _json.load(_f)
            _kw_map = {q["query"]: q["key_words"][0] for q in _bench["matching_queries"]}
            df_q["primary_kw"] = df_q["query"].map(_kw_map)
        fig_b_score_distribution(df_q, out / f"b1_score_distribution.{fmt}", top_k)
        fig_b_score_buckets(df_q,      out / f"b3_score_buckets.{fmt}")
        if "group" in df_q.columns:
            fig_b_score_violin_by_group(df_q, out / f"b2_score_violin_by_group.{fmt}")
            fig_b_group_variety(df_q,         out / f"b6_group_variety.{fmt}")
            fig_b_intra_variability(df_q,     out / f"b7_intra_variability.{fmt}")

    if group_csv:
        df_g  = pd.read_csv(group_csv)
        top_k = int(
            [c for c in df_g.columns if c.startswith("top") and c != "top1"][0]
            .replace("top","")
        )
        # Merge per-group score std from df_q so b5 and b6 share the same sort order
        if query_csv:
            _grp_std = (
                pd.read_csv(query_csv)
                .groupby("group")["top1_score"].std()
                .rename("score_std").reset_index()
            )
            df_g = df_g.merge(_grp_std, on="group", how="left")
        fig_b_group_accuracy(df_g, out / f"b4_group_accuracy.{fmt}", top_k)
        fig_b_group_lollipop(df_g, out / f"b5_group_lollipop.{fmt}", top_k)
        if query_csv:
            fig_b_combined(df_q, df_g, out / f"b56_combined.{fmt}")

    if kg_csv:
        df_kg = pd.read_csv(kg_csv)
        df_kg.rename(columns={"food": "food_name"}, inplace=True)
        fig_c_kg_summary(df_kg, out / f"c1_kg_correctness.{fmt}")

    print(f"\nAll figures saved to {out}/\n")


if __name__ == "__main__":
    main()
