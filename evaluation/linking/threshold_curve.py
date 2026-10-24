#!/usr/bin/env python3
"""
Precision and recall of the matcher's accept decision as a function of τ.

A match is accepted when its score is >= τ. Two sources of correctness:
    benchmark   the 200 surface-form queries (../results/part_b_matching.csv),
                correct when the top-1 match contains a key word (evaluate.py)
    survey      the 23 matches judged in the human evaluation
                (../../survey/analysis/results/matching_summary.csv),
                incorrect for the three known-incorrect controls (M02–M04),
                correct for the 20 system matches
For each source and τ = 0.50 ... 0.95:
    precision   accepted and correct / accepted
    recall      accepted and correct / correct

Usage
-----
    cd evaluation/linking
    python threshold_curve.py              # vertical line at τ = 0.68
    python threshold_curve.py --tau 0.75

Output: results/threshold_curve.csv, results/threshold_curve.png, results/threshold_curve.pdf
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LINKING_DIR = Path(__file__).parent
RESULTS_DIR = LINKING_DIR / "results"
BENCHMARK = LINKING_DIR.parent / "results" / "part_b_matching.csv"
SURVEY = LINKING_DIR.parents[1] / "survey" / "analysis" / "results" / "matching_summary.csv"
TAUS = np.round(np.arange(0.50, 0.951, 0.01), 2)
CONTROLS = ["M02", "M03", "M04"]  # known-incorrect matches shown in every survey block


def load_sources() -> dict:
    bench = pd.read_csv(BENCHMARK)
    survey = pd.read_csv(SURVEY)
    return {
        "Benchmark": pd.DataFrame({"score": bench["top1_score"],
                                   "correct": bench["top1_correct"].astype(str).str.lower().eq("true")}),
        # survey scores are on a 0–100 scale; the known-incorrect controls are the
        # errors (annotators accept two of them by majority, so the vote is not used)
        "Human evaluation": pd.DataFrame({"score": survey["score"] / 100,
                                          "correct": ~survey["match_id"].isin(CONTROLS)}),
    }


def curve(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tau in TAUS:
        accepted = df["score"] >= tau
        tp = (accepted & df["correct"]).sum()
        rows.append({"tau": tau, "accepted": int(accepted.sum()), "wrong_accepted": int((accepted & ~df["correct"]).sum()),
                     "precision": 100 * tp / accepted.sum() if accepted.sum() else np.nan,
                     "recall": 100 * tp / df["correct"].sum()})
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tau", type=float, default=0.68, help="threshold marked with a vertical line")
    args = p.parse_args()

    sources = load_sources()
    curves = pd.concat([curve(df).assign(source=name) for name, df in sources.items()])
    RESULTS_DIR.mkdir(exist_ok=True)
    curves.to_csv(RESULTS_DIR / "threshold_curve.csv", index=False)
    for name, df in sources.items():
        at = curves[(curves["source"] == name) & np.isclose(curves["tau"], args.tau)].iloc[0]
        print(f"{name}: {len(df)} matches, {(~df['correct']).sum()} incorrect; at τ = {args.tau}: "
              f"precision {at['precision']:.1f}, recall {at['recall']:.1f}, {at['wrong_accepted']} incorrect accepted")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = {"Benchmark": "#1f77b4", "Human evaluation": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for name in sources:
        c = curves[curves["source"] == name]
        ax.step(c["tau"], c["precision"], where="post", color=colors[name], linewidth=1.8)
        ax.step(c["tau"], c["recall"], where="post", color=colors[name], linewidth=1.8, linestyle="--")
    ax.axvline(args.tau, color="gray", linestyle=":", linewidth=1.2)
    ax.text(args.tau + 0.005, 104, f"τ = {args.tau}", color="gray", fontsize=9, va="bottom")
    ax.set_xlim(TAUS[0], TAUS[-1])
    ax.set_ylim(0, 103)
    ax.set_xlabel("Acceptance threshold τ")
    ax.set_ylabel("%")
    ax.grid(alpha=0.3)
    handles = [Line2D([], [], color=colors[name], linewidth=1.8,
                      label=name) for name in sources]
    handles += [Line2D([], [], color="black", linewidth=1.5, label="Precision"),
                Line2D([], [], color="black", linewidth=1.5, linestyle="--", label="Recall")]
    ax.legend(handles=handles, fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(RESULTS_DIR / f"threshold_curve.{ext}", dpi=300)
    print(f"Written {RESULTS_DIR}/threshold_curve.csv, .png, .pdf")


if __name__ == "__main__":
    main()
