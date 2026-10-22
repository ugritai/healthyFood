"""
Error analysis: which relations annotators reject or cannot judge, and why.

- Most-rejected relations (highest % Disagree)
- Not sure by health outcome: does uncertainty track the outcome (how
  mature its evidence is) rather than the food?
- Claim direction: protective ("lower risk") vs risk-increasing
  ("higher risk") disease relations
- Oversimplified flags by relation type
- Difficulty ranking by vote entropy (how split the votes are)

Usage:
    python error_analysis.py [path/to/export.xlsx]
"""

import re

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    CORRECTNESS_ORDER,
    ITEMS_CSV,
    TYPE_LABELS,
    TYPE_ORDER,
    SurveyData,
    load,
    normalized_entropy,
    pct,
    rating_matrix,
    save_table,
    section,
    xlsx_from_argv,
)
from descriptive import item_summary


def most_rejected(summary: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    return summary.sort_values("pct_disagree", ascending=False).head(n)[
        ["item_id", "type", "food", "target", "pct_agree", "pct_not_sure", "pct_disagree", "n_raters"]
    ]


def not_sure_by_target(items: pd.DataFrame) -> pd.DataFrame:
    """Disease outcomes evaluated with more than one food, so the effect of
    the outcome can be separated from that of a single food."""
    disease = items[items["type"] == "disease"]
    agg = disease.groupby("target").agg(
        n_foods=("food", "nunique"),
        n_judgments=("label", "size"),
        pct_not_sure=("label", lambda s: pct((s == "not_sure").mean())),
        pct_disagree=("label", lambda s: pct((s == "disagree").mean())),
    )
    return agg[agg.n_foods > 1].sort_values("pct_not_sure", ascending=False).reset_index()


def direction(items: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Disease relation texts bold the risk direction ('**lower** risk of ...')."""
    relations = pd.read_csv(ITEMS_CSV)
    relations["direction"] = relations["relation_text"].apply(
        lambda t: (m.group(1) if (m := re.search(r"\*\*(lower|higher)\*\*", str(t))) else None)
    )
    merged = items.merge(relations.dropna(subset=["direction"])[["item_id", "direction"]], on="item_id")
    agg = merged.groupby("direction").agg(
        n_items=("item_id", "nunique"),
        n_judgments=("label", "size"),
        pct_agree=("label", lambda s: pct((s == "agree").mean())),
        pct_disagree=("label", lambda s: pct((s == "disagree").mean())),
        pct_not_sure=("label", lambda s: pct((s == "not_sure").mean())),
    )
    ordinal = merged["label"].map({"agree": 1.0, "not_sure": 0.5, "disagree": 0.0})
    _, p = stats.mannwhitneyu(ordinal[merged.direction == "lower"], ordinal[merged.direction == "higher"])
    return agg.reset_index(), p


def difficulty(items: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    h = normalized_entropy(rating_matrix(items, "item_id", CORRECTNESS_ORDER)).round(2)
    return summary.merge(h.reset_index(), on="item_id").sort_values("difficulty", ascending=False)


def disagree_vs_oversimplified(items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Is a Disagree vote a rejection of the claim, or a sign the annotator
    thinks the claim is too simplified? The two questions were asked
    separately, so this is an association, not the stated reason."""
    rows = []
    for t in [*TYPE_ORDER, "all"]:
        df = items if t == "all" else items[items.type == t]
        rows.append({"type": t, **{f"pct_oversimplified_{lab}": pct(df[df.label == lab].oversimplified_flag.mean())
                                   for lab in CORRECTNESS_ORDER}})
    by_label = pd.DataFrame(rows)

    decisive = items[items.label != "not_sure"]
    odds_ratio, p_fisher = stats.fisher_exact(pd.crosstab(decisive.label, decisive.oversimplified_flag))
    # Within-annotator: the same person, when disagreeing vs agreeing.
    per_annotator = items.groupby(["respondent_id", "label"]).oversimplified_flag.mean().unstack()
    paired = per_annotator.dropna(subset=["agree", "disagree"])
    _, p_paired = stats.wilcoxon(paired.agree, paired.disagree)
    # Sensitivity: count as rejections only Disagree votes not flagged as oversimplified.
    strict = items.assign(strict_disagree=(items.label == "disagree") & ~items.oversimplified_flag)
    votes = strict.groupby("item_id").agg(
        n_agree=("label", lambda s: (s == "agree").sum()), n_strict=("strict_disagree", "sum")
    )
    verdict = np.sign(votes.n_agree - votes.n_strict).map({1: "accept", -1: "reject", 0: "tie"})
    stats_ = {
        "odds_ratio": odds_ratio, "p_fisher": p_fisher,
        "n_paired": len(paired), "paired_agree": pct(paired.agree.mean()),
        "paired_disagree": pct(paired.disagree.mean()), "p_paired": p_paired,
        "pct_disagree_not_flagged": pct(1 - items[items.label == "disagree"].oversimplified_flag.mean()),
        "strict_accept": int((verdict == "accept").sum()), "strict_reject": int((verdict == "reject").sum()),
        "strict_tie": int((verdict == "tie").sum()),
    }
    return by_label, stats_


def run(data: SurveyData) -> str:
    section("ERROR ANALYSIS")
    items = data.items
    summary = item_summary(items)
    rejected = most_rejected(summary)
    by_target = not_sure_by_target(items)
    by_direction, p_dir = direction(items)
    diff = difficulty(items, summary)
    over = pd.DataFrame(
        {"type": TYPE_ORDER, "pct_oversimplified": [pct(items[items.type == t].oversimplified_flag.mean()) for t in TYPE_ORDER]}
    )
    over_agree = pct(items[items.label == "agree"].oversimplified_flag.mean())
    over_by_label, ov = disagree_vs_oversimplified(items)
    strict_precision = pct(ov["strict_accept"] / (ov["strict_accept"] + ov["strict_reject"]))
    for name, df in [("error_most_rejected", rejected), ("error_not_sure_by_outcome", by_target),
                     ("error_direction", by_direction), ("error_difficulty", diff),
                     ("error_oversimplified_by_label", over_by_label)]:
        save_table(df, name)
    print(rejected.to_string(index=False))
    print(by_target.to_string(index=False))
    print(by_direction.to_string(index=False), f"\nMann-Whitney p = {p_dir:.3g}")

    top = by_target.iloc[0]
    low = by_target.iloc[-1]
    return f"""## 5. Error analysis

### Most-rejected relations

{rejected.to_markdown(index=False)}

The relations rejected most often involve contested or evolving epidemiological evidence (e.g. eggs and coronary
heart disease, coffee and bladder cancer, soy and breast cancer). In these cases the source study supports the stored
direction, but current dietetic training emphasises a different or null effect.

### Not sure depends on the health outcome

Disease outcomes evaluated with more than one food:

{by_target.to_markdown(index=False)}

*Not sure* ranges from {top.pct_not_sure}% ({top.target}) to {low.pct_not_sure}% ({low.target}). Because
each outcome is paired with several different foods, this suggests uncertainty reflects how mature the evidence
is for an outcome, rather than noise in particular graph edges.


### Claim direction (disease relations)

{by_direction.to_markdown(index=False)}

Protective vs risk-increasing claims: Mann–Whitney p = {p_dir:.3g} (ordinal Agree=1, Not sure=0.5, Disagree=0).

### Oversimplified

{over.replace({'type': TYPE_LABELS}).to_markdown(index=False)}

Even among judgments that **agreed** with the relation, {over_agree}% flagged it as oversimplified.
Binary food–outcome edges lose dose, preparation and population context, which is a limitation worth stating.

### Is Disagree a rejection or an oversimplification flag?

% of judgments flagged as oversimplified, by correctness answer:

{over_by_label.replace({'type': TYPE_LABELS}).to_markdown(index=False)}

Disagree votes are flagged as oversimplified far more often than Agree votes
(odds ratio {ov['odds_ratio']:.2f}, Fisher p = {ov['p_fisher']:.2g}). The same holds within annotators: the
{ov['n_paired']} annotators who gave both answers flagged {ov['paired_disagree']}% of their Disagree votes vs
{ov['paired_agree']}% of their Agree votes (Wilcoxon p = {ov['p_paired']:.2g}). {ov['pct_disagree_not_flagged']}% of Disagree
votes were *not* flagged, so part of the disagreement is an outright rejection.

Sensitivity (upper bound): counting only unflagged Disagree votes as rejections gives {ov['strict_accept']} accepted,
{ov['strict_reject']} rejected, {ov['strict_tie']} tied (precision {strict_precision}%). The two questions were asked separately, so
this shows an association, not the annotator's stated reason.

### Hardest relations (vote entropy; 1 = maximally split)

{diff.head(8)[['item_id', 'food', 'target', 'pct_agree', 'pct_not_sure', 'pct_disagree', 'difficulty']].to_markdown(index=False)}
"""


if __name__ == "__main__":
    print(run(load(xlsx_from_argv())))
