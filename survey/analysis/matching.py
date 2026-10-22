"""
Ingredient-matching task: do humans judge the automatic ingredient →
food-entry matches as correct?

- Per-match distribution of Perfect / Acceptable / Incorrect
- Majority vote per match (correct = Perfect or Acceptable)
- Breakdown by food group
- Automated evaluation's own ground truth (evaluation/results/
  part_b_matching.csv) vs human judgment
- How well the similarity score separates human-judged correct from
  incorrect matches (AUC, Youden-optimal threshold)

Note: M02–M04 are known-incorrect matches shown in every block (there are
no more real errors in the data), so they have ~4x more raters than the rest.

Usage:
    python matching.py [path/to/export.xlsx]
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import (
    JUDGMENT_ORDER,
    PART_B_CSV,
    SurveyData,
    load,
    pct,
    save_table,
    section,
    xlsx_from_argv,
)


def match_summary(matches: pd.DataFrame) -> pd.DataFrame:
    shares = pd.crosstab(matches.match_id, matches.label, normalize="index").reindex(columns=JUDGMENT_ORDER, fill_value=0)
    meta = matches.groupby("match_id")[["query", "food_group", "matched_food", "score"]].first()
    out = meta.join((shares * 100).round(1).add_prefix("pct_"))
    out["pct_human_correct"] = (out.pct_perfect + out.pct_acceptable).round(1)
    out["n_raters"] = matches.groupby("match_id").size()
    out["majority"] = np.select(
        [out.pct_human_correct > 50, out.pct_human_correct < 50], ["correct", "incorrect"], "tie"
    )
    if PART_B_CSV.exists():
        auto = pd.read_csv(PART_B_CSV)[["query", "top1_correct"]].drop_duplicates("query")
        out = out.reset_index().merge(auto, on="query", how="left").set_index("match_id")
    return out.reset_index()


def by_food_group(matches: pd.DataFrame) -> pd.DataFrame:
    return (
        matches.groupby("food_group")
        .agg(
            n_matches=("match_id", "nunique"),
            mean_score=("score", "mean"),
            pct_perfect=("label", lambda s: pct((s == "perfect").mean())),
            pct_incorrect=("label", lambda s: pct((s == "incorrect").mean())),
        )
        .round(1)
        .sort_values("pct_incorrect", ascending=False)
        .reset_index()
    )


def score_threshold(matches: pd.DataFrame) -> dict:
    """Judgment-level ROC: does the similarity score separate matches humans
    call correct from those they call incorrect?"""
    pos = matches.loc[matches.human_correct, "score"]
    neg = matches.loc[~matches.human_correct, "score"]
    u, p = stats.mannwhitneyu(pos, neg, alternative="greater")
    best = max(
        (
            ((matches.score >= t) & matches.human_correct).sum() / len(pos)
            - ((matches.score >= t) & ~matches.human_correct).sum() / len(neg),
            t,
        )
        for t in np.unique(matches.score)
    )
    t = best[1]
    return {
        "auc": round(u / (len(pos) * len(neg)), 3),
        "mannwhitney_p": float(p),
        "best_threshold": float(t),
        "tpr_pct": pct(((matches.score >= t) & matches.human_correct).sum() / len(pos)),
        "fpr_pct": pct(((matches.score >= t) & ~matches.human_correct).sum() / len(neg)),
        "youdens_j": round(best[0], 3),
    }


def run(data: SurveyData) -> str:
    section("INGREDIENT MATCHING")
    m = data.matches
    summary = match_summary(m)
    groups = by_food_group(m)
    thr = score_threshold(m)
    save_table(summary, "matching_summary")
    save_table(groups, "matching_food_group")
    save_table(pd.DataFrame([thr]), "matching_score_threshold")
    print(summary.to_string(index=False))
    print(groups.to_string(index=False))
    print(thr)

    share = m.label.value_counts(normalize=True)
    n_correct = (summary.majority == "correct").sum()
    auto_md = ""
    if "top1_correct" in summary:
        both = summary.dropna(subset=["top1_correct"])
        agree = (both.top1_correct == (both.majority == "correct")).sum()
        auto_md = (
            f"\nThe automated evaluation's ground truth and the human majority agree on {agree}/{len(both)} matches. "
            "Disagreements:\n\n"
            + both[both.top1_correct != (both.majority == "correct")][
                ["match_id", "query", "matched_food", "score", "top1_correct", "pct_human_correct"]
            ].to_markdown(index=False)
        )
    return f"""## 6. Ingredient matching

{len(m)} judgments on {m.match_id.nunique()} matches from {m.respondent_id.nunique()} respondents.
Overall: {pct(share.get('perfect', 0))}% Perfect, {pct(share.get('acceptable', 0))}% Acceptable,
{pct(share.get('incorrect', 0))}% Incorrect. By majority vote, {n_correct}/{len(summary)} matches are judged correct.
M02–M04 are known-incorrect matches deliberately shown in every block.
{auto_md}

### Similarity score vs human judgment

AUC = {thr['auc']} (Mann–Whitney p = {thr['mannwhitney_p']:.2g}). The best single threshold is {thr['best_threshold']:.1f}:
it keeps {thr['tpr_pct']}% of human-correct matches and still lets through {thr['fpr_pct']}% of human-incorrect ones
(Youden's J = {thr['youdens_j']}). Use it as a "send to human review below this score" heuristic, not as a hard rule.



### By food group

{groups.to_markdown(index=False)}
"""


if __name__ == "__main__":
    print(run(load(xlsx_from_argv())))
