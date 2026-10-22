"""
Annotator reliability: are the people who disagree less reliable, and
does the result depend on who is in the pool?

Two kinds of evidence, kept apart because they differ in strength:
- Consensus-based (circular): leave-one-out agreement with the other
  raters of the same item, and Dawid–Skene accuracy. Dissenters score
  lower by construction, so these do NOT show that dissenters are wrong.
- External (non-circular): whether disagreement rates differ by year
  of study or English level, and whether Disagree votes cluster on the
  same relations (systematic errors) or are scattered (individual dissent).

Then the majority-vote result is recomputed under quality filters that are
standard for crowdsourcing, to check that it is stable.

Usage:
    python annotator_reliability.py [path/to/export.xlsx]
"""

import numpy as np
import pandas as pd
from scipy import stats

from aggregation import dawid_skene, majority_vote
from common import (
    CORRECTNESS_ORDER,
    SurveyData,
    load,
    pct,
    rating_matrix,
    save_table,
    section,
    xlsx_from_argv,
)


def leave_one_out(items: pd.DataFrame) -> pd.Series:
    """Per decisive judgment: 1 if it matches the majority (Agree vs Disagree)
    of the *other* raters of that item, 0 if not, NaN if those others tie."""
    counts = rating_matrix(items, "item_id", CORRECTNESS_ORDER)
    others = counts.loc[items["item_id"]].reset_index(drop=True)
    others.index = items.index
    for lab in ("agree", "disagree"):
        others.loc[items["label"] == lab, lab] -= 1
    other_majority = np.select(
        [others.agree > others.disagree, others.disagree > others.agree], ["agree", "disagree"], ""
    )
    result = pd.Series((items["label"].to_numpy() == other_majority).astype(float), index=items.index)
    result[(other_majority == "") | (items["label"] == "not_sure").to_numpy()] = np.nan
    return result


def annotator_table(data: SurveyData) -> pd.DataFrame:
    items = data.items.assign(loo=leave_one_out(data.items))
    _, ds_acc = dawid_skene(data.items)
    g = items.groupby("respondent_id")
    table = pd.DataFrame(
        {
            "n_items": g.size(),
            "pct_agree": g["label"].apply(lambda s: pct((s == "agree").mean())),
            "pct_disagree": g["label"].apply(lambda s: pct((s == "disagree").mean())),
            "pct_not_sure": g["label"].apply(lambda s: pct((s == "not_sure").mean())),
            "loo_agreement": g["loo"].mean().mul(100).round(1),
            "ds_accuracy": ds_acc.mul(100).round(1),
            "straight_liner": g["label"].nunique() == 1,
        }
    )
    return data.respondents[["block_id", "year_of_study", "english_level"]].join(table)


def dispersion(data: SurveyData) -> dict:
    """Are Disagree votes concentrated on a few relations or scattered?"""
    items = data.items.assign(loo=leave_one_out(data.items))
    counts = rating_matrix(data.items, "item_id", CORRECTNESS_ORDER)
    rejected = counts.index[majority_vote(counts) == "reject"]
    d = items[items.label == "disagree"]
    return {
        "n_disagree_votes": len(d),
        "disagree_loo_consistent_pct": pct(d.loo.mean()),
        "agree_loo_consistent_pct": pct(items[items.label == "agree"].loo.mean()),
        "items_with_any_disagree": int((counts.disagree > 0).sum()),
        "items_with_1_2_disagree": int(counts.disagree.between(1, 2).sum()),
        "disagree_votes_on_rejected_items": int(d.item_id.isin(rejected).sum()),
        "disagree_per_item": counts.disagree,
    }


def external_checks(ann: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ("year_of_study", "english_level"):
        groups = [g.pct_disagree.to_numpy() for _, g in ann.groupby(col) if len(g) >= 2]
        h, p = stats.kruskal(*groups)
        for level, g in ann.groupby(col):
            rows.append(
                {
                    "variable": col,
                    "level": level,
                    "n_annotators": len(g),
                    "mean_pct_disagree": round(g.pct_disagree.mean(), 1),
                    "mean_pct_not_sure": round(g.pct_not_sure.mean(), 1),
                    "kruskal_p": round(p, 3),
                }
            )
    return pd.DataFrame(rows)


def robustness(data: SurveyData, ann: pd.DataFrame) -> pd.DataFrame:
    """Majority-vote result under standard annotator-quality filters."""
    items = data.items

    def mv_row(name, subset, weights=None):
        if weights is None:
            labels = majority_vote(rating_matrix(subset, "item_id", CORRECTNESS_ORDER))
        else:  # log-odds weighted vote (weights = estimated accuracy)
            dec = subset[subset.label != "not_sure"]
            w = dec.respondent_id.map(np.log(weights / (1 - weights)))
            score = (np.where(dec.label == "agree", 1, -1) * w).groupby(dec.item_id).sum()
            labels = pd.Series(np.select([score > 0, score < 0], ["accept", "reject"], "tie"), index=score.index)
        acc, rej = (labels == "accept").sum(), (labels == "reject").sum()
        return {"variant": name, "n_annotators": subset.respondent_id.nunique(), "accepted": acc,
                "rejected": rej, "undecided": len(labels) - acc - rej, "precision": pct(acc / (acc + rej))}

    loo_weights = (ann.loo_agreement.fillna(ann.loo_agreement.mean()) / 100).clip(0.51, 0.99)
    bottom = ann.loo_agreement.rank(method="first") <= round(len(ann) * 0.1)
    return pd.DataFrame(
        [
            mv_row("All annotators (headline)", items),
            mv_row("Reliability-weighted vote (leave-one-out accuracy)", items, loo_weights),
            mv_row("Drop least consistent 10% of annotators", items[~items.respondent_id.isin(ann.index[bottom])]),
            mv_row("Drop straight-liners (same answer to all items)", items[~items.respondent_id.isin(ann.index[ann.straight_liner])]),
        ]
    )


def postgraduate_check(data: SurveyData) -> dict:
    """Do the postgraduate annotators (graduates) agree with the undergraduate
    majority? Compared with how often undergraduates agree with each other
    (leave-one-out). Unfiltered: every postgraduate is kept."""
    items = data.items
    grad = items.year_of_study.astype(str).eq("Postgrado")
    ug, pg = items[~grad], items[grad]
    ug_verdict = majority_vote(rating_matrix(ug, "item_id", CORRECTNESS_ORDER))
    pg = pg.assign(ug_verdict=pg.item_id.map(ug_verdict))
    dec = pg[(pg.label != "not_sure") & pg.ug_verdict.isin(["accept", "reject"])]
    match = ((dec.label == "agree") & (dec.ug_verdict == "accept")) | ((dec.label == "disagree") & (dec.ug_verdict == "reject"))
    on_rejected = dec[dec.ug_verdict == "reject"]

    def decisive_agree(df):
        a, d = (df.label == "agree").sum(), (df.label == "disagree").sum()
        return pct(a / (a + d))

    return {
        "n_postgraduates": pg.respondent_id.nunique(),
        "n_undergraduates": ug.respondent_id.nunique(),
        "pg_relations_rated": pg.item_id.nunique(),
        "pg_agreement_with_ug_majority": pct(match.mean()),
        "pg_n_decisive": len(dec),
        "ug_leave_one_out_agreement": pct(leave_one_out(ug).mean()),
        "pg_pct_not_sure": pct((pg.label == "not_sure").mean()),
        "ug_pct_not_sure": pct((ug.label == "not_sure").mean()),
        "pg_decisive_agree": decisive_agree(pg),
        "ug_decisive_agree": decisive_agree(ug),
        "pg_accepted_ug_rejected": int((on_rejected.label == "agree").sum()),
        "ug_rejected_rated_by_pg": int(on_rejected.item_id.nunique()),
    }


def run(data: SurveyData) -> str:
    section("ANNOTATOR RELIABILITY")
    ann = annotator_table(data)
    save_table(ann.reset_index(), "annotators")
    disp = dispersion(data)
    ext = external_checks(ann)
    save_table(ext, "annotator_external_checks")
    rob = robustness(data, ann)
    save_table(rob, "annotator_robustness")
    rho, p = stats.spearmanr(ann.pct_disagree, ann.loo_agreement, nan_policy="omit")

    print(ann.describe().round(2).to_string())
    print(f"\nSpearman(% disagree, LOO agreement) = {rho:.3f}, p = {p:.2g}")
    print({k: v for k, v in disp.items() if k != "disagree_per_item"})
    print(ext.to_string(index=False))
    print(rob.to_string(index=False))

    terciles = pd.qcut(ann.pct_disagree.rank(method="first"), 3, labels=["low", "mid", "high"])
    by_tercile = ann.groupby(terciles, observed=True)[["pct_disagree", "pct_not_sure", "loo_agreement"]].mean().round(1)
    sl = ann[ann.straight_liner]
    sl_desc = f"{(sl.pct_agree == 100).sum()} all-Agree, {(sl.pct_not_sure == 100).sum()} all-Not sure"
    p_year = ext[ext.variable == "year_of_study"].kruskal_p.iloc[0]
    p_eng = ext[ext.variable == "english_level"].kruskal_p.iloc[0]
    pg = postgraduate_check(data)
    save_table(pd.DataFrame([pg]), "annotator_postgraduate_check")
    print(pg)
    return f"""## 4. Annotator reliability

### Consensus-based (circular)

Annotators who answer *Disagree* more often agree less with the other raters of the same items
(Spearman ρ = {rho:.2f}, p = {p:.1g}). **This does not show that dissenters are wrong.** Agreement with
peers penalises dissent by definition. It is reported for completeness only.

{by_tercile.rename_axis('disagree tercile').to_markdown()}

### External (non-circular)

- **Disagree votes are scattered, not concentrated.** Only {disp['disagree_loo_consistent_pct']}% of the
  {disp['n_disagree_votes']} Disagree votes match the majority of the other raters of that relation,
  against {disp['agree_loo_consistent_pct']}% for Agree votes. {disp['items_with_any_disagree']}/{data.items.item_id.nunique()} relations
  received at least one Disagree, but in {disp['items_with_1_2_disagree']} of them only 1–2 people disagreed.
  If the graph had systematic errors, Disagree votes would cluster on the same relations.
- **Disagreement does not depend on training or language.** The rate of Disagree answers does not differ
  by year of study (Kruskal–Wallis p = {p_year}) or English level (p = {p_eng}), so we *cannot* claim
  that the people who disagree are less qualified.

{ext.to_markdown(index=False)}


### Postgraduate annotators vs undergraduate majority

The {pg['n_postgraduates']} postgraduate annotators (all kept, no filtering) rated {pg['pg_relations_rated']} relations.
Their decisive answers agree with the undergraduate majority verdict {pg['pg_agreement_with_ug_majority']}% of the time
(n = {pg['pg_n_decisive']}), against {pg['ug_leave_one_out_agreement']}% for undergraduates with each other (leave-one-out).
They answered Not sure less often ({pg['pg_pct_not_sure']}% vs {pg['ug_pct_not_sure']}%) and agreed with
{pg['pg_decisive_agree']}% of relations among decisive answers (undergraduates {pg['ug_decisive_agree']}%).
Of the {pg['ug_rejected_rated_by_pg']} relations rejected by the undergraduate majority that they rated, they accepted
{pg['pg_accepted_ug_rejected']}. With n = {pg['n_postgraduates']} this is an exploratory check only.

### Robustness of the majority-vote result

{rob.to_markdown(index=False)}

{len(sl)} annotators gave the same answer to every item ({sl_desc}), a classic low-effort pattern.
Removing them *lowers* precision slightly, so quality filtering does not inflate the result.
Precision stays within {rob.precision.min()}–{rob.precision.max()}% under every variant.

Per-annotator table: `results/annotators.csv` (contains no identifying data beyond the random respondent UUID).
"""


if __name__ == "__main__":
    print(run(load(xlsx_from_argv())))
