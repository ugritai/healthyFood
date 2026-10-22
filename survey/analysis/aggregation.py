"""
Item-level aggregation: turning 15–20 judgments per relation into one
accepted / rejected decision, as is standard for crowdsourced labels.

Methods compared (the first is the headline result):
- majority_vote      Agree vs Disagree, Not sure treated as abstention
- plurality          most frequent of the three answers (Not sure can win)
- strict_majority    accepted/rejected only if > 50% of *all* raters agree
- dawid_skene        EM estimate of each annotator's reliability, decisive
                     judgments only (Dawid & Skene, 1979)

Precision = accepted / (accepted + rejected) over decided items.

Usage:
    python aggregation.py [path/to/export.xlsx]
"""

import numpy as np
import pandas as pd

from common import (
    CORRECTNESS_ORDER,
    TYPE_LABELS,
    TYPE_ORDER,
    SurveyData,
    load,
    pct,
    rating_matrix,
    save_table,
    section,
    xlsx_from_argv,
)

METHOD_LABELS = {
    "majority_vote": "Majority vote (Not sure = abstain)",
    "plurality": "Plurality (Not sure can win)",
    "strict_majority": "Strict majority (>50% of all raters)",
    "dawid_skene": "Dawid–Skene (decisive judgments)",
}


def majority_vote(counts: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select([counts.agree > counts.disagree, counts.disagree > counts.agree], ["accept", "reject"], "tie"),
        index=counts.index,
    )


def plurality(counts: pd.DataFrame) -> pd.Series:
    def decide(row):
        winners = row[row == row.max()].index
        if len(winners) > 1:
            return "tie"
        return {"agree": "accept", "disagree": "reject", "not_sure": "not_sure"}[winners[0]]

    return counts.apply(decide, axis=1)


def strict_majority(counts: pd.DataFrame) -> pd.Series:
    share = counts.div(counts.sum(axis=1), axis=0)
    return pd.Series(
        np.select([share.agree > 0.5, share.disagree > 0.5], ["accept", "reject"], "undecided"),
        index=counts.index,
    )


def dawid_skene(items: pd.DataFrame, n_iter: int = 100) -> tuple[pd.Series, pd.Series]:
    """Binary Dawid–Skene on decisive judgments. Returns the per-item label
    and each annotator's estimated accuracy (mean of their confusion-matrix
    diagonal)."""
    dec = items[items["label"] != "not_sure"]
    item_ids = sorted(dec["item_id"].unique())
    annotators = sorted(dec["respondent_id"].unique())
    ii = dec["item_id"].map({x: i for i, x in enumerate(item_ids)}).to_numpy()
    aa = dec["respondent_id"].map({x: i for i, x in enumerate(annotators)}).to_numpy()
    kk = (dec["label"] == "disagree").astype(int).to_numpy()  # 0 = agree, 1 = disagree

    # initialise with soft majority vote
    t = np.full((len(item_ids), 2), 0.5)
    for c in (0, 1):
        np.add.at(t[:, c], ii[kk == c], 1)
    t /= t.sum(axis=1, keepdims=True)

    for _ in range(n_iter):
        conf = np.full((len(annotators), 2, 2), 0.5)  # Laplace smoothing
        for c in (0, 1):
            np.add.at(conf[:, :, c], aa[kk == c], t[ii[kk == c]])
        conf /= conf.sum(axis=2, keepdims=True)
        log_t = np.tile(np.log(t.mean(axis=0)), (len(item_ids), 1))
        np.add.at(log_t, ii, np.log(conf[aa, :, kk]))
        t = np.exp(log_t - log_t.max(axis=1, keepdims=True))
        t /= t.sum(axis=1, keepdims=True)

    labels = pd.Series(np.where(t[:, 0] > 0.5, "accept", "reject"), index=item_ids)
    accuracy = pd.Series(conf[:, [0, 1], [0, 1]].mean(axis=1), index=annotators, name="ds_accuracy")
    return labels, accuracy


def all_methods(items: pd.DataFrame) -> pd.DataFrame:
    counts = rating_matrix(items, "item_id", CORRECTNESS_ORDER)
    ds_labels, _ = dawid_skene(items)
    labels = pd.DataFrame(
        {
            "majority_vote": majority_vote(counts),
            "plurality": plurality(counts),
            "strict_majority": strict_majority(counts),
            "dawid_skene": ds_labels,
        }
    )
    meta = items.groupby("item_id")[["type", "food", "target"]].first()
    return meta.join(counts.add_prefix("n_")).join(labels)


def summarize(labels: pd.DataFrame, methods=METHOD_LABELS) -> pd.DataFrame:
    rows = []
    for method in methods:
        for t in [*TYPE_ORDER, "all"]:
            col = labels[method] if t == "all" else labels.loc[labels["type"] == t, method]
            acc, rej = (col == "accept").sum(), (col == "reject").sum()
            rows.append(
                {
                    "method": method,
                    "type": t,
                    "accepted": acc,
                    "rejected": rej,
                    "undecided": len(col) - acc - rej,
                    "precision": pct(acc / (acc + rej)) if acc + rej else np.nan,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def run(data: SurveyData) -> str:
    section("AGGREGATION")
    labels = all_methods(data.items)
    save_table(labels.reset_index(), "item_labels")
    summary = summarize(labels)
    save_table(summary, "aggregation_summary")
    print(summary.to_string(index=False))


    mv = summary[summary.method == "majority_vote"].set_index("type")
    ds = summary[(summary.method == "dawid_skene") & (summary.type == "all")].iloc[0]
    wide = summary.pivot(index="method", columns="type", values="precision")[[*TYPE_ORDER, "all"]]
    counts = summary[summary.type == "all"].set_index("method")[["accepted", "rejected", "undecided"]]
    table = counts.join(wide).loc[list(METHOD_LABELS)].rename(index=METHOD_LABELS, columns=TYPE_LABELS)
    table = table.rename(columns={"all": "precision (all)"})
    rejected = labels[labels.majority_vote == "reject"][["type", "food", "target", "n_agree", "n_not_sure", "n_disagree"]]
    a = mv.loc["all"]
    return f"""## 3. Item-level aggregation

The judgments for each relation are aggregated into a single accept/reject decision.
**Headline method: majority vote with *Not sure* treated as an abstention.**
"Not sure" means the annotator could not judge the relation, not that it is wrong.

Majority vote: **{a.accepted} accepted, {a.rejected} rejected, {a.undecided} ties → precision {a.precision}%**
(food–disease {mv.loc['disease'].precision}%, food–aging {mv.loc['aging'].precision}%,
cooking method {mv.loc['cooking_method'].precision}%).

Other methods, reported for robustness. The strict majority covers only the relations it can decide, so
its precision is not comparable on its own. Dawid–Skene, which weights annotators by estimated
reliability, is the most conservative ({ds.precision}%).

{table.to_markdown()}


### Relations rejected by majority vote

{rejected.to_markdown()}


Per-item labels under every method: `results/item_labels.csv`.
"""


if __name__ == "__main__":
    print(run(load(xlsx_from_argv())))
