"""
Decision confidence: how certain is each relation's majority-vote verdict?

For every relation, using decisive judgments (Agree vs Disagree; Not sure
abstains):
- margin           Agree votes minus Disagree votes
- sign_test_p      two-sided sign test of Agree vs Disagree (H0: 50/50)
- entropy          how split all votes are, in [0, 1] (1 = maximally split)
- p_accept / p_reject
                   share of bootstrap resamples of the relation's own raters
                   in which it is accepted / rejected

Each relation is then classed as:
- clear_accept     accepted and sign_test_p < 0.05
- clear_reject     rejected and sign_test_p < 0.05
- contested        everything else (no statistically clear majority)

The same criterion is applied to acceptances and rejections, so the
comparison is symmetric. Precision is reported with Wilson 95% CIs.

Usage:
    python decision_confidence.py [path/to/export.xlsx]
"""

import numpy as np
import pandas as pd
from scipy import stats

from aggregation import majority_vote
from common import (
    CORRECTNESS_ORDER,
    ITEMS_CSV,
    RNG_SEED,
    TYPE_LABELS,
    TYPE_ORDER,
    SurveyData,
    load,
    normalized_entropy,
    rating_matrix,
    save_table,
    section,
    xlsx_from_argv,
)

ALPHA = 0.05
N_RESAMPLES = 5000


def wilson(k: int, n: int) -> tuple[float, float]:
    ci = stats.binomtest(k, n).proportion_ci(method="wilson")
    return round(ci.low * 100, 1), round(ci.high * 100, 1)


def confidence_table(items: pd.DataFrame) -> pd.DataFrame:
    counts = rating_matrix(items, "item_id", CORRECTNESS_ORDER)
    verdict = majority_vote(counts)
    entropy = normalized_entropy(counts)
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for item_id, c in counts.iterrows():
        n = int(c.sum())
        draws = rng.multinomial(n, c.to_numpy() / n, size=N_RESAMPLES)
        agree, disagree = draws[:, 0], draws[:, 2]
        decisive = int(c.agree + c.disagree)
        p = stats.binomtest(int(max(c.agree, c.disagree)), decisive, 0.5).pvalue if decisive else 1.0
        rows.append(
            {
                "item_id": item_id,
                "n_raters": n,
                "n_agree": int(c.agree),
                "n_not_sure": int(c.not_sure),
                "n_disagree": int(c.disagree),
                "verdict": verdict[item_id],
                "margin": int(c.agree - c.disagree),
                "sign_test_p": round(p, 4),
                "entropy": round(entropy[item_id], 2),
                "p_accept": round((agree > disagree).mean() * 100, 1),
                "p_reject": round((disagree > agree).mean() * 100, 1),
            }
        )
    table = pd.DataFrame(rows)
    clear = table.sign_test_p < ALPHA
    table["confidence"] = np.select(
        [clear & (table.verdict == "accept"), clear & (table.verdict == "reject")],
        ["clear_accept", "clear_reject"],
        "contested",
    )
    meta = items.groupby("item_id")[["type", "food", "target"]].first()
    sources = pd.read_csv(ITEMS_CSV).set_index("item_id")["citation"].rename("source")
    return meta.join(sources).join(table.set_index("item_id")).reset_index()


def summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in [*TYPE_ORDER, "all"]:
        g = table if t == "all" else table[table.type == t]
        acc, rej = (g.verdict == "accept").sum(), (g.verdict == "reject").sum()
        c_acc, c_rej = (g.confidence == "clear_accept").sum(), (g.confidence == "clear_reject").sum()
        lo, hi = wilson(acc, acc + rej)
        c_lo, c_hi = wilson(c_acc, c_acc + c_rej)
        rows.append(
            {
                "type": t,
                "n": len(g),
                "accepted": acc,
                "rejected": rej,
                "tie": len(g) - acc - rej,
                "precision": round(100 * acc / (acc + rej), 1),
                "precision_ci": f"{lo}–{hi}",
                "clear_accepted": c_acc,
                "clear_rejected": c_rej,
                "contested": (g.confidence == "contested").sum(),
                "clear_precision": round(100 * c_acc / (c_acc + c_rej), 1),
                "clear_precision_ci": f"{c_lo}–{c_hi}",
            }
        )
    return pd.DataFrame(rows)


def run(data: SurveyData) -> str:
    section("DECISION CONFIDENCE")
    table = confidence_table(data.items)
    save_table(table, "decision_confidence")
    summ = summary(table)
    save_table(summ, "decision_confidence_summary")
    print(summ.to_string(index=False))

    acc = table[table.verdict == "accept"]
    rej = table[table.verdict == "reject"]
    _, p_entropy = stats.mannwhitneyu(acc.entropy, rej.entropy)
    a = summ.set_index("type").loc["all"]
    doubtful = table[table.verdict != "accept"].sort_values("p_reject", ascending=False)
    doubt_cols = ["item_id", "type", "food", "target", "n_agree", "n_not_sure", "n_disagree",
                  "margin", "sign_test_p", "entropy", "p_reject", "confidence"]
    return f"""## 7. Decision confidence

A relation's verdict is **clear** when a two-sided sign test on its Agree vs Disagree votes gives p < {ALPHA};
otherwise it is **contested**. The same test is applied to acceptances and rejections.

{summ.replace({'type': {**TYPE_LABELS, 'all': 'All'}}).to_markdown(index=False)}

- Precision {a.precision}% (95% CI {a.precision_ci}); on clear decisions {a.clear_precision}% (95% CI {a.clear_precision_ci}).
- {a.clear_accepted} relations were accepted with a clear majority; **{a.clear_rejected} {'was' if a.clear_rejected == 1 else 'were'} rejected with a clear majority**.
- Median margin: accepted {acc.margin.median():.0f} votes, rejected {(-rej.margin).median():.0f} votes.
- Mean vote entropy: accepted {acc.entropy.mean():.2f}, rejected {rej.entropy.mean():.2f} (Mann–Whitney p = {p_entropy:.3f}).

### Doubt values for every relation that was not accepted

`p_reject` = share of bootstrap resamples of the relation's raters in which it is rejected.

{doubtful[doubt_cols].to_markdown(index=False)}


Sources of these relations: `results/decision_confidence.csv` (column `source`).
"""


if __name__ == "__main__":
    print(run(load(xlsx_from_argv())))
