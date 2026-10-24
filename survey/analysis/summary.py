"""
Headline summary: the majority-vote accuracy and the same figure under
every alternative metric, each with a Wilson 95% CI over decided
relations, plus the response distribution of both survey tasks.

Usage:
    python summary.py [path/to/export.xlsx]
"""

import numpy as np
import pandas as pd

from aggregation import all_methods
from annotator_reliability import annotator_table, robustness
from common import (
    CORRECTNESS_ORDER,
    JUDGMENT_ORDER,
    TYPE_LABELS,
    TYPE_ORDER,
    SurveyData,
    load,
    pct,
    save_table,
    section,
    xlsx_from_argv,
)
from decision_confidence import confidence_table, wilson
from disagreement import compare


def metrics(data: SurveyData) -> pd.DataFrame:
    labels = all_methods(data.items)
    conf = confidence_table(data.items)
    rob = robustness(data, annotator_table(data)).set_index("variant")

    def row(name, group, accepted, rejected):
        lo, hi = wilson(int(accepted), int(accepted + rejected))
        return {"metric": name, "group": group, "accepted": int(accepted), "rejected": int(rejected),
                "accuracy": pct(accepted / (accepted + rejected)), "ci_lo": lo, "ci_hi": hi}

    mv = labels.majority_vote
    rows = [
        row("Majority vote", "headline", (mv == "accept").sum(), (mv == "reject").sum()),
        row("Clear verdicts only (sign test p < 0.05)", "certainty",
            (conf.confidence == "clear_accept").sum(), (conf.confidence == "clear_reject").sum()),
    ]
    for variant, name in [
        ("Reliability-weighted vote (leave-one-out accuracy)", "Reliability-weighted vote"),
        ("Drop least consistent 10% of annotators", "Without least consistent 10% of annotators"),
        ("Drop straight-liners (same answer to all items)", "Without straight-liners"),
    ]:
        r = rob.loc[variant]
        rows.append(row(name, "annotators", r.accepted, r.rejected))
    for method, name in [("plurality", "Plurality vote"), ("dawid_skene", "Dawid–Skene")]:
        rows.append(row(name, "aggregation", (labels[method] == "accept").sum(), (labels[method] == "reject").sum()))
    _, soft = compare(data.items)
    for method, name in [("crowdtruth", "CrowdTruth 2.0"), ("mace", "MACE")]:
        rows.append(row(name, "disagreement", (soft[method] == "accept").sum(), (soft[method] == "reject").sum()))
    return pd.DataFrame(rows)


PLANTED_ERRORS = ["M02", "M03", "M04"]  # known-incorrect matches shown in every block


def response_shares(data: SurveyData) -> tuple[pd.DataFrame, pd.DataFrame]:
    """% of each answer: KG relations by type, matches by system-correct vs planted error."""
    items = data.items
    rel = {}
    for t in TYPE_ORDER:
        g = items[items.type == t]
        rel[f"{TYPE_LABELS[t]}  (n={len(g)})"] = g.label.value_counts(normalize=True)
    rel[f"All relations  (n={len(items)})"] = items.label.value_counts(normalize=True)
    rel = pd.DataFrame(rel).T.reindex(columns=CORRECTNESS_ORDER, fill_value=0) * 100

    m = data.matches
    planted = m.match_id.isin(PLANTED_ERRORS)
    match = pd.DataFrame(
        {
            f"System matches  ({m[~planted].match_id.nunique()} matches, n={(~planted).sum()})":
                m[~planted].label.value_counts(normalize=True),
            f"Planted errors  ({m[planted].match_id.nunique()} matches, n={planted.sum()})":
                m[planted].label.value_counts(normalize=True),
        }
    ).T.reindex(columns=JUDGMENT_ORDER, fill_value=0) * 100
    return rel, match


def run(data: SurveyData) -> str:
    section("SUMMARY")
    table = metrics(data)
    save_table(table, "summary_metrics")
    print(table.to_string(index=False))
    rel, match = response_shares(data)
    save_table(rel.round(1).reset_index(names="group"), "summary_responses_relations")
    save_table(match.round(1).reset_index(names="group"), "summary_responses_matching")
    h = table.iloc[0]
    shown = table.assign(**{"95% CI": table.ci_lo.astype(str) + "–" + table.ci_hi.astype(str)})[
        ["metric", "accepted", "rejected", "accuracy", "95% CI"]
    ]
    return f"""## 0. Summary

**Majority-vote accuracy: {h.accuracy}% (95% CI {h.ci_lo}–{h.ci_hi}%)** - {h.accepted} of {h.accepted + h.rejected}
decided relations accepted. Every alternative metric is shown below with the same accuracy definition
(accepted / decided) and a Wilson 95% CI.

{shown.to_markdown(index=False)}


### The two kinds of responses

{rel.round(1).to_markdown()}

{match.round(1).to_markdown()}

"""


if __name__ == "__main__":
    print(run(load(xlsx_from_argv())))
