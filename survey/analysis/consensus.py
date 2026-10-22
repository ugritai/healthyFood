"""
Consensus grades: one human-consensus score and grade per evaluated graph
element, ready to be written to Neo4j (data/kg_edits/add_consensus.py).

- consensus_score   share of Agree among decisive answers (soft label)
- consensus_grade   strong     accepted with a clear majority (sign test p < 0.05)
                    moderate   accepted, majority not significant
                    contested  tie, or rejected without a clear majority
                    rejected   rejected with a clear majority

Graph elements differ by relation family:
- food–disease   one (:Food)-[:Affects]->(:Disease) edge per relation
- aging          one <component>_level property on an
                 (:`Envejecimiento Saludable`) node per relation
- cooking        generic (:Cooking_Methods) nodes, one per method; the
                 survey illustrated each method with example foods, so the
                 votes of all items sharing a method are pooled

Only evaluated elements get a grade; everything else stays ungraded.

Usage:
    python consensus.py [path/to/export.xlsx]
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import CORRECTNESS_ORDER, ITEMS_CSV, SurveyData, load, rating_matrix, save_table, section, xlsx_from_argv

ALPHA = 0.05


def grade(agree: int, disagree: int) -> tuple[float, str, float]:
    decisive = agree + disagree
    if decisive == 0:
        return np.nan, "contested", 1.0
    p = stats.binomtest(int(max(agree, disagree)), int(decisive), 0.5).pvalue
    score = agree / decisive
    if agree > disagree:
        g = "strong" if p < ALPHA else "moderate"
    elif disagree > agree:
        g = "rejected" if p < ALPHA else "contested"
    else:
        g = "contested"
    return round(score, 3), g, round(p, 4)


def grades(items: pd.DataFrame) -> pd.DataFrame:
    counts = rating_matrix(items, "item_id", CORRECTNESS_ORDER)
    meta = items.groupby("item_id")[["type", "food", "target"]].first()
    per_item = meta.join(counts)

    # aging items state the level they evaluate ("Coffee scores 1/4 on ..."),
    # which identifies the graph node(s) when a food has several variants
    texts = pd.read_csv(ITEMS_CSV).set_index("item_id")["relation_text"]
    levels = texts.str.extract(r"scores (\d)/4", expand=False)

    rows = []
    for item_id, r in per_item[per_item.type != "cooking_method"].iterrows():
        score, g, p = grade(r.agree, r.disagree)
        rows.append({"element": "Affects edge" if r.type == "disease" else "aging level",
                     "items": item_id, "type": r.type, "food": r.food, "target": r.target,
                     "n_agree": r.agree, "n_not_sure": r.not_sure, "n_disagree": r.disagree,
                     "consensus_score": score, "consensus_grade": g, "sign_test_p": p,
                     "aging_level": levels.get(item_id) if r.type == "aging" else None})
    cooking = per_item[per_item.type == "cooking_method"].groupby("target")
    for method, g_ in cooking:
        a, n, d = g_.agree.sum(), g_.not_sure.sum(), g_.disagree.sum()
        score, g, p = grade(a, d)
        rows.append({"element": "Cooking_Methods node", "items": ",".join(g_.index), "type": "cooking_method",
                     "food": "; ".join(g_.food), "target": method,
                     "n_agree": a, "n_not_sure": n, "n_disagree": d,
                     "consensus_score": score, "consensus_grade": g, "sign_test_p": p})
    return pd.DataFrame(rows)


def run(data: SurveyData) -> str:
    section("CONSENSUS GRADES")
    table = grades(data.items)
    save_table(table, "consensus_grades")
    print(table.to_string(index=False))
    dist = table.groupby(["element", "consensus_grade"]).size().unstack(fill_value=0)
    dist = dist.reindex(columns=[c for c in ["strong", "moderate", "contested", "rejected"] if c in dist.columns])
    return f"""## 9. Consensus grades for the graph

One consensus score (share of Agree among decisive answers) and grade per evaluated graph element.
Cooking methods are generic nodes, so the votes of every item that illustrates a method are pooled.

{dist.to_markdown()}

Full table (ready for `data/kg_edits/add_consensus.py`): `results/consensus_grades.csv`.
"""


if __name__ == "__main__":
    print(run(load(xlsx_from_argv())))
