"""
Descriptive statistics: who answered, how many judgments, and the raw
distribution of Agree / Not sure / Disagree per item and per relation
type. No aggregation or modelling — the plain numbers everything else
builds on.

Usage:
    python descriptive.py [path/to/export.xlsx]
"""

import pandas as pd

from common import (
    CORRECTNESS_ORDER,
    TYPE_LABELS,
    TYPE_ORDER,
    SurveyData,
    load,
    pct,
    save_table,
    section,
    xlsx_from_argv,
)


def item_summary(items: pd.DataFrame) -> pd.DataFrame:
    """One row per item: % of each answer, % flagged oversimplified, rater count."""
    shares = pd.crosstab(items["item_id"], items["label"], normalize="index").reindex(
        columns=CORRECTNESS_ORDER, fill_value=0
    )
    meta = items.groupby("item_id")[["type", "food", "target"]].first()
    out = meta.join((shares * 100).round(1).add_prefix("pct_"))
    out["pct_oversimplified"] = (items.groupby("item_id")["oversimplified_flag"].mean() * 100).round(1)
    out["n_raters"] = items.groupby("item_id").size()
    return out.reset_index()


def type_summary(items: pd.DataFrame) -> pd.DataFrame:
    """Judgment-level distribution per relation type, plus decisive precision
    = Agree / (Agree + Disagree), i.e. with Not sure treated as abstention."""
    rows = []
    for t, g in [*((t, items[items["type"] == t]) for t in TYPE_ORDER), ("all", items)]:
        share = g["label"].value_counts(normalize=True)
        a, d = share.get("agree", 0), share.get("disagree", 0)
        rows.append(
            {
                "type": t,
                "n_items": g["item_id"].nunique(),
                "n_judgments": len(g),
                "pct_agree": pct(a),
                "pct_disagree": pct(d),
                "pct_not_sure": pct(share.get("not_sure", 0)),
                "decisive_precision": pct(a / (a + d)),
                "pct_oversimplified": pct(g["oversimplified_flag"].mean()),
            }
        )
    return pd.DataFrame(rows)


def demographics(data: SurveyData) -> dict[str, pd.Series]:
    r = data.respondents
    return {col: r[col].value_counts() for col in ["year_of_study", "specialization", "english_level", "gender", "block_id"]}


def run(data: SurveyData) -> str:
    section("DESCRIPTIVE")
    items = data.items
    print(
        f"{len(items)} judgments on {items['item_id'].nunique()} items from "
        f"{items['respondent_id'].nunique()} respondents "
        f"({data.n_duplicate_item_rows} duplicate rows from repeated submissions removed)."
    )
    per_item = item_summary(items)
    save_table(per_item, "item_summary")
    by_type = type_summary(items)
    save_table(by_type, "type_summary")
    print("\nBy relation type:\n" + by_type.to_string(index=False))

    demo = demographics(data)
    for col, counts in demo.items():
        print(f"\n{col}:\n{counts.to_string()}")
    ages = data.respondents["age"]

    excluded_md = "; ".join(f"{i} ({r})" for i, r in data.excluded_items.items()) or "none"
    n_raters = per_item["n_raters"]
    demo_md = "\n".join(
        f"- **{col.replace('_', ' ')}**: " + ", ".join(f"{k} ({v})" for k, v in counts.items())
        for col, counts in demo.items()
    )
    return f"""## 1. Data and annotators

- **Source:** `{data.source.name}`
- **Cleaning:** {data.n_duplicate_item_rows} duplicate item rows and {data.n_duplicate_match_rows} duplicate
  match rows were removed. They came from respondents who submitted the same block more than
  once (page reloads) with identical answers; only the first submission is kept.
- **Excluded items:** {excluded_md}
- **Judgments:** {len(items)} on {items['item_id'].nunique()} relations from
  {items['respondent_id'].nunique()} respondents ({n_raters.min()}–{n_raters.max()} raters per relation).
- **Age:** mean {ages.mean():.1f}, median {ages.median():.0f}, range {ages.min()}–{ages.max()}.

{demo_md}

### Judgments by relation type

Decisive precision = Agree / (Agree + Disagree), i.e. *Not sure* treated as an abstention.

{by_type.to_markdown(index=False)}


Per-item table: `results/item_summary.csv`.
"""


if __name__ == "__main__":
    print(run(load(xlsx_from_argv())))
