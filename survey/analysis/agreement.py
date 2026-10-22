"""
Inter-annotator agreement for both questionnaires: Fleiss' kappa
(generalised to a variable number of raters per item, so all items are
used), Gwet's AC1 (stable when one category dominates), raw pairwise
agreement, and 95% bootstrap CIs.

For the main items the statistics are given twice: over all three answers,
and over decisive answers only (Agree vs Disagree, Not sure dropped),
which matches how the majority-vote results treat Not sure.

Usage:
    python agreement.py [path/to/export.xlsx]
"""

import pandas as pd

from common import (
    CORRECTNESS_ORDER,
    JUDGMENT_ORDER,
    N_BOOTSTRAP,
    SurveyData,
    _observed_agreement,
    bootstrap_ci,
    fleiss_kappa,
    gwet_ac1,
    load,
    pct,
    rating_matrix,
    save_table,
    section,
    xlsx_from_argv,
)


def agreement_row(name: str, table: pd.DataFrame) -> dict:
    table = table[table.sum(axis=1) >= 2]  # pairwise agreement needs >= 2 raters
    m = table.to_numpy(dtype=float)
    k_lo, k_hi = bootstrap_ci(m, fleiss_kappa)
    a_lo, a_hi = bootstrap_ci(m, gwet_ac1)
    return {
        "dataset": name,
        "n_items": len(m),
        "pairwise_agreement_pct": pct(_observed_agreement(m).mean()),
        "fleiss_kappa": round(fleiss_kappa(m), 3),
        "kappa_ci_lo": round(k_lo, 3),
        "kappa_ci_hi": round(k_hi, 3),
        "gwet_ac1": round(gwet_ac1(m), 3),
        "ac1_ci_lo": round(a_lo, 3),
        "ac1_ci_hi": round(a_hi, 3),
    }


def compute(data: SurveyData) -> pd.DataFrame:
    items_3 = rating_matrix(data.items, "item_id", CORRECTNESS_ORDER)
    items_2 = items_3[["agree", "disagree"]]
    matches_3 = rating_matrix(data.matches, "match_id", JUDGMENT_ORDER)
    matches_2 = pd.DataFrame(
        {"correct": matches_3["perfect"] + matches_3["acceptable"], "incorrect": matches_3["incorrect"]}
    )
    return pd.DataFrame(
        [
            agreement_row("items: agree / not sure / disagree", items_3),
            agreement_row("items: agree vs disagree (not sure dropped)", items_2),
            agreement_row("matching: perfect / acceptable / incorrect", matches_3),
            agreement_row("matching: correct vs incorrect", matches_2),
        ]
    )


def run(data: SurveyData) -> str:
    section("AGREEMENT")
    table = compute(data)
    save_table(table, "agreement")
    print(table.to_string(index=False))
    main = table.iloc[0]
    return f"""## 2. Inter-annotator agreement

Fleiss' kappa is generalised to a variable number of raters per item, so all items are used.
95% CIs come from {N_BOOTSTRAP} bootstrap resamples of *items*. (Resampling respondents would put the
same person in the sample twice, and a person always agrees with themselves.)

{table.to_markdown(index=False)}

Main items: Fleiss' κ = {main.fleiss_kappa} [{main.kappa_ci_lo}, {main.kappa_ci_hi}],
Gwet's AC1 = {main.gwet_ac1} [{main.ac1_ci_lo}, {main.ac1_ci_hi}]. Agreement is slight to fair
(Landis & Koch). This is expected for nutrition evidence, where the literature itself is often mixed;
see Section 5 for where the disagreement is concentrated.
"""


if __name__ == "__main__":
    print(run(load(xlsx_from_argv())))
