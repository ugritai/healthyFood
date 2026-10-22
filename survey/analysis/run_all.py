"""
Run every survey analysis and write a single report.

Outputs (all git-ignored — they contain real respondent data):
    REPORT.md     all results in one file
    results/      one CSV per table

Usage:
    python run_all.py [path/to/export.xlsx]

Each module can also be run on its own, e.g. `python aggregation.py`.
"""

from datetime import date

import agreement
import aggregation
import consensus
import annotator_reliability
import decision_confidence
import descriptive
import disagreement
import error_analysis
import matching
import summary
from common import REPORT_PATH, load, xlsx_from_argv

MODULES = [summary, descriptive, agreement, aggregation, annotator_reliability, error_analysis, matching, decision_confidence, disagreement, consensus]


def main():
    data = load(xlsx_from_argv())
    print(f"Reading responses from {data.source}")
    sections = [module.run(data) for module in MODULES]

    header = f"""# FoodMedKG human evaluation — analysis report

Generated {date.today().isoformat()} by `run_all.py` from `{data.source.name}`.
Each section is produced by the module of the same topic in `survey/analysis/`:

| Section | Module |
|---|---|
| 0. Summary | `summary.py` |
| 1. Data and annotators | `descriptive.py` |
| 2. Inter-annotator agreement | `agreement.py` |
| 3. Item-level aggregation | `aggregation.py` |
| 4. Annotator reliability | `annotator_reliability.py` |
| 5. Error analysis | `error_analysis.py` |
| 6. Ingredient matching | `matching.py` |
| 7. Decision confidence | `decision_confidence.py` |
| 8. Learning from disagreement | `disagreement.py` |
| 9. Consensus grades for the graph | `consensus.py` |
"""
    REPORT_PATH.write_text("\n\n".join([header, *sections]), encoding="utf-8")
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
