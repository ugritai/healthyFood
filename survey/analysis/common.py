"""
FoodMedKG survey analysis — shared loading, statistics and plotting helpers.

Every analysis module imports from here, so the data is loaded and
cleaned the same way everywhere.

Cleaning: some respondents submitted their block more than once (page
reloads — the repeated rows carry identical answers). Only the first
submission of each (respondent, item) pair is kept; otherwise those
respondents would count two or more times towards every statistic.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).parent
SURVEY_DIR = ANALYSIS_DIR.parent
REPO_DIR = SURVEY_DIR.parent

RAW_DIR = REPO_DIR / "data" / "survey_results"
ITEMS_CSV = SURVEY_DIR / "items.csv"
PART_B_CSV = REPO_DIR / "evaluation" / "results" / "part_b_matching.csv"

TABLES_DIR = ANALYSIS_DIR / "results"
REPORT_PATH = ANALYSIS_DIR / "REPORT.md"

RESPONSES_SHEET = "responses"
MATCH_SHEET = "ingredient_match_responses"

# Survey answers are in Spanish; analysis uses short English codes.
CORRECTNESS_CODES = {"De acuerdo": "agree", "En desacuerdo": "disagree", "No estoy seguro/a": "not_sure"}
CORRECTNESS_ORDER = ["agree", "not_sure", "disagree"]
JUDGMENT_CODES = {"Perfecto": "perfect", "Aceptable": "acceptable", "Incorrecto": "incorrect"}
JUDGMENT_ORDER = ["perfect", "acceptable", "incorrect"]

TYPE_LABELS = {"disease": "Food–disease", "aging": "Food–aging", "cooking_method": "Cooking method"}
TYPE_ORDER = ["disease", "aging", "cooking_method"]

DEMOGRAPHIC_COLS = ["block_id", "year_of_study", "specialization", "gender", "english_level", "age"]

# Survey items excluded from the analysis because the statement shown to
# annotators did not match the graph edge it was meant to evaluate (found
# by checking every food–disease item against its edge in Neo4j; the other
# 39 differ at most cosmetically, e.g. "oats" vs "oat", "turmeric" vs
# "curcuma longa"). Judgments on these items say nothing about the graph.
# Known wording mismatch, currently NOT excluded (all items are analysed):
#   I068 showed "cow's milk"; the graph edge is "cow's milk protein"
#   -[Affects, not suitable]-> cardiovascular diseases.
# To exclude an item, add it here as {"item_id": "reason"}.
EXCLUDED_ITEMS: dict[str, str] = {}

RNG_SEED = 42
N_BOOTSTRAP = 2000

pd.set_option("display.width", 160)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
@dataclass
class SurveyData:
    items: pd.DataFrame  # one row per (respondent, item) judgment, deduplicated
    matches: pd.DataFrame  # one row per (respondent, match) judgment, deduplicated
    source: Path
    n_duplicate_item_rows: int = 0
    n_duplicate_match_rows: int = 0
    excluded_items: dict = field(default_factory=dict)  # item_id -> reason
    respondents: pd.DataFrame = field(default_factory=pd.DataFrame)


def find_default_xlsx() -> Path:
    candidates = sorted(RAW_DIR.glob("*.xlsx"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .xlsx file found in {RAW_DIR}. Pass a path explicitly.")
    return candidates[0]


def xlsx_from_argv() -> Path:
    return Path(sys.argv[1]) if len(sys.argv) > 1 else find_default_xlsx()


def _read_sheet(xlsx_path: Path, sheet: str, id_col: str) -> tuple[pd.DataFrame, int]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    if df.empty:
        raise ValueError(f"The '{sheet}' sheet in {xlsx_path.name} is empty — nothing to analyze.")
    df = df.sort_values("timestamp")
    deduped = df.drop_duplicates(subset=["respondent_id", id_col], keep="first")
    return deduped.reset_index(drop=True), len(df) - len(deduped)


def load(xlsx_path: Path | None = None) -> SurveyData:
    xlsx_path = xlsx_path or find_default_xlsx()
    items, n_dup_items = _read_sheet(xlsx_path, RESPONSES_SHEET, "item_id")
    excluded = {i: r for i, r in EXCLUDED_ITEMS.items() if i in set(items["item_id"])}
    items = items[~items["item_id"].isin(excluded)].reset_index(drop=True)
    items["label"] = items["correctness"].map(CORRECTNESS_CODES)
    items["oversimplified_flag"] = items["oversimplified"] == "Sí"

    matches, n_dup_matches = _read_sheet(xlsx_path, MATCH_SHEET, "match_id")
    matches["label"] = matches["correct_judgment"].map(JUDGMENT_CODES)
    matches["human_correct"] = matches["label"] != "incorrect"

    respondents = (
        items[["respondent_id", *DEMOGRAPHIC_COLS]]
        .drop_duplicates(subset="respondent_id")
        .set_index("respondent_id")
    )
    return SurveyData(items, matches, xlsx_path, n_dup_items, n_dup_matches, excluded, respondents)


# --------------------------------------------------------------------------
# Rating matrices and agreement statistics
# --------------------------------------------------------------------------
def rating_matrix(df: pd.DataFrame, id_col: str, categories: list) -> pd.DataFrame:
    """One row per item, one column per category, cell = number of raters."""
    return df.groupby([id_col, "label"]).size().unstack(fill_value=0).reindex(columns=categories, fill_value=0)


def _observed_agreement(table: np.ndarray) -> np.ndarray:
    n = table.sum(axis=1)
    return (table * (table - 1)).sum(axis=1) / (n * (n - 1))


def fleiss_kappa(table: np.ndarray) -> float:
    """Fleiss' kappa generalised to a variable number of raters per item
    (per-item observed agreement averaged over items; chance agreement
    from pooled category proportions)."""
    table = np.asarray(table, dtype=float)
    p_bar = _observed_agreement(table).mean()
    p_j = table.sum(axis=0) / table.sum()
    p_e = (p_j**2).sum()
    return 1.0 if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def gwet_ac1(table: np.ndarray) -> float:
    """Gwet's AC1 — a chance correction that stays stable when one
    category dominates, where kappa tends to collapse."""
    table = np.asarray(table, dtype=float)
    p_bar = _observed_agreement(table).mean()
    p_j = (table / table.sum(axis=1, keepdims=True)).mean(axis=0)
    p_e = (p_j * (1 - p_j)).sum() / (table.shape[1] - 1)
    return 1.0 if p_e == 1 else (p_bar - p_e) / (1 - p_e)


def bootstrap_ci(table: np.ndarray, stat, n: int = N_BOOTSTRAP) -> tuple[float, float]:
    """95% percentile CI from resampling items with replacement.

    Items rather than respondents are resampled: resampling respondents
    would put the same person in the pool twice, and a person always
    agrees with themselves, which inflates agreement."""
    table = np.asarray(table, dtype=float)
    rng = np.random.default_rng(RNG_SEED)
    draws = [stat(table[rng.integers(0, len(table), len(table))]) for _ in range(n)]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def normalized_entropy(table: pd.DataFrame) -> pd.Series:
    """Entropy of each item's vote distribution in [0, 1]; 1 = maximally split."""
    counts = table.to_numpy(dtype=float)
    p = counts / counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.where(p > 0, p * np.log2(p), 0.0).sum(axis=1)
    return pd.Series(h / np.log2(table.shape[1]), index=table.index, name="difficulty")


def pct(x: float) -> float:
    return round(float(x) * 100, 1)


# --------------------------------------------------------------------------
# Output helpers
# --------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str) -> Path:
    TABLES_DIR.mkdir(exist_ok=True)
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    return path


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
