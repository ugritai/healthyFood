#!/usr/bin/env python3
"""
Baselines and weight analysis for the ingredient matcher (needs Neo4j).

Runs the 200 surface-form queries of benchmark.json (`matching_queries`)
against the Food Composition entries of the graph with several matching
methods, using the candidate filter (processed entries excluded) and the
correctness criterion (a key word appears in the matched name) of
evaluate.py, and reports top-1 accuracy, top-3 accuracy and MRR.

    string only       Jaro–Winkler, token-set ratio, TF-IDF of character 3–5-grams
    embeddings only   all-MiniLM-L6-v2 (the system's model), stronger or
                      multilingual sentence-embedding models, and the AI2Cuisine model
                      fine-tuned on food data when its folder is present
                      (EMBEDDING_MODELS, FINETUNED_MODEL)
    hybrid            S = w * cosine + (1 - w) * Jaro–Winkler, the system's score,
                      for w = 0.0, 0.1, ..., 1.0 with each embedding model

The weight w is also chosen by 5-fold cross-validation (w picked on four
folds by top-1 accuracy, evaluated on the fifth), so that the reported
accuracy of a tuned weight is not measured on the queries used to tune it.

Everything is also run on misspelled versions of the queries, to measure
robustness to misspellings. Each misspelling is one deleted, substituted or
swapped (adjacent) letter, never the first letter of a word, in a word of at
least --typo-min-len letters, applied to a fraction --typo-rate of the
queries (e.g. "broccolis" -> "brocolis", "baby spinach" -> "baby sninach").
--typo-seeds versions are built, each with its own random seed, and results
are reported as mean and SD over them. The default is the setting reported in
the paper (25% of the queries, words of six or more letters); the stress tests
use, e.g., --typo-rate 1 --typo-min-len 4 --tag _all4.

Usage
-----
    cd evaluation/linking
    python baselines.py                                         # paper setting
    python baselines.py --typo-rate 1 --typo-min-len 6 --tag _long6
    python baselines.py --typo-rate 1 --typo-min-len 4 --tag _all4
    python baselines.py --models all-MiniLM-L6-v2 BAAI/bge-base-en-v1.5

Output (results/, with the --tag suffix): baselines_results.csv (every method
and query version), baselines_summary.csv (clean vs. misspelled, mean and SD),
weight_sweep.csv and weight_sweep.png (accuracy by w), typo_queries.csv (the
misspelled queries).
"""

import argparse
import os
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LINKING_DIR = Path(__file__).parent
EVAL_DIR = LINKING_DIR.parent
RESULTS_DIR = LINKING_DIR / "results"
sys.path.insert(0, str(EVAL_DIR))
from evaluate import _connect_neo4j, _is_correct, _load_food_df  # noqa: E402

SYSTEM_MODEL = "all-MiniLM-L6-v2"
SYSTEM_WEIGHT = 0.7
EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    "intfloat/multilingual-e5-base",
]
# AI2Cuisine model: BERT sentence-transformer fine-tuned on food data (from bert-base-nli-max-tokens),
# from Morales-Garzón et al. (2025), AI2Cuisine; not versioned (419 MB)
FINETUNED_MODEL = EVAL_DIR / "12-20_22-39_1000016b_4ep_3000wu_2e-07lr"
if FINETUNED_MODEL.exists():
    EMBEDDING_MODELS.append(str(FINETUNED_MODEL))
MODEL_LABELS = {str(FINETUNED_MODEL): "AI2Cuisine"}
E5_PREFIX = "query: "  # E5 models expect a prefix; symmetric matching uses "query: " on both sides
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 1)
TOP_K = 3


def ranking_metrics(scores: np.ndarray, correct: np.ndarray) -> dict:
    """scores: queries x candidates; correct: same shape, True where the candidate is right."""
    order = np.argsort(-scores, axis=1, kind="stable")
    hits = np.take_along_axis(correct, order, axis=1)
    first = np.where(hits.any(axis=1), hits.argmax(axis=1), -1)
    rr = np.where(first >= 0, 1.0 / (first + 1), 0.0)
    return {"top1": 100 * (first == 0).mean(), f"top{TOP_K}": 100 * ((first >= 0) & (first < TOP_K)).mean(),
            "mrr": rr.mean()}


def top1_hits(scores: np.ndarray, correct: np.ndarray) -> np.ndarray:
    return correct[np.arange(len(scores)), scores.argmax(axis=1)]


def cross_validated_weight(sem: np.ndarray, jw: np.ndarray, correct: np.ndarray, seed: int) -> dict:
    """Choose w on four folds by top-1 accuracy and score the fifth; every query is scored once, as a test query."""
    hits = {w: top1_hits(w * sem + (1 - w) * jw, correct) for w in WEIGHTS}
    folds = np.random.default_rng(seed).permutation(len(sem)) % 5
    chosen = []
    test_scores = np.empty_like(sem)
    for k in range(5):
        train, test = folds != k, folds == k
        top = max(hits[w][train].mean() for w in WEIGHTS)
        tied = [w for w in WEIGHTS if hits[w][train].mean() == top]
        best = tied[(len(tied) - 1) // 2]  # neutral tie-break: the middle of the tied weights
        chosen.append(float(best))
        test_scores[test] = best * sem[test] + (1 - best) * jw[test]
    return {"cv_weights": chosen, "cv_scores": test_scores}


def add_typo(text: str, rng: np.random.Generator, min_len: int = 4) -> str:
    """One random typo (deletion, substitution or swap of adjacent letters), never on the first letter,
    in a word of at least min_len letters; the text is returned unchanged if it has no such word."""
    words = text.split(" ")
    idx = [i for i, wd in enumerate(words) if len(wd) >= min_len and wd.isalpha()]
    if not idx:
        return text
    i = int(rng.choice(idx))
    wd = words[i]
    while True:
        op = rng.choice(["delete", "substitute", "swap"])
        pos = int(rng.integers(1, len(wd) - (1 if op == "swap" else 0)))
        if op == "delete":
            new = wd[:pos] + wd[pos + 1:]
        elif op == "substitute":
            new = wd[:pos] + rng.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c != wd[pos]]) + wd[pos + 1:]
        else:
            new = wd[:pos] + wd[pos + 1] + wd[pos] + wd[pos + 2:]
        if new != wd:
            words[i] = new
            return " ".join(words)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=EMBEDDING_MODELS)
    p.add_argument("--neo4j-uri", default="bolt://localhost:17687")
    p.add_argument("--neo4j-user", default="neo4j")
    p.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "neo4j"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--typo-seeds", type=int, default=5, help="number of misspelled versions of the queries")
    p.add_argument("--typo-rate", type=float, default=0.25, help="fraction of queries given a misspelling")
    p.add_argument("--typo-min-len", type=int, default=6, help="only words with at least this many letters are misspelled")
    p.add_argument("--tag", default="", help="suffix for the output files, to keep several settings")
    args = p.parse_args()

    from rapidfuzz import fuzz, process
    from rapidfuzz.distance import JaroWinkler
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer

    queries = json.loads((EVAL_DIR / "benchmark.json").read_text(encoding="utf-8"))["matching_queries"]
    texts = [q["query"] for q in queries]
    query_sets = {"clean": texts}
    for k in range(args.typo_seeds):
        rng = np.random.default_rng(args.seed + k)
        query_sets[f"typo-{k}"] = [add_typo(t, rng, args.typo_min_len) if rng.random() < args.typo_rate else t
                                   for t in texts]
    n_changed = np.mean([sum(a != b for a, b in zip(texts, qs)) for name, qs in query_sets.items() if name != "clean"])
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame({"query": texts, **{name: qs for name, qs in query_sets.items() if name != "clean"}}).to_csv(
        RESULTS_DIR / f"typo_queries{args.tag}.csv", index=False)

    driver = _connect_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)
    df = _load_food_df(driver)
    driver.close()
    cands = df[~df["is_processed"]].reset_index(drop=True)
    names = cands["food_name"].tolist()
    print(f"{len(texts)} queries x {len(names):,} candidate entries; {args.typo_seeds} misspelled versions "
          f"(rate {args.typo_rate}, words >= {args.typo_min_len} letters): {n_changed:.0f} queries changed on average")
    correct = np.array([[_is_correct(n, q["key_words"]) for n in names] for q in queries])

    lower = [n.lower() for n in names]
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True).fit(lower)
    tf_cands = tfidf.transform(lower)
    changed = {name: np.array([a != b for a, b in zip(texts, qs)]) for name, qs in query_sets.items()}

    def metrics(scores: np.ndarray, qs_name: str) -> dict:
        m = ranking_metrics(scores, correct)
        mask = changed[qs_name]
        m["top1_changed"] = 100 * top1_hits(scores[mask], correct[mask]).mean() if mask.any() else np.nan
        return m

    jw, rows = {}, []
    for qs_name, qs in query_sets.items():
        ql = [t.lower() for t in qs]
        jw[qs_name] = process.cdist(ql, lower, scorer=JaroWinkler.normalized_similarity, workers=-1)
        tsr = process.cdist(ql, lower, scorer=fuzz.token_set_ratio, workers=-1) / 100
        tf = (tfidf.transform(ql) @ tf_cands.T).toarray()
        for method, scores in (("Jaro–Winkler", jw[qs_name]), ("Token-set ratio", tsr), ("TF-IDF char 3–5-grams", tf)):
            rows.append({"queries": qs_name, "family": "string", "method": method, "model": "", "w": "",
                         **metrics(scores, qs_name)})

    sweep = []
    for model_name in args.models:
        label = MODEL_LABELS.get(model_name, model_name)
        print(f"  encoding with {label} ...", flush=True)
        model = SentenceTransformer(model_name)
        prefix = E5_PREFIX if model_name.startswith("intfloat/") else ""
        c_emb = model.encode([prefix + n for n in names], normalize_embeddings=True, convert_to_numpy=True,
                             batch_size=128, show_progress_bar=False)
        for qs_name, qs in query_sets.items():
            sem = model.encode([prefix + t for t in qs], normalize_embeddings=True, convert_to_numpy=True) @ c_emb.T
            j = jw[qs_name]
            for w in WEIGHTS:
                sweep.append({"queries": qs_name, "model": label, "w": float(w),
                              **metrics(w * sem + (1 - w) * j, qs_name)})
            rows.append({"queries": qs_name, "family": "embeddings", "method": "Cosine only", "model": label,
                         "w": 1.0, **metrics(sem, qs_name)})
            rows.append({"queries": qs_name, "family": "hybrid", "method": f"w = {SYSTEM_WEIGHT} (system weight)",
                         "model": label, "w": SYSTEM_WEIGHT,
                         **metrics(SYSTEM_WEIGHT * sem + (1 - SYSTEM_WEIGHT) * j, qs_name)})
            cv = cross_validated_weight(sem, j, correct, args.seed)
            rows.append({"queries": qs_name, "family": "hybrid", "method": "w by 5-fold CV", "model": label,
                         "w": "/".join(f"{x:.1f}" for x in cv["cv_weights"]), **metrics(cv["cv_scores"], qs_name)})

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / f"baselines_results{args.tag}.csv", index=False)
    sw = pd.DataFrame(sweep)
    sw.to_csv(RESULTS_DIR / f"weight_sweep{args.tag}.csv", index=False)

    # Summary: clean queries vs. misspelled versions (mean and SD over the versions)
    res["set"] = np.where(res["queries"] == "clean", "clean", "typo")
    keys = ["family", "method", "model"]
    clean = res[res["set"] == "clean"].set_index(keys)["top1"].rename("clean_top1")
    typo = res[res["set"] == "typo"].groupby(keys, sort=False)["top1"].agg(["mean", "std"])
    typo_changed = res[res["set"] == "typo"].groupby(keys, sort=False)["top1_changed"].mean().rename("typo_changed_top1")
    summary = pd.concat([clean, typo.rename(columns={"mean": "typo_top1", "std": "typo_sd"}), typo_changed],
                        axis=1).reset_index()
    summary["drop"] = summary["clean_top1"] - summary["typo_top1"]
    summary.to_csv(RESULTS_DIR / f"baselines_summary{args.tag}.csv", index=False)
    sw["set"] = np.where(sw["queries"] == "clean", "clean", "typo")
    curve = sw.groupby(["set", "model", "w"], sort=False)["top1"].mean().reset_index()
    with pd.option_context("display.width", 200, "display.max_colwidth", 40):
        print(summary.round(1).to_string(index=False))
        for st in ("clean", "typo"):
            print(f"\nTop-1 accuracy by weight w, {st} queries" + (" (mean over versions)" if st == "typo" else "") + ":")
            print(curve[curve["set"] == st].pivot(index="w", columns="model", values="top1").round(1).to_string())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    titles = ("Correctly spelled queries",
              f"With misspellings ({args.typo_rate:.0%} of queries; mean of {args.typo_seeds} versions)")
    for ax, st, title in zip(axes, ("clean", "typo"), titles):
        for model_name, g in curve[curve["set"] == st].groupby("model", sort=False):
            ax.plot(g["w"], g["top1"], marker="o", label=model_name,
                    linewidth=2.5 if model_name == SYSTEM_MODEL else 1.2)
        ax.axvline(SYSTEM_WEIGHT, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("w (weight of the semantic score; 1 − w for Jaro–Winkler)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"weight_sweep{args.tag}.png", dpi=200)
    print(f"\nWritten to {RESULTS_DIR}: baselines_results{args.tag}.csv, baselines_summary{args.tag}.csv, "
          f"weight_sweep{args.tag}.csv, weight_sweep{args.tag}.png, typo_queries{args.tag}.csv")


if __name__ == "__main__":
    main()
