#!/usr/bin/env python3
"""
FoodMedKG Quantitative Evaluation
==================================
Three evaluation components:

  (A) LLM Structured Output Quality  - run for each model listed in --models
      - JSON validity rate
      - Required field completeness rate
      - Preparation method validity rate
      - Ingredient coverage F1 vs. ground-truth benchmark

  (B) Ingredient Matching Accuracy  [requires --with-db]
      - Top-1 accuracy, Top-3 accuracy, MRR

  (C) KG Disease Association Correctness  [requires --with-db]
      - % of ground-truth (food, disease, suitable) triples in the KG

Usage
-----
    cd evaluation

    # Single model, all parts:
    python evaluate.py --with-db --save-csv results/

    # Compare three models side-by-side:
    python evaluate.py --models llama3.1:8b-instruct-q8_0 qwen3:8b-q8_0 gemma2:9b-instruct-q8_0 --with-db --save-csv results/

    # Skip LLM (Parts B + C only):
    python evaluate.py --with-db --skip-llm

Options
-------
    --models M [M ...]  Ollama model tags to evaluate (default: llama3.1:8b-instruct-q8_0).
    --with-db           Enable Parts B and C (need Neo4j).
    --skip-llm          Skip Part A.
    --save-csv DIR       Write CSV results to DIR.
    --runs N            LLM runs per recipe (default 1).
    --neo4j-uri URI     Neo4j bolt URI (default bolt://localhost:17687).
    --neo4j-user U      Neo4j username (default neo4j).
    --neo4j-pass P      Neo4j password (default: $NEO4J_PASSWORD, else neo4j).
    --top-k K           Candidate window for matching eval (default 3).
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).parent
APP_DIR = EVAL_DIR.parent / "app"

ALLOWED_PREPARATIONS = {
    "steamed", "fried", "raw", "boiled", "roasted",
    "pan-fried", "stewed", "sautéed", "cooked",
}
REQUIRED_TOP_FIELDS = {"recipe_name", "instructions", "ingredients"}
REQUIRED_INGREDIENT_FIELDS = {"name", "preparation", "amount", "weight"}
STOP_WORDS = {"a", "an", "the", "of", "with", "and", "or", "in", "on"}

PROMPT_TEMPLATE = """\
Generate a recipe with the provided name, then extract each ingredient and its final preparation method.

Rules:
- Each ingredient must have its own preparation.
- If the preparation method is not explicitly stated, infer the most likely one, preferibly from the list:  'steamed', 'fried', 'raw', 'boiled', 'roasted', 'pan-fried, 'stewed', 'sautéed', 'cooked'.
- The ingredients have to be in singular, like 'potatoes' → 'potato', 'tomatoes' → 'tomato', 'leaves' → 'leaf', etc.
- Never leave a field empty of the JSON empty.

Return the result strictly as a JSON object with no explanations, no preamble, and no extra text.
The JSON object must have this exact structure:

{{
  "recipe_name" : "<name of the recipe introduced>",
  "instructions":[
    {{
        "text": "<each one of the steps of the recipe>"
    }}
  ],
  "ingredients": [
    {{
      "name": "<ingredient name only, clean, singular>",
      "preparation": "<one word, chosen strictly from: 'steamed', 'fried', 'raw', 'boiled', 'roasted', 'pan-fried, 'stewed', 'sautéed', 'cooked'>",
      "amount": "<amount with unit, e.g., '1 cup', '200 grams', '2 units'>",
      "weight": "<normalized weight in grams>"
    }}
  ]
}}
---

Now, analyze the following recipe (serving size: one person):
{topic}

"""


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    words = re.sub(r"[^a-z\s]", "", text.lower()).split()
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def _ingredient_match(extracted: str, reference: str) -> bool:
    return bool(_tokens(extracted) & _tokens(reference))


def _is_raw_like(text: str) -> bool:
    return any(x in text.lower() for x in ["raw", "fresh", "whole", "unprocessed"])


def _is_processed_like(text: str) -> bool:
    return any(x in text.lower() for x in [
        "cooked", "boiled", "fried", "roasted", "processed", "steamed",
        "grilled", "dehydrated", "dried", "baked", "microwaved", "powdered", "smoked",
    ])


def compute_f1(extracted: list[str], expected: list[str]) -> tuple[float, float, float]:
    """Token-overlap F1 between LLM-extracted and ground-truth ingredient lists."""
    if not extracted or not expected:
        return 0.0, 0.0, 0.0
    tp_extracted = sum(1 for e in extracted if any(_ingredient_match(e, ref) for ref in expected))
    tp_expected  = sum(1 for ref in expected if any(_ingredient_match(e, ref) for e in extracted))
    precision = tp_extracted / len(extracted)
    recall    = tp_expected  / len(expected)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_f1_best(extracted: list[str], reference_sets: list[list[str]]) -> tuple[float, float, float]:
    """Best F1 over a collection of valid reference ingredient sets."""
    best = (0.0, 0.0, 0.0)
    for ref in reference_sets:
        p, r, f = compute_f1(extracted, ref)
        if f > best[2]:
            best = (p, r, f)
    return best


def compute_f1_soft(
    extracted: list[str],
    expected: list[str],
    embed_fn,
    threshold: float = 0.80,
) -> tuple[float, float, float]:
    """Semantic F1: a match counts if cosine similarity ≥ threshold."""
    if not extracted or not expected:
        return 0.0, 0.0, 0.0
    from sentence_transformers.util import cos_sim
    ext_embs = embed_fn(extracted)
    ref_embs = embed_fn(expected)
    sim = cos_sim(ext_embs, ref_embs).cpu().numpy()    # shape (n_ext, n_ref); .cpu() for GPU (e.g. Apple MPS)
    tp_ext = sum(1 for i in range(len(extracted)) if sim[i].max() >= threshold)
    tp_ref = sum(1 for j in range(len(expected))  if sim[:, j].max() >= threshold)
    precision = tp_ext / len(extracted)
    recall    = tp_ref / len(expected)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_f1_soft_best(
    extracted: list[str],
    reference_sets: list[list[str]],
    embed_fn,
    threshold: float = 0.80,
) -> tuple[float, float, float]:
    """Best soft F1 over multiple valid reference sets."""
    best = (0.0, 0.0, 0.0)
    for ref in reference_sets:
        p, r, f = compute_f1_soft(extracted, ref, embed_fn, threshold)
        if f > best[2]:
            best = (p, r, f)
    return best


# ---------------------------------------------------------------------------
# Part A - LLM Structured Output Quality
# ---------------------------------------------------------------------------

def evaluate_llm(
    benchmark: dict,
    model_tag: str = "llama3.1:8b-instruct-q8_0",
    runs_per_recipe: int = 1,
    soft_model=None,
    soft_threshold: float = 0.80,
    max_recipes: int | None = None,
) -> list[dict]:
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_ollama import OllamaLLM
    except ImportError as exc:
        sys.exit(f"[ERROR] Missing dependency: {exc}")

    llm   = OllamaLLM(model=model_tag, options={"temperature": 0})
    prompt = PromptTemplate(input_variables=["topic"], template=PROMPT_TEMPLATE)
    chain  = prompt | llm

    recipes = benchmark["recipes"]
    if max_recipes:
        recipes = recipes[:max_recipes]
    n = len(recipes)
    use_soft = soft_model is not None
    embed_fn = (lambda texts: soft_model.encode(texts, convert_to_tensor=True)) if use_soft else None

    print(f"\n{'='*62}")
    print(f"  PART A - LLM Quality  model={model_tag}  ({n} recipes × {runs_per_recipe} run(s))")
    if use_soft:
        print(f"  Soft matching enabled  threshold={soft_threshold}")
    print(f"{'='*62}\n")

    rows = []
    for idx, recipe in enumerate(recipes, 1):
        name      = recipe["name"]
        expected  = recipe["expected_ingredients"]
        variants  = recipe.get("ingredient_variants", [])
        all_refs  = [expected] + variants
        print(f"  [{idx:2d}/{n}] {name} ...", end=" ", flush=True)

        run_metrics = []
        for _ in range(runs_per_recipe):
            try:
                raw = chain.invoke({"topic": name})
            except Exception as exc:
                print(f"\n  [WARNING] LLM call failed for '{name}': {exc}")
                run_metrics.append({"json_valid": False, "fields_complete": 0.0,
                                    "prep_valid_rate": 0.0, "precision": 0.0,
                                    "recall": 0.0, "f1": 0.0,
                                    "f1_flexible": 0.0, "f1_soft": 0.0,
                                    "n_extracted": 0})
                continue

            try:
                data       = json.loads(raw)
                json_valid = True
            except json.JSONDecodeError:
                data       = {}
                json_valid = False

            # Top-level field completeness
            present_top     = sum(1 for f in REQUIRED_TOP_FIELDS if f in data)
            fields_complete = present_top / len(REQUIRED_TOP_FIELDS)

            # Ingredient-level field completeness (averaged, then combined)
            ingredients = data.get("ingredients", []) if json_valid else []
            if ingredients:
                ing_scores = [
                    sum(1 for f in REQUIRED_INGREDIENT_FIELDS if ing.get(f))
                    / len(REQUIRED_INGREDIENT_FIELDS)
                    for ing in ingredients
                ]
                fields_complete = (fields_complete + float(np.mean(ing_scores))) / 2

            # Preparation method validity
            prep_list = [(ing.get("preparation") or "").lower().strip() for ing in ingredients]
            prep_valid_rate = (
                sum(1 for p in prep_list if p in ALLOWED_PREPARATIONS) / len(prep_list)
                if prep_list else 0.0
            )

            # Ingredient coverage - strict (vs. canonical ground truth only)
            extracted_names = [(ing.get("name") or "").lower() for ing in ingredients if ing.get("name")]
            precision, recall, f1 = compute_f1(extracted_names, expected)

            # Ingredient coverage - flexible (best F1 over all valid variants)
            _, _, f1_flexible = compute_f1_best(extracted_names, all_refs)

            # Ingredient coverage - soft semantic matching
            if use_soft and extracted_names:
                _, _, f1_soft = compute_f1_soft_best(extracted_names, all_refs, embed_fn, soft_threshold)
            else:
                f1_soft = 0.0

            # Extra ingredients: extracted but not matching any reference variant
            all_ref_flat = [ing for ref in all_refs for ing in ref]
            extras = [e for e in extracted_names
                      if not any(_ingredient_match(e, ref) for ref in all_ref_flat)]

            run_metrics.append({
                "json_valid": json_valid, "fields_complete": fields_complete,
                "prep_valid_rate": prep_valid_rate, "precision": precision,
                "recall": recall, "f1": f1, "f1_flexible": f1_flexible,
                "f1_soft": f1_soft, "n_extracted": len(extracted_names),
                "extras": extras,
            })

        row = {
            "recipe":          name,
            "json_valid_rate": sum(m["json_valid"] for m in run_metrics) / runs_per_recipe,
            "fields_complete": float(np.mean([m["fields_complete"]  for m in run_metrics])),
            "prep_valid_rate": float(np.mean([m["prep_valid_rate"]  for m in run_metrics])),
            "precision":       float(np.mean([m["precision"]        for m in run_metrics])),
            "recall":          float(np.mean([m["recall"]           for m in run_metrics])),
            "f1":              float(np.mean([m["f1"]               for m in run_metrics])),
            "f1_flexible":     float(np.mean([m["f1_flexible"]      for m in run_metrics])),
            "f1_soft":         float(np.mean([m["f1_soft"]          for m in run_metrics])),
            "avg_n_extracted": float(np.mean([m["n_extracted"]      for m in run_metrics])),
            "n_expected":      len(expected),
            "extra_ingredients": "|".join(sorted(set(
                e for m in run_metrics for e in m.get("extras", [])
            ))),
        }
        rows.append(row)
        soft_str = f"  F1-soft={row['f1_soft']:.2f}" if use_soft else ""
        print(f"F1={row['f1']:.2f}  F1-flex={row['f1_flexible']:.2f}{soft_str}  valid={row['json_valid_rate']:.0%}")

    return rows


def print_llm_table(rows: list[dict]) -> None:
    has_soft = any(r.get("f1_soft", 0) > 0 for r in rows)
    hdr = (
        f"{'Recipe':<26} {'Valid':>6} {'Complete':>9} {'PrepOK':>7} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'F1-flex':>8}"
        + (f" {'F1-soft':>7}" if has_soft else "")
        + f" {'#Ext':>5} {'#GT':>4}"
    )
    sep = "-" * len(hdr)
    print(f"\n{'='*len(hdr)}")
    print("  PART A - Results")
    print(f"{'='*len(hdr)}")
    print(hdr)
    print(sep)
    for r in rows:
        line = (
            f"{r['recipe']:<26} "
            f"{r['json_valid_rate']:>6.0%} "
            f"{r['fields_complete']:>9.1%} "
            f"{r['prep_valid_rate']:>7.1%} "
            f"{r['precision']:>6.2f} "
            f"{r['recall']:>6.2f} "
            f"{r['f1']:>6.2f} "
            f"{r.get('f1_flexible', r['f1']):>8.2f}"
        )
        if has_soft:
            line += f" {r.get('f1_soft', 0):>7.2f}"
        line += f" {r['avg_n_extracted']:>5.1f} {r['n_expected']:>4d}"
        print(line)
    print(sep)
    keys = ["json_valid_rate", "fields_complete", "prep_valid_rate", "precision", "recall", "f1", "f1_flexible", "f1_soft"]
    avgs = {k: float(np.mean([r.get(k, 0) for r in rows])) for k in keys}
    avg_line = (
        f"{'AVERAGE':<26} "
        f"{avgs['json_valid_rate']:>6.0%} "
        f"{avgs['fields_complete']:>9.1%} "
        f"{avgs['prep_valid_rate']:>7.1%} "
        f"{avgs['precision']:>6.2f} "
        f"{avgs['recall']:>6.2f} "
        f"{avgs['f1']:>6.2f} "
        f"{avgs['f1_flexible']:>8.2f}"
    )
    if has_soft:
        avg_line += f" {avgs['f1_soft']:>7.2f}"
    print(avg_line)
    summary = (
        f"\n  → JSON Valid={avgs['json_valid_rate']:.1%}  "
        f"Completeness={avgs['fields_complete']:.1%}  "
        f"Prep Valid={avgs['prep_valid_rate']:.1%}  "
        f"F1-strict={avgs['f1']:.3f}  F1-flexible={avgs['f1_flexible']:.3f}"
    )
    if has_soft:
        summary += f"  F1-soft={avgs['f1_soft']:.3f}"
    print(summary + "\n")


# ---------------------------------------------------------------------------
# Part B - Ingredient Matching Accuracy
# ---------------------------------------------------------------------------

def _connect_neo4j(uri: str, user: str, password: str):
    from neo4j import GraphDatabase
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        return driver
    except Exception as exc:
        sys.exit(f"[ERROR] Cannot connect to Neo4j at {uri}: {exc}")


def _load_food_df(driver):
    import pandas as pd
    with driver.session() as session:
        raw = session.run(
            "MATCH (f:Composition) RETURN f.food_name AS food_name, f.id AS id, f.FooDB_ID AS food_id"
        ).data()
    df = pd.DataFrame(raw)
    df["food_name"] = df["food_name"].fillna("").astype(str)
    df["is_raw"]       = df["food_name"].apply(_is_raw_like)
    df["is_processed"] = df["food_name"].apply(_is_processed_like)
    return df


def _matching_top_k(query: str, df, db_embeddings, model, k: int = 3) -> list[tuple[str, float]]:
    """Top-k candidates, filtering processed entries to match app behaviour."""
    from sentence_transformers.util import cos_sim
    from rapidfuzz.distance import JaroWinkler

    input_emb  = model.encode(query, convert_to_tensor=True)
    cos_scores = cos_sim(input_emb, db_embeddings)[0].cpu().numpy()

    df = df.copy()
    df["semantic_score"] = cos_scores
    df["string_score"]   = df["food_name"].apply(
        lambda x: JaroWinkler.normalized_similarity(x.lower(), query.lower())
    )
    df["final_score"] = 0.7 * df["semantic_score"] + 0.3 * df["string_score"]

    candidates = df[~df["is_processed"]]
    top = candidates.nlargest(k, "final_score")[["food_name", "final_score"]].values.tolist()
    return [(food_name, score) for food_name, score in top]


def _is_correct(food_name: str, key_words: list[str]) -> bool:
    name_lower = food_name.lower()
    return any(kw.lower() in name_lower for kw in key_words)


def evaluate_matching(benchmark: dict, driver, top_k: int = 3) -> list[dict]:
    from sentence_transformers import SentenceTransformer

    print(f"\n{'='*62}")
    print(f"  PART B - Ingredient Matching Accuracy  (top-{top_k} window)")
    print(f"{'='*62}\n")

    df = _load_food_df(driver)
    print(f"  Loaded {len(df):,} food entries from Neo4j.")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Computing embeddings ...", end=" ", flush=True)
    db_embeddings = model.encode(df["food_name"].tolist(), convert_to_tensor=True)
    print("done.\n")

    rows = []
    queries = benchmark["matching_queries"]
    n = len(queries)

    for idx, item in enumerate(queries, 1):
        query     = item["query"]
        key_words = item["key_words"]
        print(f"  [{idx:2d}/{n}] {query:<20} ", end="", flush=True)

        candidates = _matching_top_k(query, df, db_embeddings, model, k=top_k)

        top1_name, top1_score = candidates[0]
        top1_correct  = _is_correct(top1_name, key_words)
        topk_correct  = any(_is_correct(name, key_words) for name, _ in candidates)

        rr = 0.0
        for rank, (name, _) in enumerate(candidates, 1):
            if _is_correct(name, key_words):
                rr = 1.0 / rank
                break

        rows.append({
            "query":          query,
            "group":          item.get("group", "Unknown"),
            "top1_match":     top1_name,
            "top1_score":     round(top1_score, 4),
            "top1_correct":   top1_correct,
            f"top{top_k}_correct": topk_correct,
            "reciprocal_rank": round(rr, 4),
        })
        mark = "✓" if top1_correct else "✗"
        print(f"{mark}  → {top1_name}  (score={top1_score:.3f})")

    return rows


def print_matching_table(rows: list[dict], top_k: int) -> None:
    top_k_key = f"top{top_k}_correct"
    hdr = f"{'Query':<28} {'Top-1 Match':<32} {'Score':>6} {'T-1':>4} {f'T-{top_k}':>4} {'RR':>6}"
    sep = "-" * len(hdr)
    print(f"\n{'='*62}")
    print("  PART B - Per-Query Results")
    print(f"{'='*62}")
    print(hdr)
    print(sep)
    for r in rows:
        print(
            f"{r['query']:<28} "
            f"{r['top1_match'][:31]:<32} "
            f"{r['top1_score']:>6.3f} "
            f"{'✓' if r['top1_correct'] else '✗':>4} "
            f"{'✓' if r[top_k_key] else '✗':>4} "
            f"{r['reciprocal_rank']:>6.3f}"
        )
    print(sep)
    top1_acc = float(np.mean([r["top1_correct"] for r in rows]))
    topk_acc = float(np.mean([r[top_k_key]      for r in rows]))
    mrr      = float(np.mean([r["reciprocal_rank"] for r in rows]))
    print(f"{'OVERALL':<28} {'':32} {'':6} {top1_acc:>4.0%} {topk_acc:>4.0%} {mrr:>6.3f}")
    print(f"\n  → Top-1={top1_acc:.1%}  Top-{top_k}={topk_acc:.1%}  MRR={mrr:.3f}")

    # Per-group breakdown
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for r in rows:
        groups[r.get("group", "Unknown")].append(r)

    if len(groups) > 1:
        print(f"\n{'='*62}")
        print("  PART B - Per-Group Breakdown")
        print(f"{'='*62}")
        ghdr = f"{'Group':<32} {'N':>4} {'Top-1':>6} {f'Top-{top_k}':>6} {'MRR':>6}"
        print(ghdr)
        print("-" * len(ghdr))
        group_summary = []
        for grp in sorted(groups):
            grp_rows = groups[grp]
            g_top1 = float(np.mean([r["top1_correct"]     for r in grp_rows]))
            g_topk = float(np.mean([r[top_k_key]          for r in grp_rows]))
            g_mrr  = float(np.mean([r["reciprocal_rank"]  for r in grp_rows]))
            n      = len(grp_rows)
            print(f"{grp:<32} {n:>4} {g_top1:>6.1%} {g_topk:>6.1%} {g_mrr:>6.3f}")
            group_summary.append({"group": grp, "n": n, "top1": g_top1, f"top{top_k}": g_topk, "mrr": g_mrr})
        print()
        return group_summary
    print()


# ---------------------------------------------------------------------------
# Part C - KG Disease Association Correctness
# ---------------------------------------------------------------------------

def evaluate_kg(benchmark: dict, driver) -> list[dict]:
    ground_truth = benchmark.get("kg_ground_truth", [])
    n = len(ground_truth)

    print(f"\n{'='*62}")
    print(f"  PART C - KG Disease Association Correctness  ({n} triples)")
    print(f"{'='*62}\n")

    rows = []
    with driver.session() as session:
        for item in ground_truth:
            food    = item["food_name"]
            disease = item["disease"]
            expected_suitable = item["suitable"]
            note    = item.get("note", "")

            result = session.run(
                """
                MATCH (f:Food {Name: $food})-[r:Affects]->(d:Disease {Disease: $disease})
                RETURN r.`Suitable for Disease` AS suitable
                """,
                food=food, disease=disease,
            ).data()

            if result:
                actual_suitable = result[0]["suitable"]
                found    = True
                correct  = (bool(actual_suitable) == bool(expected_suitable))
            else:
                found   = False
                correct = False
                actual_suitable = None

            status = "✓" if correct else ("✗ wrong suitability" if found else "✗ not found")
            print(f"  {food:<22} → {disease:<30} {status}")

            rows.append({
                "food":              food,
                "disease":           disease,
                "expected_suitable": expected_suitable,
                "actual_suitable":   actual_suitable,
                "found":             found,
                "correct":           correct,
                "note":              note,
            })

    return rows


def print_kg_table(rows: list[dict]) -> None:
    hdr = f"{'Food':<22} {'Disease':<32} {'Exp':>5} {'Got':>5} {'OK':>4}"
    sep = "-" * len(hdr)
    print(f"\n{'='*62}")
    print("  PART C - Results")
    print(f"{'='*62}")
    print(hdr)
    print(sep)
    for r in rows:
        exp = "T" if r["expected_suitable"] else "F"
        got = ("T" if r["actual_suitable"] else "F") if r["found"] else "-"
        ok  = "✓" if r["correct"] else "✗"
        print(f"{r['food']:<22} {r['disease']:<32} {exp:>5} {got:>5} {ok:>4}")
    print(sep)
    hit_rate = float(np.mean([r["correct"] for r in rows]))
    found_rate = float(np.mean([r["found"] for r in rows]))
    print(f"{'SUMMARY':<22} {'':32} {'':5} {'':5} {hit_rate:>4.0%}")
    print(f"\n  → Found in KG={found_rate:.1%}  Suitability Correct={hit_rate:.1%}\n")


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(llm_results_by_model: dict, matching_rows, kg_rows, out_dir: str, top_k: int, group_summary=None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    for model_tag, llm_rows in llm_results_by_model.items():
        safe_name = re.sub(r"[^a-z0-9]", "_", model_tag.lower())
        path = os.path.join(out_dir, f"part_a_llm_{safe_name}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "recipe", "json_valid_rate", "fields_complete", "prep_valid_rate",
                "precision", "recall", "f1", "f1_flexible", "f1_soft",
                "avg_n_extracted", "n_expected", "extra_ingredients",
            ], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(llm_rows)
        print(f"  Saved → {path}")

    if matching_rows:
        top_k_key = f"top{top_k}_correct"
        path = os.path.join(out_dir, "part_b_matching.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "query", "group", "top1_match", "top1_score", "top1_correct", top_k_key, "reciprocal_rank",
            ])
            writer.writeheader()
            writer.writerows(matching_rows)
        print(f"  Saved → {path}")

    if kg_rows:
        path = os.path.join(out_dir, "part_c_kg.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "food", "disease", "expected_suitable", "actual_suitable", "found", "correct", "note",
            ])
            writer.writeheader()
            writer.writerows(kg_rows)
        print(f"  Saved → {path}")

    if group_summary:
        top_k_col = f"top{top_k}"
        path = os.path.join(out_dir, "part_b_by_group.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["group", "n", "top1", top_k_col, "mrr"])
            writer.writeheader()
            writer.writerows(group_summary)
        print(f"  Saved → {path}")

    if len(llm_results_by_model) > 1:
        path = os.path.join(out_dir, "part_a_comparison.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "model", "json_valid_rate", "fields_complete", "prep_valid_rate",
                "precision", "recall", "f1", "f1_flexible", "f1_soft",
            ])
            writer.writeheader()
            for model_tag, rows in llm_results_by_model.items():
                writer.writerow({
                    "model": model_tag,
                    **{k: round(float(np.mean([r.get(k, 0) for r in rows])), 4)
                       for k in ["json_valid_rate", "fields_complete", "prep_valid_rate",
                                 "precision", "recall", "f1", "f1_flexible", "f1_soft"]},
                })
        print(f"  Saved → {path}")


def print_model_comparison(llm_results_by_model: dict) -> None:
    if len(llm_results_by_model) < 2:
        return
    print(f"\n{'='*80}")
    print("  PART A - Model Comparison Summary")
    print(f"{'='*80}")
    all_rows_flat = [r for rows in llm_results_by_model.values() for r in rows]
    has_soft = any(r.get("f1_soft", 0) > 0 for r in all_rows_flat)
    hdr = (
        f"{'Model':<36} {'Valid':>6} {'Complete':>9} {'PrepOK':>7} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'F1-flex':>8}"
        + (f" {'F1-soft':>7}" if has_soft else "")
    )
    print(hdr)
    print("-" * len(hdr))
    for model_tag, rows in llm_results_by_model.items():
        avgs = {k: float(np.mean([r.get(k, 0) for r in rows]))
                for k in ["json_valid_rate", "fields_complete", "prep_valid_rate",
                           "precision", "recall", "f1", "f1_flexible", "f1_soft"]}
        label = model_tag[:35]
        line = (
            f"{label:<36} "
            f"{avgs['json_valid_rate']:>6.0%} "
            f"{avgs['fields_complete']:>9.1%} "
            f"{avgs['prep_valid_rate']:>7.1%} "
            f"{avgs['precision']:>6.2f} "
            f"{avgs['recall']:>6.2f} "
            f"{avgs['f1']:>6.2f} "
            f"{avgs['f1_flexible']:>8.2f}"
        )
        if has_soft:
            line += f" {avgs['f1_soft']:>7.2f}"
        print(line)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FoodMedKG quantitative evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--models",     nargs="+",
                        default=["llama3.1:8b-instruct-q8_0"],
                        metavar="MODEL",
                        help="Ollama model tag(s) to evaluate for Part A")
    parser.add_argument("--with-db",    action="store_true")
    parser.add_argument("--skip-llm",   action="store_true")
    parser.add_argument("--save-csv",   metavar="DIR")
    parser.add_argument("--runs",       type=int, default=1, metavar="N")
    parser.add_argument("--neo4j-uri",  default="bolt://localhost:17687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-pass", default=os.environ.get("NEO4J_PASSWORD", "neo4j"))
    parser.add_argument("--top-k",          type=int,   default=3,    metavar="K")
    parser.add_argument("--soft-threshold", type=float, default=None, metavar="T",
                        help="Enable soft semantic F1 with cosine similarity threshold (e.g. 0.80)")
    parser.add_argument("--max-recipes",   type=int,   default=None, metavar="N",
                        help="Limit Part A to the first N recipes (for quick testing)")
    parser.add_argument("--benchmark",     default="benchmark.json", metavar="FILE",
                        help="Benchmark JSON file to use (default: benchmark.json)")
    args = parser.parse_args()

    benchmark_path = EVAL_DIR / args.benchmark
    with open(benchmark_path) as f:
        benchmark = json.load(f)

    llm_results_by_model: dict[str, list[dict]] = {}
    matching_rows = None
    kg_rows       = None
    group_summary = None

    soft_model = None
    if not args.skip_llm and args.soft_threshold is not None:
        try:
            from sentence_transformers import SentenceTransformer
            print(f"\n  Loading embedding model for soft matching (threshold={args.soft_threshold}) ...")
            soft_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("  Done.\n")
        except ImportError:
            print("  [WARN] sentence-transformers not installed - soft matching disabled.")

    if not args.skip_llm:
        for model_tag in args.models:
            rows = evaluate_llm(
                benchmark, model_tag=model_tag, runs_per_recipe=args.runs,
                soft_model=soft_model, soft_threshold=args.soft_threshold or 0.80,
                max_recipes=args.max_recipes,
            )
            print_llm_table(rows)
            llm_results_by_model[model_tag] = rows
        if len(args.models) > 1:
            print_model_comparison(llm_results_by_model)

    if args.with_db:
        driver = _connect_neo4j(args.neo4j_uri, args.neo4j_user, args.neo4j_pass)

        matching_rows = evaluate_matching(benchmark, driver, top_k=args.top_k)
        group_summary = print_matching_table(matching_rows, top_k=args.top_k)

        kg_rows = evaluate_kg(benchmark, driver)
        print_kg_table(kg_rows)

        driver.close()

    if args.save_csv:
        print(f"\n{'='*62}")
        print(f"  Saving CSVs to {args.save_csv}/")
        print(f"{'='*62}")
        save_csv(llm_results_by_model, matching_rows, kg_rows, args.save_csv,
                 top_k=args.top_k, group_summary=group_summary if args.with_db else None)

    print("Done.\n")


if __name__ == "__main__":
    main()
