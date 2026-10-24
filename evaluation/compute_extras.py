#!/usr/bin/env python3
"""
Patch existing part_a_llm_*.csv files with an extra_ingredients column.

Reruns only the LLM ingredient-extraction step (no scoring, no embeddings).
Writes results back into the same CSV files.

Usage
-----
    cd evaluation
    python compute_extras.py                          # all models found in results/
    python compute_extras.py --models llama3.1:8b-instruct-q8_0
    python compute_extras.py --results results/batch2 --benchmark benchmark_batch2.json
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).parent
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
      "preparation": "<one word>",
      "amount": "<amount with unit>",
      "weight": "<normalized weight in grams>"
    }}
  ]
}}
---

Now, analyze the following recipe (serving size: one person):
{topic}

"""


def _tokens(text: str) -> set[str]:
    words = re.sub(r"[^a-z\s]", "", text.lower()).split()
    return {w for w in words if w not in STOP_WORDS and len(w) > 2}


def _ingredient_match(extracted: str, reference: str) -> bool:
    return bool(_tokens(extracted) & _tokens(reference))


def _safe_model_name(tag: str) -> str:
    return re.sub(r"[^a-z0-9]", "_", tag.lower())


def compute_extras_for_model(model_tag: str, benchmark: dict, results_dir: Path) -> None:
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_ollama import OllamaLLM
    except ImportError as exc:
        sys.exit(f"[ERROR] {exc}")

    csv_path = results_dir / f"part_a_llm_{_safe_model_name(model_tag)}.csv"
    if not csv_path.exists():
        print(f"  [SKIP] No CSV found for {model_tag} at {csv_path}")
        return

    llm    = OllamaLLM(model=model_tag, options={"temperature": 0})
    prompt = PromptTemplate(input_variables=["topic"], template=PROMPT_TEMPLATE)
    chain  = prompt | llm

    recipes = benchmark["recipes"]
    print(f"\n  Model: {model_tag}  ({len(recipes)} recipes)")

    extras_map: dict[str, str] = {}
    for idx, recipe in enumerate(recipes, 1):
        name     = recipe["name"]
        expected = recipe["expected_ingredients"]
        variants = recipe.get("ingredient_variants", [])
        all_refs = [ing for ref in [expected] + variants for ing in ref]

        print(f"    [{idx:2d}/{len(recipes)}] {name} ...", end=" ", flush=True)
        try:
            raw  = chain.invoke({"topic": name})
            data = json.loads(raw)
            ings = data.get("ingredients", [])
            extracted = [(i.get("name") or "").lower() for i in ings if i.get("name")]
        except Exception:
            extracted = []

        extras = [e for e in extracted
                  if not any(_ingredient_match(e, ref) for ref in all_refs)]
        extras_map[name] = "|".join(sorted(set(extras)))
        print(f"extras: {extras or '-'}")

    # Read existing CSV, patch column, write back
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = list(rows[0].keys()) if rows else []
    if "extra_ingredients" not in fieldnames:
        fieldnames.append("extra_ingredients")

    for row in rows:
        row["extra_ingredients"] = extras_map.get(row["recipe"], "")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Patched → {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",    nargs="+", default=None, metavar="MODEL")
    parser.add_argument("--results",   default="results",        metavar="DIR")
    parser.add_argument("--benchmark", default="benchmark.json", metavar="FILE")
    args = parser.parse_args()

    results_dir = EVAL_DIR / args.results
    with open(EVAL_DIR / args.benchmark) as f:
        benchmark = json.load(f)

    if args.models:
        models = args.models
    else:
        # Discover from existing CSVs
        csvs   = sorted(results_dir.glob("part_a_llm_*.csv"))
        models = [re.sub(r"[_]", ":", p.stem.replace("part_a_llm_", "")) for p in csvs]
        # Try to match back to real model tags via ollama list
        import subprocess
        try:
            out = subprocess.check_output(["ollama", "list"], text=True)
            available = [line.split()[0] for line in out.strip().splitlines()[1:]]
            resolved = []
            for m in models:
                match = next((a for a in available if _safe_model_name(a) == _safe_model_name(m)), m)
                resolved.append(match)
            models = resolved
        except Exception:
            pass

    print(f"Computing extras for {len(models)} model(s) in {results_dir}/")
    for model in models:
        compute_extras_for_model(model, benchmark, results_dir)

    print("\nDone. Re-run visualize.py to regenerate figures.")


if __name__ == "__main__":
    main()
