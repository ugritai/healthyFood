#!/usr/bin/env python3
"""
Step 2 - extract ingredients with the LLMs (needs Ollama).

Collects the ingredient names that each model actually extracts, so that the
matcher and the KG coverage can be evaluated on real LLM output instead of
hand-written queries. Uses the prompt and settings of the LLM evaluation
(PROMPT_TEMPLATE from evaluate.py, temperature 0, up to three attempts when
the output is not valid JSON).

Recipe sets (--recipes):
    benchmark   the 50 recipes of the LLM evaluation (benchmark.json and
                benchmark_batch2.json), with the recipe name as input, as in
                Sec. 5.1 (default)
    held-out    the RecipeNLG recipes sampled by sample_recipes.py
                (data/recipes.jsonl), with the full recipe text as input

Usage
-----
    cd evaluation/linking
    python extract.py --models llama3.1:8b-instruct-q8_0 qwen3:8b-q8_0
    python extract.py --recipes held-out --models qwen3:8b-q8_0

Output: data/extractions_<recipes>.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

LINKING_DIR = Path(__file__).parent
EVAL_DIR = LINKING_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
from evaluate import PROMPT_TEMPLATE  # noqa: E402

MAX_RETRIES = 3
PAPER_MODELS = ["llama3.1:8b-instruct-q8_0", "qwen3:8b-q8_0", "gemma2:9b-instruct-q8_0",
                "mistral:7b-instruct-q8_0", "dolphin-llama3:8b-v2.9-q8_0"]


def benchmark_recipes() -> list[dict]:
    recipes = []
    for batch, f in (("batch1", "benchmark.json"), ("batch2", "benchmark_batch2.json")):
        data = json.loads((EVAL_DIR / f).read_text(encoding="utf-8"))
        for i, r in enumerate(data["recipes"] if isinstance(data, dict) else data, 1):
            recipes.append({"recipe_id": f"{batch}-{i:02d}", "lang": "en", "title": r["name"], "input": r["name"]})
    return recipes


def held_out_recipes() -> list[dict]:
    recipes = []
    for line in (LINKING_DIR / "data" / "recipes.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            r["input"] = "\n".join([r["title"], "", "Ingredients:", *r["ingredients"], "", "Directions:", *r["directions"]])
            recipes.append(r)
    return recipes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=PAPER_MODELS, help="Ollama model tags (default: the five in the paper)")
    p.add_argument("--recipes", choices=["benchmark", "held-out"], default="benchmark")
    args = p.parse_args()

    from langchain_core.prompts import PromptTemplate
    from langchain_ollama import OllamaLLM

    recipes = benchmark_recipes() if args.recipes == "benchmark" else held_out_recipes()
    OUT = LINKING_DIR / "data" / f"extractions_{args.recipes}.jsonl"
    OUT.parent.mkdir(exist_ok=True)
    prompt = PromptTemplate(input_variables=["topic"], template=PROMPT_TEMPLATE)

    with OUT.open("w", encoding="utf-8") as f:
        for model in args.models:
            chain = prompt | OllamaLLM(model=model, options={"temperature": 0})
            n_ingredients = 0
            print(f"\n== {model}  ({len(recipes)} recipes, {args.recipes})")
            for i, r in enumerate(recipes, 1):
                data, raw = None, ""
                for _ in range(MAX_RETRIES):
                    raw = chain.invoke({"topic": r["input"]})
                    try:
                        data = json.loads(raw)
                        break
                    except json.JSONDecodeError:
                        continue
                ingredients = (data or {}).get("ingredients", []) if isinstance(data, dict) else []
                names = [str(ing.get("name", "")).strip() for ing in ingredients if isinstance(ing, dict) and ing.get("name")]
                n_ingredients += len(names)
                f.write(json.dumps({"recipe_id": r["recipe_id"], "lang": r["lang"], "region": r.get("region", ""),
                                    "title": r["title"],
                                    "recipes": args.recipes, "model": model, "json_valid": data is not None,
                                    "ingredients": names, "raw": raw}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  [{i:3d}/{len(recipes)}] {r['title'][:45]:<45} {len(names):2d} ingredients")
            print(f"  {n_ingredients} ingredient mentions")
    print(f"\nWritten {OUT}")


if __name__ == "__main__":
    main()
