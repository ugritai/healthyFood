#!/usr/bin/env python3
"""
Step 1 - sample held-out recipes from RecipeNLG.

Draws real recipes from RecipeNLG (Kaggle: paultimothymooney/recipenlg,
RecipeNLG_dataset.csv), excluding the recipes used in the LLM evaluation
(benchmark.json, benchmark_batch2.json), and writes them as one JSON object
per line (data/recipes.jsonl) for `extract.py --recipes held-out`.

Selection:
  - short recipes: between --min-ingredients and --max-ingredients
    ingredients, counted on RecipeNLG's NER column (one clean name per
    ingredient), so quantities and notes in the raw lines do not matter;
  - regional variety: RecipeNLG has no cuisine field, so each recipe is
    assigned a region when its title contains a keyword of exactly one
    region in REGIONS (titles matching several regions are skipped), and the
    same number of recipes is drawn per region (--per-region).

The NER list is kept with each recipe as the ingredients named in the source.

Usage
-----
    cd evaluation/linking
    python sample_recipes.py --corpus ../../data/RecipeNLG_dataset.csv
    python sample_recipes.py --corpus ../../data/RecipeNLG_dataset.csv --per-region 10 --max-ingredients 4
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

LINKING_DIR = Path(__file__).parent
EVAL_DIR = LINKING_DIR.parent
OUT = LINKING_DIR / "data" / "recipes.jsonl"

# Title keywords per region; matched as whole words, case-insensitive
REGIONS = {
    "Mexican": ["taco", "tacos", "enchilada", "enchiladas", "quesadilla", "salsa", "guacamole", "tortilla", "burrito",
                "tamale", "tamales", "mole", "mexican", "chilaquiles", "pozole"],
    "Italian": ["pasta", "risotto", "pesto", "lasagna", "lasagne", "gnocchi", "bruschetta", "focaccia", "italian",
                "carbonara", "tiramisu", "polenta", "frittata", "marinara", "minestrone"],
    "Indian": ["curry", "masala", "dal", "dhal", "chutney", "tandoori", "biryani", "paneer", "naan", "raita", "korma",
               "indian", "samosa", "lassi", "pakora"],
    "Chinese": ["chinese", "wonton", "chow", "lo mein", "szechuan", "sichuan", "kung pao", "dim sum", "bok choy",
                "hoisin", "fried rice", "egg roll"],
    "Japanese": ["japanese", "teriyaki", "sushi", "miso", "tempura", "udon", "ramen", "soba", "edamame", "yakitori",
                 "katsu"],
    "Korean / Southeast Asian": ["korean", "kimchi", "bulgogi", "thai", "pad thai", "satay", "vietnamese", "pho",
                                 "filipino", "adobo", "indonesian", "malaysian", "laksa"],
    "Middle Eastern": ["hummus", "falafel", "tahini", "tabbouleh", "tabouli", "baba ghanoush", "shawarma", "kebab",
                       "kabob", "pita", "lebanese", "turkish", "persian", "couscous", "tagine", "moroccan"],
    "Greek / Mediterranean": ["greek", "tzatziki", "feta", "souvlaki", "moussaka", "spanakopita", "gyro", "gyros",
                              "mediterranean", "baklava"],
    "Spanish / Latin American": ["paella", "gazpacho", "spanish", "empanada", "empanadas", "churros", "chorizo",
                                 "sangria", "cuban", "brazilian", "argentine", "peruvian", "ceviche", "arepa",
                                 "arepas", "plantain", "plantains", "sofrito"],
    "French / Central European": ["french", "crepe", "crepes", "quiche", "ratatouille", "souffle", "gratin",
                                  "bourguignon", "german", "schnitzel", "strudel", "pierogi", "goulash", "polish",
                                  "hungarian", "sauerkraut"],
}


def used_recipe_names() -> set[str]:
    names = set()
    for f in ("benchmark.json", "benchmark_batch2.json"):
        data = json.loads((EVAL_DIR / f).read_text(encoding="utf-8"))
        recipes = data["recipes"] if isinstance(data, dict) else data
        names |= {r["name"].strip().lower() for r in recipes}
    return names


def as_list(value) -> list[str]:
    if isinstance(value, str) and value.strip().startswith("["):
        return [str(x) for x in json.loads(value)]
    return [s for s in str(value).splitlines() if s.strip()]


def sample_by_keywords(df: pd.DataFrame, per_region: int, seed: int) -> pd.DataFrame:
    patterns = {region: re.compile(r"\b(" + "|".join(map(re.escape, kws)) + r")\b", re.I)
                for region, kws in REGIONS.items()}

    def region_of(title: str) -> str | None:
        hits = [region for region, pattern in patterns.items() if pattern.search(title)]
        return hits[0] if len(hits) == 1 else None

    df = df.assign(cuisine=df["title"].map(region_of)).dropna(subset=["cuisine"])
    picked = []
    for region in REGIONS:
        pool = df[df["cuisine"] == region]
        if len(pool) < per_region:
            raise SystemExit(f"{region}: only {len(pool)} candidate recipes; lower --per-region or add keywords")
        picked.append(pool.sample(n=per_region, random_state=seed))
        print(f"  {region:<28} {len(pool):>7,} candidates -> {per_region}")
    return pd.concat(picked)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--corpus", required=True, help="RecipeNLG_dataset.csv")
    p.add_argument("--per-region", type=int, default=10)
    p.add_argument("--min-ingredients", type=int, default=2)
    p.add_argument("--max-ingredients", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.corpus, usecols=["title", "ingredients", "directions", "link", "NER"])
    print(f"{len(df):,} recipes in {Path(args.corpus).name}")
    df = df.dropna(subset=["title", "ingredients", "directions", "NER"])
    df["ner"] = df["NER"].map(as_list)
    df = df[df["ner"].map(len).between(args.min_ingredients, args.max_ingredients)]
    df = df[~df["title"].str.strip().str.lower().isin(used_recipe_names())]
    df = df.drop_duplicates(subset="title")
    print(f"{len(df):,} with {args.min_ingredients}–{args.max_ingredients} ingredients, not in the LLM evaluation")

    picked = sample_by_keywords(df, args.per_region, args.seed)

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for i, (_, row) in enumerate(picked.iterrows(), 1):
            f.write(json.dumps({
                "recipe_id": f"nlg-{i:03d}", "lang": "en", "region": row["cuisine"], "title": row["title"].strip(),
                "ingredients": as_list(row["ingredients"]), "directions": as_list(row["directions"]),
                "ner": row["ner"], "link": row["link"], "source": "RecipeNLG",
            }, ensure_ascii=False) + "\n")
    print(f"{len(picked)} recipes written to {OUT}")


if __name__ == "__main__":
    main()
