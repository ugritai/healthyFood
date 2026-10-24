# Entity linking: alternative matchers and real LLM extractions

Two complements to the surface-form benchmark of `../benchmark.json`
(`matching_queries`: 200 queries written from foods known to be in the KG).

```
linking/
├── baselines.py        alternative matchers, weight analysis, misspellings  → results/
├── threshold_curve.py  precision/recall of the accept decision vs. τ        → results/threshold_curve.*
│                       (benchmark + matches judged in the human evaluation)
├── negatives_ablation.py  rejection at τ of absent foods, real extractions  → results/negatives_*
├── sample_recipes.py   1. sample held-out RecipeNLG recipes                 → data/recipes.jsonl
├── extract.py          2. extract ingredients with the five LLMs            → data/extractions_<recipes>.jsonl
├── data/               inputs and intermediate files
└── results/            tables and figures
```

All scripts run from this folder with the project's `.venv` (Python 3.12).
`baselines.py` and `negatives_ablation.py` need Neo4j with the paper's graph
(`bolt://localhost:17687`); `extract.py` needs Ollama with the five models.

## Alternative matchers and the weights of the hybrid score (`baselines.py`)

Runs the 200 queries with the candidate filter (processed entries excluded)
and the correctness criterion of `evaluate.py`, and compares:

- **string only**: Jaro–Winkler, token-set ratio, TF-IDF of character 3–5-grams;
- **embeddings only**: all-MiniLM-L6-v2 (the system's model), all-mpnet-base-v2,
  bge-small-en-v1.5, bge-base-en-v1.5, e5-base-v2, multilingual-e5-base, and
  AI2Cuisine (BERT fine-tuned on food data, Morales-Garzón et al., 2025).
  The other models are downloaded automatically from Hugging Face. AI2Cuisine
  (419 MB) is downloaded from
  [Google Drive](https://drive.google.com/drive/folders/16ZXUbQukqnpAaIMkSy5hckFvAZAWZpTE?usp=sharing):
  place the model folder in `evaluation/` so that
  `evaluation/12-20_22-39_1000016b_4ep_3000wu_2e-07lr/` contains the model
  files. Without it, `baselines.py` runs the other models only;
- **hybrid** `S = w·cosine + (1 − w)·Jaro–Winkler` for w = 0.0 … 1.0 with each
  model, the system's w = 0.7, and w chosen by 5-fold cross-validation
  (w picked on four folds, evaluated on the fifth; ties broken by taking the
  middle of the tied weights).

Everything is run on the correctly spelled queries and on five misspelled
versions of them. A misspelling is one deleted, substituted or swapped
(adjacent) letter, never the first letter of a word, in a word of a minimum
length (e.g. *broccolis* → *brocolis*, *baby spinach* → *baby sninach*,
*broccoli* → *brcocoli*); each version uses its own random seed, and results
are given as mean and SD over the five versions.

| Setting | Command | Queries misspelled |
|---|---|---|
| **Paper**: 25% of queries, words of ≥ 6 letters | `python baselines.py` | ≈ 34 of 200 |
| Stress test: all queries, words of ≥ 6 letters | `python baselines.py --typo-rate 1 --tag _long6` | ≈ 128 |
| Stress test: all queries, words of ≥ 4 letters | `python baselines.py --typo-rate 1 --typo-min-len 4 --tag _all4` | ≈ 196 |

Outputs in `results/` (with the tag suffix): `baselines_summary.csv` (one row
per method: accuracy on correctly spelled queries, on the misspelled versions
and on the misspelled queries only), `baselines_results.csv` (top-1, top-3 and
MRR for every method and version), `weight_sweep.csv` / `weight_sweep.png`
(accuracy by w), `typo_queries.csv` (the misspelled queries).

## Rejection of absent ingredients on real LLM extractions

Tests the reject side of the accept/reject decision at τ on the ingredient
names that an LLM actually extracts from held-out recipes, without annotators.

```bash
python sample_recipes.py --corpus ../../data/RecipeNLG_dataset.csv   # 1. 10 regions x 10 recipes
python extract.py --recipes held-out                                  # 2. five models of the paper
python negatives_ablation.py                                          # 3. Qwen3-8B, τ = 0.68
python negatives_ablation.py --model all                              #    the five models
```

1. **Recipes.** RecipeNLG (Kaggle: `paultimothymooney/recipenlg`,
   `RecipeNLG_dataset.csv`, not redistributed; place it in `data/` of the
   repository root) has no cuisine field, so a region is assigned from title
   keywords (`REGIONS` in `sample_recipes.py`); titles matching several regions
   are skipped. Recipes have 2–4 ingredients, counted on RecipeNLG's `NER`
   column, and the 50 recipes of the LLM evaluation are excluded. The sampled
   recipes are in `data/recipes.jsonl`.
2. **Extraction.** Same prompt and temperature (0) as the LLM evaluation and the
   app, with the full recipe text as input. The outputs used in the paper are
   in `data/extractions_held-out.jsonl`.
3. **Rejection test.** For each distinct extracted name, every KG entry whose
   name contains any of its words (singular/plural forms too) is removed from
   the matcher's candidates, so the food is absent by construction, and the
   matcher is run on the remaining entries; the correct decision is to reject
   (S < τ).

Outputs in `results/`: `negatives_items.csv` (one row per name: entries
removed, top-1 with the full KG and after removal, scores, rejected),
`negatives_results.csv` (rejection rate per name and per mention) and
`negatives_sweep.csv` (rejection rate over τ). Synonyms written with a
different word (e.g. cilantro / Coriander) or spelling (yogurt / Yoghurt) stay
among the candidates; the accepted cases are printed for inspection.
