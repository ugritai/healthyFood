# Data, model and parameter versions

Versions of every external resource used in the paper.

## Data sources

| Source | Version | Notes |
|---|---|---|
| FooDB | Public CSV release available at foodb.ca, downloaded in 2025 (derived tables built July–October 2025) | Food composition. Only constituents whose source is USDA or DTU are kept (`data/3_FooDB_Pivoted.py`), so the USDA and DTU values are those included in this FooDB release. |
| USDA | Through FooDB (see above) | Not downloaded separately. |
| DTU | Through FooDB (see above) | Not downloaded separately. |
| Food–Disease Relation dataset | Nacev et al. (2021) | Food–disease relations (`FDR` tables). |
| Applied Pathophysiology for Dietetics | Relations compiled from the textbook [47] | Released in `data/applied_pathophysiology_relations.csv` (see below). |
| RecipeNLG | Kaggle `paultimothymooney/recipenlg`, `RecipeNLG_dataset.csv` | Not redistributed. Used for the held-out recipes in `evaluation/linking/`. |

`foodb_internal_id` and the `FooDB_id` property in the graph are the project's
internal FooDB food ids, not FooDB's public ids (e.g. `FOOD00534`).

### Checksums of the source tables used to build the graph

SHA-256 of the tables the graph was built from (recorded in the graph's
`Import_Source` nodes, pipeline version 2026.10.22).

| Table | SHA-256 |
|---|---|
| Food_Simplificada | `c406816ab89edc3dab4453d9c346cf3f18e06eb5fdd8ef075a9b4539bfde2ccc` |
| FooDB_Alimento_Constituyentes | `e402d3a8ef10bfb689f422d0fa43bbedf55021cb7748a5f174e83dfa57c1fc60` |
| FDR_FooDB_Id | `56a5e1db2346f893805c9b06f98b43b4103ef523b69aae293149620bed413b18` |
| FAALD_FooDB_Id | `8908c2acbe126c8ee64a8c71a8b5c6df18ca848cc9d12c4c88e3cde2392cc512` |
| Cooking_Methods | `ad38f0f0a99f86aa01688df9fc14c2649dc2b99884f00987489121885de9ac07` |
| ES_Final | `07f551e6157c080c8b325c1c7d1427b7f0dd28f413acea06695c61939072ad6f` |
| Alimento_Enfermedad_Completa | `1e46b729678eef106c891c7818493980eba5b9a0380196c159dfa6ac327691c3` |

## Applied Pathophysiology relations

`data/applied_pathophysiology_relations.csv` contains the 149 food–disease
relations compiled from [47]: food name, internal FooDB id (empty when no
FooDB food was assigned), disease, `suitable_for_disease` (1 = suitable,
0 = not suitable) and the scientific article that supports the relation
(citation and link). It contains no text from the textbook.

## Models

| Model | Version |
|---|---|
| Sentence embedding | `sentence-transformers/all-MiniLM-L6-v2`, Hugging Face revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Qwen3-8B | Ollama `qwen3:8b-q8_0`, digest `e56358ca25dd14db6853a9f68a92d717aaa6f0a94250a72d1a0f3d86a9f30130` |
| Llama 3.1 8B | Ollama `llama3.1:8b-instruct-q8_0`, digest `b158ded76fa05be6bce8a682099ce5df8c5571340a04cf63a2923464679db576` |
| Gemma 2 9B | Ollama `gemma2:9b-instruct-q8_0`, digest `54faa8324fdf5138279b835f87f39c30db7a534b4211fabad80e476ddbd9e409` |
| Mistral 7B | Ollama `mistral:7b-instruct-q8_0`, digest `2162e081e7f0e24d0e46bed166e68ae5ea05cba5f7daa64266b839c076f7715b` |
| Dolphin Llama 3 8B | Ollama `dolphin-llama3:8b-v2.9-q8_0`, digest `9b6fb904576c74715568a74357750f1b405f64c073e7ec73ffae4d9509f021d0` |

## Software

- Python 3.12; library versions in the `requirements.txt` of each folder.
- Neo4j 2025.09.0 (Community).

## Parameters

| Parameter | Value | Where |
|---|---|---|
| Hybrid matching score | 0.7 × cosine (all-MiniLM-L6-v2) + 0.3 × Jaro–Winkler | `evaluation/evaluate.py`, `app/streamlit_app.py` |
| Acceptance threshold τ | 0.68 | `TAU` in `app/streamlit_app.py`; human evaluation and `evaluation/linking/` |
| Soft-matching threshold for extraction F1 | cosine ≥ 0.80 | `evaluation/evaluate.py` |
| LLM temperature | 0 | `evaluation/evaluate.py`, `app/streamlit_app.py` |
| Prompt template and JSON schema | - | `PROMPT_TEMPLATE` in `evaluation/evaluate.py`; prompt in `app/streamlit_app.py` |
