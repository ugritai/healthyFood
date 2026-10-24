# FoodMedKG

Code and data for the paper *FoodMedKG: Integrating Biomedical Knowledge and
Culinary Data for Health-Aware Decision Support*.

A language model extracts the ingredients of a recipe written in natural
language, each ingredient is linked to a food in the knowledge graph, and the
graph returns its composition, its relations with diseases and healthy ageing,
and the effect of cooking methods.

Versions of data sources, models and parameters are listed in
[DATA_VERSIONS.md](DATA_VERSIONS.md).

## Repository structure

```
app/                 Streamlit application
data/                scripts that build the source tables of the graph
                     (steps in data/data_preparation.txt) and the
                     Applied Pathophysiology relations
evaluation/          LLM extraction and ingredient-matching evaluation
evaluation/linking/  matcher baselines, threshold and absent-ingredient tests
survey/              human evaluation app and analysis
```

Each folder has its own `requirements.txt` (Python 3.12).

## Application

Needs a running Neo4j database with the graph and Ollama.

```bash
pip install -r app/requirements.txt
cd app
streamlit run streamlit_app.py
```

## Evaluation

```bash
pip install -r evaluation/requirements.txt
cd evaluation
bash run_all_models.sh          # five LLMs on both benchmarks (pulled with Ollama)
python visualize.py             # figures from results/ into figures/
```

- `benchmark.json` (15 recipes) and `benchmark_batch2.json` (35 recipes): ground-truth
  ingredients and two variants per recipe.
- `evaluate.py`: extraction metrics (strict, flexible and soft F1; valid JSON and
  preparation rates; extra ingredients), ingredient matching and KG associations.
- `compute_extras.py`: adds the extra-ingredient column to existing results.

The entity-linking experiments are described in
[evaluation/linking/README.md](evaluation/linking/README.md) and the human
evaluation in [survey/README.md](survey/README.md).

## License

See [LICENSE](LICENSE).
