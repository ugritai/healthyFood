# Knowledge graph

The application, the ingredient-matching and KG evaluations
(`evaluation/evaluate.py --with-db`) and the entity-linking experiments
(`evaluation/linking/baselines.py`, `negatives_ablation.py`) query FoodMedKG in
Neo4j. The graph can be obtained in two ways:

1. **Load the released dump** (recommended). This is the exact graph used in
   the paper.
2. **Rebuild it** from the source tables and apply the post-load edits.

Versions and SHA-256 checksums of all sources are listed in
[../DATA_VERSIONS.md](../DATA_VERSIONS.md).

## 1. Load the released dump

`data/neo4j_dump/foodmedkg.dump` (Neo4j 2025.09.0 Community, about 11 MB).

```bash
# with Neo4j stopped
neo4j-admin database load foodmedkg --from-path=data/neo4j_dump --overwrite-destination
neo4j start
# in Neo4j Browser or cypher-shell (system database):
CREATE DATABASE foodmedkg IF NOT EXISTS
```

The scripts connect to `bolt://localhost:17687` by default. Either set
`server.bolt.listen_address=:17687` in `neo4j.conf`, or pass the URI and
credentials of your instance (`--neo4j-uri`, `--neo4j-user`, `--neo4j-pass`).
The password is read from the `NEO4J_PASSWORD` environment variable:

```bash
export NEO4J_PASSWORD=<your Neo4j password>
```

## 2. Rebuild the graph

### 2.1 Source data

| Source | Where to get it | Used for |
|---|---|---|
| FooDB | CSV release from https://foodb.ca/downloads | Foods, food groups, composition |
| Food–Disease Relation dataset | Nacev et al. (2021) | Food–disease relations |
| Applied Pathophysiology for Dietetics | Compiled by the authors from [47]; `data/applied_pathophysiology_relations.csv` | Food–disease relations |
| Healthy ageing study | Digitised by the authors from Tessier, A.J., Wang, F., Korat, A.A. et al. Optimal dietary patterns for healthy aging. *Nat Med* 31, 1644–1652 (2025). https://doi.org/10.1038/s41591-025-03570-5 | Healthy-ageing values per food category |
| Cooking methods | Created manually by the authors | Effect of cooking methods |

### 2.2 Source tables

The numbered scripts build the source tables from the raw data; the steps,
including the manual ones, are described in
[data_preparation.txt](data_preparation.txt). The intermediate tables were
kept in MongoDB (`mongodb://localhost:27017`).

Because several steps are manual (cleaning of the Food–Disease Relation
dataset, the cooking-methods table and the mapping of healthy-ageing
categories to FooDB foods), the seven resulting tables are released in
`data/source_tables/` so that the graph can be rebuilt without repeating
them:

| Table | Content |
|---|---|
| `Food_Simplificada.csv` | Foods and their FooDB group |
| `FooDB_Alimento_Constituyentes.csv` | Composition (USDA and DTU constituents) |
| `FDR_FooDB_Id.csv` | Food–Disease Relation dataset, cleaned and linked to FooDB |
| `FAALD_FooDB_Id.csv` | Applied Pathophysiology relations linked to FooDB |
| `Alimento_Enfermedad_Completa.csv` | Merged food–disease table, with DOIDs |
| `Cooking_Methods.csv` | Cooking methods |
| `ES_Final.csv` | Healthy-ageing levels per food and food group |

Their SHA-256 checksums are in `DATA_VERSIONS.md` and are also stored in the
graph (`Import_Source` nodes), so a rebuilt graph can be checked against the
released one.

### 2.3 Load into Neo4j

```bash
pip install -r data/requirements.txt
python data/load_neo4j.py     # creates nodes and relations in database foodmedkg
```

### 2.4 Post-load edits

The scripts in `data/kg_edits/` correct and extend the loaded graph. Each one
documents in its header what it changes and why. Run them in this order:

| # | Script | Change |
|---|---|---|
| 1 | `add_provenance.py` | Provenance nodes (`Dataset_Metadata`, `Import_Run`, `Import_Source`) |
| 2 | `add_aging_values.py` | Decimal healthy-ageing values next to the 0–4 levels |
| 3 | `fix_traced_errors.py` | Errors found when tracing each food–disease edge to its source |
| 4 | `add_disease_doid.py` | Disease Ontology identifiers on Disease nodes |
| 5 | `add_fruit_juices.py` | The "Fruit juices" healthy-ageing category lost in the original build |
| 6 | `add_mixed_effects.py` | Relations whose sources report both beneficial and harmful effects |
| 7 | `add_association_scope.py` | Whether each association was reported for the food or derived from its group (`affects_scope.csv`) |
| 8 | `remove_rejected_edge.py` | Removes the relation rejected in the human evaluation (item I068) |
| 9 | `add_consensus.py` | Consensus grades from the human evaluation |

Steps 8 and 9 use the results of the human evaluation. Step 9 reads
`survey/analysis/results/consensus_grades.csv`, produced by
`survey/analysis/run_all.py` from the survey responses
([Google Drive](https://drive.google.com/drive/folders/1FeyDIut6ksEiC9HC-KRVrcTrvrxaqQ7q?usp=sharing); see
[survey/README.md](../survey/README.md)).
