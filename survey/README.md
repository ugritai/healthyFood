# FoodMedKG Human Evaluation Survey

A minimal Streamlit app for collecting human evaluation of FoodMedKG
relations from nutrition/dietetics/medicine students, split into rotating
blocks so each respondent only rates a subset of the item pool. Responses
are stored in a Google Sheet (there is no local persistence, since
Streamlit Community Cloud's filesystem is ephemeral and not shared across
sessions or instances).

## Project structure

```
survey/
├── app.py                       # Main Streamlit app
├── items.csv                    # Item pool with block assignments (sample data - replace with the real pool)
├── requirements.txt             # App dependencies
├── .streamlit/
│   └── secrets.toml.example     # Template for local secrets (copy to secrets.toml, do not commit the real one)
├── analysis/                    # Offline analysis of the exported responses (see section 6)
│   ├── run_all.py               # Runs every module below and writes REPORT.md
│   ├── summary.py               # Headline accuracy + every alternative metric
│   ├── common.py                # Loading, deduplication, agreement statistics
│   ├── descriptive.py           # Annotators and raw judgment distributions
│   ├── agreement.py             # Fleiss' kappa, Gwet's AC1, bootstrap CIs
│   ├── aggregation.py           # Majority vote and alternatives (Dawid–Skene, ...)
│   ├── annotator_reliability.py # Are dissenters less reliable? Robustness to annotator filters
│   ├── error_analysis.py        # Rejected relations, Not sure by outcome, claim direction
│   ├── matching.py              # Ingredient-matching task
│   ├── consensus.py             # Consensus score/grade per evaluated graph element
│   ├── disagreement.py          # Learning from disagreement: CrowdTruth, MACE, soft labels
│   ├── decision_confidence.py   # How certain each verdict is (sign test, bootstrap, entropy)
│   └── requirements.txt         # Analysis dependencies
└── README.md
```

## 1. `items.csv` format

Columns: `item_id, block_id, type, food, target, relation_text, relation_text_es, citation, link`

`relation_text` is the primary text shown per item (it names the food
explicitly, e.g. "Broccoli (steamed): ..." or "Coffee scores 1/4 on...")
since the graph's own per-type data doesn't always mention the food on its
own. `relation_text_es` is a Spanish translation, highlighted below it.

- `type` is one of `disease`, `aging`, `cooking_method`.
- `target` is the disease name / Healthy Aging sub-indicator / cooking
  method name.
- `link` may be empty (all `aging` items have real citations now, via
  Tessier et al. 2025; only if you add new rows without a source should
  citation/link be left blank rather than fabricated).
- `block_id` determines which respondents see which items - assign block
  numbers so that each sub-group of students gets a distinct `?block=N`
  link and no one is expected to rate the entire pool.

The included `items.csv` is a **real 80-item pool** (4 blocks of 20: 10
disease + 5 cooking_method + 5 aging each) extracted directly from the
project's Neo4j graph - see the git history for how each type was sourced
and what citations back it. Regenerate or replace it as the real study
design evolves.

## 1b. `ingredient_matches.csv` - the extra task

A short, separate task shown after the main items, evaluating a
different part of the system: whether the automatic matching of a
free-text recipe ingredient to a specific food entry in the database
looks correct. Columns: `match_id, block_id, query, food_group,
matched_food, score`. Sourced from real evaluation output
(`evaluation/results/part_b_matching.csv`), not invented.

It's rotated by block like the main items, but asymmetrically: the whole
evaluation dataset only contains a handful of known-incorrect matches
(there simply aren't more in the real data), so the same 3 (`M02`-`M04`)
repeat in every block - otherwise "covering more cases" would mean
inventing fake errors. Each block then adds 5 *different* correct
matches (`M05`-`M24`, 20 unique across the 4 blocks) spanning a range of
food groups and similarity scores, so the correct-match coverage is
broad even though the incorrect-match coverage is capped by what's
actually in the data.

## 2. Block routing

Respondents are sent a link like:

```
https://<your-app>.streamlit.app/?block=3
```

The app reads the `block` query parameter and filters `items.csv` to
`block_id == block`. If the parameter is missing or doesn't match any
block, the app shows a short landing screen asking the respondent to use
their assigned link - it never guesses or randomly assigns a block. Block
links must be distributed manually to each sub-group of students.

## 3. Setting up the Google Sheet + service account

1. **Create the Google Sheet** that will store responses. Note its exact
   name (or switch the app to open by key/ID - see the note in `app.py`
   where `SPREADSHEET_NAME` is defined). No worksheet setup is required - the app creates both the
   `responses` worksheet (main items) and `ingredient_match_responses`
   worksheet (extra task) with their header rows automatically on first
   submission if they don't already exist.

2. **Create a Google Cloud service account:**
   - Go to the [Google Cloud Console](https://console.cloud.google.com/),
     create or select a project.
   - Enable the **Google Sheets API** and **Google Drive API** for that
     project (APIs & Services → Library).
   - Go to APIs & Services → Credentials → Create Credentials → Service
     Account. Give it any name (e.g. `foodmedkg-survey`).
   - Open the new service account → Keys → Add Key → Create new key →
     JSON. This downloads a JSON key file - **keep it secret**, it grants
     access to anything shared with the service account.

3. **Share the Google Sheet** with the service account: open the Sheet →
   Share → paste the service account's `client_email` (found in the JSON
   key, looks like `xxx@yyy.iam.gserviceaccount.com`) → give it **Editor**
   access.

## 4. Local setup

```bash
cd survey
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the secrets template and fill it in with values from the downloaded
JSON key:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
[gcp_service_account]
type = "service_account"
project_id = "your-gcp-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-service-account@your-gcp-project-id.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-gcp-project-id.iam.gserviceaccount.com"
universe_domain = "googleapis.com"
```

**Never commit `.streamlit/secrets.toml`** - it is already excluded via
`.gitignore` at the repo root; double-check before pushing.

Update `SPREADSHEET_NAME` in `app.py` to match your real Sheet's name, and
replace `CONSENT_TEXT` with the final consent/info wording before
distributing real links.

Run locally:

```bash
streamlit run app.py
```

Then open `http://localhost:8501/?block=1` to test a specific block (block
`1` and `2` exist in the sample `items.csv`).

## 5. Deploying to Streamlit Community Cloud

1. Push this repository (or at least the `survey/` folder) to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) → New app →
   point it at this repo, with **main file path** set to `survey/app.py`
   (adjust if you deploy `survey/` as its own repo, in which case use
   `app.py`).
3. Under **Advanced settings → Secrets**, paste the same contents as your
   local `.streamlit/secrets.toml` (the `[gcp_service_account]` table).
   This is the production equivalent of the local secrets file - Streamlit
   Community Cloud exposes it to the app as `st.secrets` at runtime.
4. Deploy. Once live, construct one link per block for distribution, e.g.:
   - `https://<your-app>.streamlit.app/?block=1`
   - `https://<your-app>.streamlit.app/?block=2`
   - ... one per sub-group of students.

Streamlit Community Cloud's filesystem is ephemeral and reset on every
redeploy/restart, and multiple sessions may run on different underlying
instances - this is why the app never writes to local disk and uses
Google Sheets as the single source of truth for all responses.

## 6. Running the analysis

The analysis runs locally (not part of the deployed app) on an Excel
export of the Google Sheet.

**Data of the paper.** The anonymised responses of the human evaluation
used in the paper are available on
[Google Drive](https://drive.google.com/drive/folders/1FeyDIut6ksEiC9HC-KRVrcTrvrxaqQ7q?usp=sharing).
Download `FoodMedKG_Survey_Responses-dos cuestionarios.xlsx` (the
original export, with the annotators' answers unchanged) and place it in
`data/survey_results/` (at the repository root). To reproduce the paper's
numbers, pass it explicitly:

```bash
python run_all.py "../../data/survey_results/FoodMedKG_Survey_Responses-dos cuestionarios.xlsx"
```

**Your own survey.** Export the Google Sheet via File > Download > Microsoft
Excel (.xlsx) and place it in `data/survey_results/`.

```bash
cd survey/analysis
pip install -r requirements.txt
python run_all.py                         # newest .xlsx in data/survey_results/
python run_all.py path/to/export.xlsx     # or an explicit file
```

`run_all.py` writes everything to one place:

- `REPORT.md` - all results in one file, with tables
- `results/*.csv` - one CSV per table

Each module can also be run on its own (e.g. `python aggregation.py`).
They split the analysis by what it measures:

| Module | Measures |
|---|---|
| `summary.py` | Majority-vote accuracy and the same figure under every alternative metric, with Wilson CIs |
| `descriptive.py` | Annotator demographics; % Agree / Not sure / Disagree per item and per relation type |
| `agreement.py` | Inter-annotator agreement: Fleiss' kappa (variable raters per item), Gwet's AC1, item-bootstrap 95% CIs |
| `aggregation.py` | Item-level accept/reject: majority vote with Not sure as abstention (headline), plurality, strict majority, Dawid–Skene |
| `annotator_reliability.py` | Leave-one-out agreement, whether Disagree votes cluster or scatter, disagreement by year/English level, robustness to annotator filters |
| `error_analysis.py` | Most-rejected relations, Not sure by health outcome, protective vs risk claims, oversimplified flags, vote-entropy difficulty |
| `matching.py` | Ingredient-matching judgments, food groups, automated vs human labels, similarity-score threshold |
| `disagreement.py` | Learning-from-disagreement aggregation (CrowdTruth 2.0, MACE), soft accuracy, per-relation soft labels |
| `consensus.py` | Consensus score and grade for every evaluated graph element (written to Neo4j by `data/kg_edits/add_consensus.py`) |
| `decision_confidence.py` | Per-relation doubt values (margin, sign test, entropy, bootstrap reject probability); clear vs contested verdicts; precision with Wilson CIs |

**Deduplication.** Some respondents submitted their block more than once
(page reloads, identical answers). `common.load()` keeps only the first
submission per (respondent, item) so nobody is counted twice. The report
states how many rows were removed.

The raw export and all generated outputs are git-ignored; the responses
used in the paper are distributed separately (link above).

## Notes on privacy

- No names or emails are collected. Each respondent gets a random UUID4
  (`respondent_id`) generated client-side at session start, used only to
  group that session's rows together.
- Demographic fields collected are: year of study, specialization,
  gender, age, and level of English - all from closed option lists except
  age (a plain number). No other identifying information is collected.
- After a successful submit, the app shows a thank-you screen and
  disables the form for that session (via `st.session_state`) to avoid the
  common case of an accidental duplicate submission on page reload. A
  respondent who deliberately opens a fresh session/tab can still submit
  again - this is accepted as out of scope per the study design.
