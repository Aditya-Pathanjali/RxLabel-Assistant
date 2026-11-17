# RxLabel Assistant: Grounded Q&A over FDA Drug Labels

RxLabel Assistant is a retrieval-augmented generation (RAG) assistant that answers questions about prescription
drugs using official **FDA drug labels** from [openFDA](https://open.fda.gov/apis/drug/label/),
built on **Databricks** with Delta Lake, Foundation Model APIs and MLflow.

> Example: *"Who should not take Eliquis?"* → a concise answer citing the Contraindications
> section of the Eliquis label, or a refusal if the drug is not in the database.

**Not medical advice.** Educational project on public FDA label data.

---

## Architecture

```
openFDA API ──► 01 Bronze (raw JSON, Delta)
                   │
                   ▼
               02 Silver (one row per drug × label section, cleaned)
                   │
                   ▼
               02 Gold (sentence-aware chunks + drug/section metadata, CDF enabled)
                   │
                   ▼
               03 Embeddings (databricks-gte-large-en) ─► optional Vector Search index
                   │
                   ▼
question ─► 04 detect drug ─► retrieve chunks for that drug ─► LLM answer with [1] citations
              (refuse if no known drug)                        (MLflow Tracing)
                   │
                   ▼
               05 MLflow evaluation: hit rate@k, MRR, refusal rate, groundedness, latency
```

**Data:** 38 drugs (13 Bristol Myers Squibb products such as Eliquis, Opdivo and Revlimid, plus
25 widely prescribed drugs), ~370 label sections and ~2,000 chunks.

## Files

| File | Purpose |
|---|---|
| `config.py` | Catalog/schema names, model endpoints, drug list, settings |
| `drug_rag_core.py` | Pure-Python logic: ingest, clean, chunk, retrieve, prompt, evaluate |
| `rag_setup.py` | Picks the best available embeddings / search / LLM, with fallbacks |
| `01_ingest_labels.py` … `05_evaluate_mlflow.py` | Databricks notebooks, run in order |
| `scripts/fetch_openfda_labels.py` | Local download, if Databricks compute has no internet |
| `tests/test_core.py` | Local tests (`python -m pytest -q`) |

## Run it on Databricks Free Edition (~30–45 minutes)

1. **Get the code into Databricks.** Push this folder to GitHub, then in Databricks:
   *Workspace → Create → Git folder* and paste the repo URL. (Alternative: *Workspace → Import*
   and upload all `.py` files into one folder.) The notebooks import `config.py`,
   `drug_rag_core.py` and `rag_setup.py` from the same folder.
2. **Check model endpoints.** Open *Serving* in the left menu. If the names differ from
   `EMBEDDING_ENDPOINT` / `CHAT_ENDPOINT` in `config.py`, edit `config.py`.
3. **Run `01_ingest_labels`** on serverless compute. If it reports that openFDA is not
   reachable, run `python scripts/fetch_openfda_labels.py` on your laptop and upload
   `data/drug_labels.jsonl` via *Catalog → workspace → drug_rag → raw → Upload to this volume*,
   then re-run 01.
4. **Run `02_clean_and_chunk`**, then **`03_embed_and_index`** (skip 03 if the embedding endpoint
   is unavailable; 04/05 then use TF-IDF automatically).
   Optional: set `USE_VECTOR_SEARCH = True` in `config.py` if *Vector Search* is available.
5. **Run `04_ask_questions`** to try the assistant (change the `question` widget at the top).
6. **Run `05_evaluate_mlflow`**, then open *Experiments → rxlabel-assistant-evaluation* to compare
   runs side by side.

## Design decisions

- **Medallion layout (Bronze → Silver → Gold)** keeps raw data, so cleaning can be changed and
  re-run without calling the API again.
- **Drug detection + metadata filter:** retrieval only searches chunks of the drug named in the
  question, which prevents answers mixing up similar drugs (for example two anticoagulants).
- **Guardrail:** if no known drug is mentioned, the assistant refuses instead of guessing.
- **Grounded prompt:** the model may only use the provided excerpts and must cite them.
- **Graceful fallbacks:** TF-IDF retrieval and an extractive answer if endpoints are unavailable,
  so the pipeline always runs end to end.

## Results

Fill this in from your MLflow runs (do not copy numbers you did not produce):

| Run | Hit rate@4 | MRR | Out-of-scope refusal | Groundedness (1–5) | Mean latency |
|---|---|---|---|---|---|
| tfidf_baseline | | | | – | – |
| gte_no_filter | | | | – | – |
| gte_with_filter | | | | | |

Local check (TF-IDF, no Databricks): hit rate@4 ≈ 0.6, out-of-scope refusal 5/5.

## Limitations

- Evaluation questions come from templates with a known answer section, so they measure
  retrieval on clear questions, not real user phrasing.
- The groundedness judge uses an LLM from the same family as the generator and can be lenient.
- One label per drug (latest prescription label); labels change over time.
