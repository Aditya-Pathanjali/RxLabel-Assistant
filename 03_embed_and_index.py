# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Embed chunks and (optionally) build a Vector Search index
# MAGIC
# MAGIC 1. Embeds every Gold chunk with the Databricks Foundation Model API endpoint
# MAGIC    `databricks-gte-large-en` and stores the vectors in a Delta table.
# MAGIC 2. **Optional:** if `USE_VECTOR_SEARCH = True` in `config.py`, creates a Databricks Vector
# MAGIC    Search endpoint and a Delta Sync index on that table (with a `brand_name` filter column).
# MAGIC
# MAGIC If the embedding endpoint is not available in your workspace, skip this notebook: notebooks
# MAGIC 04 and 05 automatically fall back to a TF-IDF retriever.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

# Fail fast instead of MLflow silently retrying a rate-limited request for minutes;
# our own retry loop (which prints what it is doing) handles rate limits instead.
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "60"

sys.path.append(os.getcwd())

import config as cfg
import drug_rag_core as core

chunks_pdf = spark.table(cfg.GOLD_TABLE).toPandas()
print(f"{len(chunks_pdf)} chunks to embed with {cfg.EMBEDDING_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Embed the chunks (resumable)
# MAGIC
# MAGIC Free Edition rate-limits model endpoints, so chunks are embedded in groups and each group is
# MAGIC **appended to the Delta table as soon as it is done**. If the cell is interrupted or times
# MAGIC out, just run it again: chunks that already have an embedding are skipped.
# MAGIC Set `RESTART = True` to rebuild the table from scratch.

# COMMAND ----------

import time

RESTART = False
BATCH_SIZE = 32
GROUP_SIZE = BATCH_SIZE * 4  # chunks saved per write

embed = core.make_databricks_embedder(cfg.EMBEDDING_ENDPOINT, batch_size=BATCH_SIZE, max_workers=2,
                                      max_retries=6, verbose=True)

# Quick check: one small call, so a wrong endpoint name fails in seconds.
t0 = time.perf_counter()
dim = embed(chunks_pdf["embed_text"].head(2).tolist()).shape[1]
print(f"Endpoint OK: one call took {time.perf_counter() - t0:.1f}s, dim={dim}")

if RESTART:
    spark.sql(f"DROP TABLE IF EXISTS {cfg.EMBEDDINGS_TABLE}")
done_ids = set()
if spark.catalog.tableExists(cfg.EMBEDDINGS_TABLE):
    done_ids = {r.chunk_id for r in spark.table(cfg.EMBEDDINGS_TABLE).select("chunk_id").collect()}
todo = chunks_pdf[~chunks_pdf["chunk_id"].isin(done_ids)].reset_index(drop=True)
print(f"{len(done_ids)} chunks already embedded, {len(todo)} to go")

t0 = time.perf_counter()
for start in range(0, len(todo), GROUP_SIZE):
    group = todo.iloc[start:start + GROUP_SIZE].copy()
    group["embedding"] = [v.tolist() for v in embed(group["embed_text"].tolist())]
    group["embedding_model"] = cfg.EMBEDDING_ENDPOINT
    (spark.createDataFrame(group)
        .write.mode("append")
        .option("delta.enableChangeDataFeed", "true")  # required for a Delta Sync index
        .saveAsTable(cfg.EMBEDDINGS_TABLE))
    done = start + len(group)
    rate = done / max(time.perf_counter() - t0, 1e-6)
    print(f"  saved {len(done_ids) + done}/{len(chunks_pdf)} chunks "
          f"({time.perf_counter() - t0:.0f}s elapsed, ~{(len(todo) - done) / rate / 60:.1f} min left)",
          flush=True)

total = spark.table(cfg.EMBEDDINGS_TABLE).count()
print(f"Done: {total}/{len(chunks_pdf)} chunks embedded, dim={dim}")

# COMMAND ----------

# MAGIC %md ## 2 · Optional: Databricks Vector Search index

# COMMAND ----------

if cfg.USE_VECTOR_SEARCH:
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient(disable_notice=True)
    existing = [e["name"] for e in vsc.list_endpoints().get("endpoints", [])]
    if cfg.VECTOR_SEARCH_ENDPOINT not in existing:
        vsc.create_endpoint_and_wait(name=cfg.VECTOR_SEARCH_ENDPOINT, endpoint_type="STANDARD")

    try:
        index = vsc.get_index(cfg.VECTOR_SEARCH_ENDPOINT, cfg.VECTOR_INDEX)
        index.sync()
    except Exception:
        index = vsc.create_delta_sync_index_and_wait(
            endpoint_name=cfg.VECTOR_SEARCH_ENDPOINT,
            index_name=cfg.VECTOR_INDEX,
            source_table_name=cfg.EMBEDDINGS_TABLE,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_dimension=int(dim),
            embedding_vector_column="embedding",
        )
    print("Vector Search index ready:", cfg.VECTOR_INDEX)
else:
    print("USE_VECTOR_SEARCH is False: notebooks 04/05 will search the embeddings table in memory.")
