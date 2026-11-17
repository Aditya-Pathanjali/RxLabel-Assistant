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

# Free Edition endpoints can hang on large requests. Cap MLflow's per-request and total
# timeouts (default total is 10 minutes) and turn off its silent retries: our own retry
# loop prints every wait.
os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "60"
os.environ["MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT"] = "60"
os.environ["MLFLOW_DEPLOYMENT_PREDICT_TOTAL_TIMEOUT"] = "90"

sys.path.append(os.getcwd())

import config as cfg
import drug_rag_core as core

chunks_pdf = spark.table(cfg.GOLD_TABLE).toPandas()
print(f"{len(chunks_pdf)} chunks to embed with {cfg.EMBEDDING_ENDPOINT}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Embed the chunks (resumable)
# MAGIC
# MAGIC Two ways to call the embedding model, chosen automatically:
# MAGIC * **`ai_query`** (Databricks AI Functions): batch inference in SQL, where Databricks handles
# MAGIC   batching and throttling. Preferred.
# MAGIC * **`api`**: small requests (8 chunks) sent one at a time from Python, with timeouts and retries.
# MAGIC
# MAGIC Chunks are embedded in groups and each group is **appended to the Delta table as soon as it is
# MAGIC done**. If the cell is interrupted or times out, run it again: finished chunks are skipped.
# MAGIC Set `EMBED_METHOD = "api"` to skip `ai_query`, or `RESTART = True` to rebuild from scratch.

# COMMAND ----------

import time

RESTART = False
EMBED_METHOD = "auto"  # "auto" | "ai_query" | "api"

api_embed = core.make_databricks_embedder(cfg.EMBEDDING_ENDPOINT, batch_size=8, max_workers=1,
                                          max_retries=6, request_timeout_s=90, verbose=True)


def embed_with_ai_query(texts: list[str]) -> list[list[float]]:
    """Batch inference through the ai_query SQL function; returns vectors in input order."""
    sdf = spark.createDataFrame([(i, t) for i, t in enumerate(texts)], "row_id INT, embed_text STRING")
    sdf.createOrReplaceTempView("rxlabel_to_embed")
    rows = spark.sql(f"""
        SELECT row_id, ai_query('{cfg.EMBEDDING_ENDPOINT}', embed_text) AS embedding
        FROM rxlabel_to_embed
    """).collect()
    by_id = {r.row_id: list(r.embedding) for r in rows}
    return [by_id[i] for i in range(len(texts))]


def embed_with_api(texts: list[str]) -> list[list[float]]:
    return api_embed(texts).tolist()


# Pick the method with a small probe (4 chunks), so problems show up in seconds.
probe = chunks_pdf["embed_text"].head(4).tolist()
embed_fn, dim = None, None
for method, fn in [("ai_query", embed_with_ai_query), ("api", embed_with_api)]:
    if EMBED_METHOD not in ("auto", method):
        continue
    t0 = time.perf_counter()
    try:
        dim = len(fn(probe)[0])
        embed_fn = fn
        print(f"Using {method}: probe of {len(probe)} chunks took {time.perf_counter() - t0:.1f}s, dim={dim}")
        break
    except Exception as exc:
        print(f"{method} not usable ({type(exc).__name__}: {str(exc)[:150]})")
if embed_fn is None:
    raise RuntimeError("No embedding method worked: skip this notebook; 04/05 will use TF-IDF.")

GROUP_SIZE = 256 if embed_fn is embed_with_ai_query else 64  # chunks saved per write

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
    group["embedding"] = embed_fn(group["embed_text"].tolist())
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
