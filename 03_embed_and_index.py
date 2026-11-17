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

sys.path.append(os.getcwd())

import config as cfg
import drug_rag_core as core

chunks_pdf = spark.table(cfg.GOLD_TABLE).toPandas()
print(f"{len(chunks_pdf)} chunks to embed with {cfg.EMBEDDING_ENDPOINT}")

# COMMAND ----------

# MAGIC %md ## 1 · Embed the chunks

# COMMAND ----------

import time

embed = core.make_databricks_embedder(cfg.EMBEDDING_ENDPOINT, batch_size=32, max_workers=4, progress=True)

# Quick check first: one small call, so a missing endpoint fails fast instead of after minutes.
start = time.perf_counter()
test_vec = embed(chunks_pdf["embed_text"].head(2).tolist())
print(f"Endpoint OK: one call took {time.perf_counter() - start:.1f}s, dim={test_vec.shape[1]}")
n_batches = -(-len(chunks_pdf) // 32)
print(f"Embedding {len(chunks_pdf)} chunks in {n_batches} batches (4 in parallel)...")

start = time.perf_counter()
vectors = embed(chunks_pdf["embed_text"].tolist())
print(f"Embedded {len(vectors)} chunks, dim={vectors.shape[1]}, in {time.perf_counter() - start:.0f}s")

embeddings_pdf = chunks_pdf.copy()
embeddings_pdf["embedding"] = [v.tolist() for v in vectors]
embeddings_pdf["embedding_model"] = cfg.EMBEDDING_ENDPOINT

(spark.createDataFrame(embeddings_pdf)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")  # required for a Delta Sync index
    .saveAsTable(cfg.EMBEDDINGS_TABLE))

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
            embedding_dimension=int(vectors.shape[1]),
            embedding_vector_column="embedding",
        )
    print("Vector Search index ready:", cfg.VECTOR_INDEX)
else:
    print("USE_VECTOR_SEARCH is False: notebooks 04/05 will search the embeddings table in memory.")
