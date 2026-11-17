"""Builds the retriever, embedder and chat model for notebooks 04 and 05.

Picks the best option available in the workspace and falls back gracefully:
  embeddings: Databricks gte-large-en  ->  TF-IDF fitted on the chunks
  search:     Vector Search index      ->  in-memory cosine similarity
  generation: Databricks chat endpoint ->  extractive answer (top excerpt)
"""

from __future__ import annotations

import os

import numpy as np

# Free Edition endpoints can hang; cap MLflow's own timeouts (default total: 10 minutes)
# and disable its silent retries; drug_rag_core retries with visible logging instead.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")
os.environ.setdefault("MLFLOW_DEPLOYMENT_PREDICT_TIMEOUT", "60")
os.environ.setdefault("MLFLOW_DEPLOYMENT_PREDICT_TOTAL_TIMEOUT", "90")

import config as cfg
import drug_rag_core as core


def _endpoint_works(fn, *args) -> bool:
    try:
        fn(*args)
        return True
    except Exception as exc:  # endpoint missing, no permission, quota, ...
        print(f"  unavailable: {type(exc).__name__}: {str(exc)[:150]}")
        return False


def load_components(spark, embedding_mode: str = "auto", use_vector_search: bool | None = None) -> dict:
    """embedding_mode: 'auto' | 'gte' | 'tfidf'."""
    chunks = spark.table(cfg.GOLD_TABLE).toPandas()
    lookup = core.drug_lookup(chunks)
    use_vs = cfg.USE_VECTOR_SEARCH if use_vector_search is None else use_vector_search

    mode, embed, retriever = "tfidf", None, None
    if embedding_mode in ("auto", "gte") and spark.catalog.tableExists(cfg.EMBEDDINGS_TABLE):
        print(f"Checking embedding endpoint {cfg.EMBEDDING_ENDPOINT} ...")
        gte = core.make_databricks_embedder(cfg.EMBEDDING_ENDPOINT, batch_size=8, max_workers=1,
                                            request_timeout_s=60)
        if _endpoint_works(gte, ["test"]):
            mode, embed = "gte", gte
            if use_vs:
                retriever = core.VectorSearchRetriever(cfg.VECTOR_SEARCH_ENDPOINT, cfg.VECTOR_INDEX)
                mode = "gte+vector_search"
            else:
                emb = spark.table(cfg.EMBEDDINGS_TABLE).select("chunk_id", "embedding").toPandas()
                chunks = chunks.merge(emb, on="chunk_id")
                retriever = core.InMemoryRetriever(chunks.drop(columns="embedding"),
                                                   np.vstack(chunks["embedding"].to_numpy()))
    if embedding_mode == "gte" and embed is None:
        raise RuntimeError("gte embeddings requested but not available: run notebook 03 first.")

    if embed is None:  # TF-IDF fallback
        embed = core.make_tfidf_embedder(chunks["embed_text"].tolist())
        retriever = core.InMemoryRetriever(chunks, embed(chunks["embed_text"].tolist()))

    print(f"Checking chat endpoint {cfg.CHAT_ENDPOINT} ...")
    chat = core.make_databricks_chat(cfg.CHAT_ENDPOINT)
    if not _endpoint_works(chat, [{"role": "user", "content": "Reply with OK."}]):
        chat = None  # answer_question() then returns the top excerpt instead

    print(f"Retrieval mode: {mode} | generation: {'LLM ' + cfg.CHAT_ENDPOINT if chat else 'extractive fallback'}")
    return {"chunks": chunks, "lookup": lookup, "embed": embed, "retriever": retriever,
            "chat": chat, "mode": mode}
