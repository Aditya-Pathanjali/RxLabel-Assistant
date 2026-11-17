# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Evaluate with MLflow
# MAGIC
# MAGIC Builds an evaluation set of template questions with a **known correct answer location**
# MAGIC (drug + label section), plus out-of-scope questions, then compares configurations:
# MAGIC
# MAGIC | Run | Embeddings | Drug filter |
# MAGIC |---|---|---|
# MAGIC | `tfidf_baseline` | TF-IDF | on |
# MAGIC | `gte_no_filter` | gte-large-en | off |
# MAGIC | `gte_with_filter` | gte-large-en | on (+ answer generation and groundedness judge) |
# MAGIC
# MAGIC **Metrics:** hit rate@k and MRR (did retrieval find the right drug section?), in-scope answer
# MAGIC rate, out-of-scope refusal rate, groundedness (LLM judge, 1–5) and latency.
# MAGIC
# MAGIC > Limitation: questions come from templates, so this measures retrieval on clear questions,
# MAGIC > not real user phrasing. The judge uses the same model family as the generator, which can be
# MAGIC > lenient. State both when you present results.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.getcwd())

import mlflow
import pandas as pd

import config as cfg
import drug_rag_core as core
import rag_setup

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/{cfg.MLFLOW_EXPERIMENT_NAME}")

sections = spark.table(cfg.SILVER_TABLE).toPandas()
eval_df = core.build_eval_set(sections, n_questions=40, seed=7)
spark.createDataFrame(eval_df).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{cfg.CATALOG}.{cfg.SCHEMA}.eval_questions")
print(f"{eval_df['in_scope'].sum()} in-scope + {(~eval_df['in_scope']).sum()} out-of-scope questions")

# COMMAND ----------

# MAGIC %md ## Retrieval runs

# COMMAND ----------

def run_retrieval_eval(run_name: str, embedding_mode: str, use_filter: bool) -> dict:
    rag = rag_setup.load_components(spark, embedding_mode=embedding_mode)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"embedding_mode": rag["mode"], "drug_filter": use_filter, "top_k": cfg.TOP_K,
                           "chunk_max_words": cfg.CHUNK_MAX_WORDS, "chunk_overlap_words": cfg.CHUNK_OVERLAP_WORDS,
                           "n_chunks": len(rag["chunks"]), "n_drugs": rag["chunks"]["brand_name"].nunique()})
        metrics, details = core.retrieval_metrics(eval_df, rag["retriever"], rag["embed"], rag["lookup"],
                                                  k=cfg.TOP_K, use_drug_filter=use_filter)
        mlflow.log_metrics(metrics)
        mlflow.log_table(details.astype(str), artifact_file="retrieval_details.json")
    print(run_name, metrics)
    return {"run": run_name, **metrics}


results = [run_retrieval_eval("tfidf_baseline", "tfidf", use_filter=True)]
try:
    results.append(run_retrieval_eval("gte_no_filter", "gte", use_filter=False))
    results.append(run_retrieval_eval("gte_with_filter", "gte", use_filter=True))
except RuntimeError as exc:
    print("Skipping gte runs:", exc)

display(pd.DataFrame(results))

# COMMAND ----------

# MAGIC %md ## Answer quality on a sample (generation + groundedness judge)

# COMMAND ----------

rag = rag_setup.load_components(spark)  # best available configuration
sample = eval_df[eval_df["in_scope"]].head(15)

if rag["chat"] is None:
    print("No chat endpoint available: skipping generation and judge.")
else:
    rows = []
    with mlflow.start_run(run_name=f"answers_{rag['mode']}"):
        for q in sample["question"]:
            res = core.answer_question(q, rag["retriever"], rag["embed"], rag["chat"], rag["lookup"], k=cfg.TOP_K)
            score, reason = core.judge_groundedness(res["answer"], res["contexts"], rag["chat"])
            rows.append({"question": q, "answer": res["answer"], "groundedness": score,
                         "judge_reason": reason, "latency_s": round(res["latency_s"], 2)})
        answers = pd.DataFrame(rows)
        mlflow.log_params({"embedding_mode": rag["mode"], "chat_endpoint": cfg.CHAT_ENDPOINT, "n_questions": len(answers)})
        mlflow.log_metrics({"mean_groundedness": float(answers["groundedness"].mean()),
                            "share_groundedness_4_plus": float((answers["groundedness"] >= 4).mean()),
                            "mean_latency_s": float(answers["latency_s"].mean()),
                            "p95_latency_s": float(answers["latency_s"].quantile(0.95))})
        mlflow.log_table(answers, artifact_file="answers.json")
    display(answers)
