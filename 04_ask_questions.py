# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Ask the drug-label assistant
# MAGIC
# MAGIC Flow for each question:
# MAGIC 1. **Detect the drug** (brand or generic name). If none of the loaded drugs is mentioned,
# MAGIC    refuse instead of guessing (guardrail).
# MAGIC 2. **Retrieve** the most similar label chunks for that drug only (metadata filter).
# MAGIC 3. **Generate** a concise answer with the Databricks chat endpoint, citing excerpts as [1], [2].
# MAGIC
# MAGIC Every call is traced with **MLflow Tracing**, so latency and inputs/outputs of each step are
# MAGIC visible in the MLflow UI.

# COMMAND ----------

# MAGIC %pip install -q databricks-vectorsearch

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

sys.path.append(os.getcwd())

import mlflow

import config as cfg
import drug_rag_core as core
import rag_setup

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/{cfg.MLFLOW_EXPERIMENT_NAME}")

rag = rag_setup.load_components(spark)

# COMMAND ----------

@mlflow.trace(name="rxlabel_assistant")
def ask(question: str) -> dict:
    result = core.answer_question(question, rag["retriever"], rag["embed"], rag["chat"],
                                  rag["lookup"], k=cfg.TOP_K)
    return {**result, "contexts": result["contexts"].to_dict("records")}


def show(question: str) -> None:
    result = ask(question)
    print("Q:", question)
    print("Detected drug:", ", ".join(result["detected_drugs"]) or "none")
    print("A:", result["answer"], "\n")
    for i, ctx in enumerate(result["contexts"], start=1):
        print(f"  [{i}] {ctx['brand_name']} - {ctx['section_title']} (score {ctx['score']:.3f})")
    print(f"  latency {result['latency_s']:.2f}s\n" + "-" * 100)

# COMMAND ----------

dbutils.widgets.text("question", "Who should not take Eliquis?", "Your question")

for q in [
    dbutils.widgets.get("question"),
    "What is the recommended starting dose of Revlimid for multiple myeloma?",
    "Does Opdivo have any serious immune-mediated adverse reactions?",
    "Which drugs interact with clopidogrel?",
    "What is the dose of amoxicillin for adults?",  # not in the database -> should refuse
]:
    show(q)
