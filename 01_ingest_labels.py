# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest FDA drug labels (Bronze)
# MAGIC
# MAGIC Pulls the latest prescription label for each drug in `config.DRUG_BRANDS` from the
# MAGIC [openFDA drug label API](https://open.fda.gov/apis/drug/label/) and stores the raw JSON in a
# MAGIC **Bronze Delta table**.
# MAGIC
# MAGIC **If your compute cannot reach the internet** (Free Edition can restrict outbound access), the
# MAGIC notebook falls back to `drug_labels.jsonl` uploaded to the Unity Catalog volume (see README).

# COMMAND ----------

import json
import os
import sys

sys.path.append(os.getcwd())  # import the project modules from this folder

import config as cfg
import drug_rag_core as core

# COMMAND ----------

# MAGIC %md ## Create the schema and the volume for the fallback file

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.CATALOG}.{cfg.SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.CATALOG}.{cfg.SCHEMA}.{cfg.VOLUME}")
print("Fallback upload location:", cfg.FALLBACK_FILE)

# COMMAND ----------

# MAGIC %md ## Fetch from openFDA, or read the uploaded file

# COMMAND ----------

labels, source = [], ""
try:
    labels, missing = core.fetch_labels(cfg.DRUG_BRANDS)
    source = "openfda_api"
    if missing:
        print("No exact label found for:", ", ".join(missing))
except Exception as exc:  # e.g. no outbound internet from serverless compute
    print(f"openFDA API not reachable ({type(exc).__name__}): using the uploaded file instead.")
    with open(cfg.FALLBACK_FILE, encoding="utf-8") as fh:
        labels = [json.loads(line) for line in fh if line.strip()]
    source = "volume_upload"

print(f"Loaded {len(labels)} labels from {source}")

# COMMAND ----------

# MAGIC %md ## Write the Bronze Delta table (raw JSON + lineage columns)

# COMMAND ----------

from pyspark.sql import functions as F

bronze_rows = [
    {
        "query_brand": rec.get("_query_brand", ""),
        "set_id": rec.get("set_id", ""),
        "label_id": rec.get("id", ""),
        "effective_time": rec.get("effective_time", ""),
        "raw_json": json.dumps(rec, ensure_ascii=False),
        "source": source,
    }
    for rec in labels
]

bronze_df = spark.createDataFrame(bronze_rows).withColumn("ingested_at", F.current_timestamp())
bronze_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.BRONZE_TABLE)
display(spark.table(cfg.BRONZE_TABLE).select("query_brand", "set_id", "effective_time", "source", "ingested_at"))
