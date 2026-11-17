# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Clean sections (Silver) and chunk them (Gold)
# MAGIC
# MAGIC * **Silver:** one row per drug × label section (indications, dosage, contraindications,
# MAGIC   warnings, adverse reactions, interactions, …) with headings, cross-references and odd
# MAGIC   bullet characters removed.
# MAGIC * **Gold:** sentence-aware chunks (~220 words, 40-word overlap) with drug and section
# MAGIC   metadata, ready for embedding. Change Data Feed is enabled so the table can feed a
# MAGIC   Vector Search index.

# COMMAND ----------

import json
import os
import sys

sys.path.append(os.getcwd())

import pandas as pd

import config as cfg
import drug_rag_core as core

# COMMAND ----------

raw_labels = []
for row in spark.table(cfg.BRONZE_TABLE).select("query_brand", "raw_json").collect():
    raw = json.loads(row.raw_json)
    raw["_query_brand"] = row.query_brand
    raw_labels.append(raw)

sections_pdf = pd.DataFrame([row for raw in raw_labels for row in core.parse_label(raw)])
print(f"{len(sections_pdf)} sections from {sections_pdf['brand_name'].nunique()} drugs")

(spark.createDataFrame(sections_pdf)
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(cfg.SILVER_TABLE))

display(spark.sql(f"""
    SELECT section_title, COUNT(*) AS drugs, ROUND(AVG(SIZE(SPLIT(section_text, ' ')))) AS avg_words
    FROM {cfg.SILVER_TABLE} GROUP BY section_title ORDER BY drugs DESC
"""))

# COMMAND ----------

chunks_pdf = core.build_chunks(sections_pdf, cfg.CHUNK_MAX_WORDS, cfg.CHUNK_OVERLAP_WORDS)
print(f"{len(chunks_pdf)} chunks")

(spark.createDataFrame(chunks_pdf)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true")
    .saveAsTable(cfg.GOLD_TABLE))

display(spark.table(cfg.GOLD_TABLE).limit(10))
