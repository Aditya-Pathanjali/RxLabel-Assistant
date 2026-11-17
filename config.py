"""Project configuration shared by all notebooks.

Edit these values if your Databricks workspace uses different names or
does not offer some of the optional features.
"""

# --- Unity Catalog locations (Free Edition gives you the `workspace` catalog) ---
CATALOG = "workspace"
SCHEMA = "drug_rag"
VOLUME = "raw"  # Unity Catalog volume for the uploaded fallback data file

BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.labels_bronze"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.label_sections_silver"
GOLD_TABLE = f"{CATALOG}.{SCHEMA}.label_chunks_gold"
EMBEDDINGS_TABLE = f"{CATALOG}.{SCHEMA}.label_chunk_embeddings"
FALLBACK_FILE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/drug_labels.jsonl"

# --- Databricks Foundation Model API endpoints (check "Serving" in your workspace) ---
EMBEDDING_ENDPOINT = "databricks-gte-large-en"
CHAT_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# --- Optional Databricks Vector Search ---
USE_VECTOR_SEARCH = False  # set True only if Vector Search is available in your workspace
VECTOR_SEARCH_ENDPOINT = "drug-rag-vs-endpoint"
VECTOR_INDEX = f"{CATALOG}.{SCHEMA}.label_chunks_index"

# --- Retrieval / generation settings ---
TOP_K = 4
CHUNK_MAX_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
MLFLOW_EXPERIMENT_NAME = "rxlabel-assistant-evaluation"

# --- Drugs to load from openFDA (brand names) ---
# First block: Bristol Myers Squibb products. Second block: widely prescribed drugs.
DRUG_BRANDS = [
    "ELIQUIS", "OPDIVO", "YERVOY", "REVLIMID", "POMALYST", "SPRYCEL", "ORENCIA",
    "ZEPOSIA", "CAMZYOS", "SOTYKTU", "REBLOZYL", "KRAZATI", "COBENFY",
    "XARELTO", "JARDIANCE", "FARXIGA", "OZEMPIC", "MOUNJARO", "JANUVIA", "ENTRESTO",
    "LIPITOR", "CRESTOR", "PLAVIX", "BRILINTA", "SYNTHROID", "ZOLOFT", "LEXAPRO",
    "KEYTRUDA", "HUMIRA", "STELARA", "DUPIXENT", "BIKTARVY", "PAXLOVID", "LYRICA",
    "SINGULAIR", "NEXIUM", "LANTUS", "TRULICITY",
]
