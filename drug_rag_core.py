"""Core logic for the drug-label RAG assistant.

Pure Python (requests, numpy, pandas) so it runs both inside Databricks
notebooks and locally for testing. Databricks-specific calls (model serving)
are isolated in the `make_databricks_*` factory functions.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

OPENFDA_LABEL_URL = "https://api.fda.gov/drug/label.json"

# Common repackager/relabeler names in openFDA; their labels copy the original manufacturer's.
REPACKAGER_HINTS = (
    "REPACK", "A-S MEDICATION", "APHENA", "PROFICIENT RX", "NUCARE", "BRYANT RANCH",
    "QUALITY CARE", "PD-RX", "DIRECT_RX", "DIRECT RX", "PREFERRED PHARMACEUTICALS",
    "HJ HARKINS", "HENRY SCHEIN", "CARDINAL HEALTH", "MEDVANTX", "LAKE ERIE MEDICAL",
    "RPK PHARMACEUTICALS", "ADVANCED RX", "NORTHWIND", "ASCLEMED", "PACKAGING",
)

# (openFDA field, human-readable section title), in label order.
SECTION_FIELDS: list[tuple[str, str]] = [
    ("boxed_warning", "Boxed Warning"),
    ("indications_and_usage", "Indications and Usage"),
    ("dosage_and_administration", "Dosage and Administration"),
    ("dosage_forms_and_strengths", "Dosage Forms and Strengths"),
    ("contraindications", "Contraindications"),
    ("warnings_and_cautions", "Warnings and Precautions"),
    ("warnings", "Warnings"),
    ("adverse_reactions", "Adverse Reactions"),
    ("drug_interactions", "Drug Interactions"),
    ("use_in_specific_populations", "Use in Specific Populations"),
    ("overdosage", "Overdosage"),
    ("mechanism_of_action", "Mechanism of Action"),
]

EmbedFn = Callable[[Sequence[str]], np.ndarray]
ChatFn = Callable[[list[dict]], str]


# ---------------------------------------------------------------------------
# 1. Ingestion
# ---------------------------------------------------------------------------

def fetch_label(brand: str, timeout: float = 30.0, session=None) -> dict | None:
    """Fetch the most recent openFDA prescription label whose brand name matches exactly."""
    import requests

    http = session or requests
    params = {"search": f'openfda.brand_name:"{brand}"', "limit": 100}
    resp = http.get(OPENFDA_LABEL_URL, params=params, timeout=timeout)
    if resp.status_code == 404:  # openFDA returns 404 when nothing matches
        return None
    resp.raise_for_status()
    results = resp.json().get("results", [])

    def is_match(rec: dict) -> bool:
        names = [n.upper() for n in rec.get("openfda", {}).get("brand_name", [])]
        return brand.upper() in names

    def is_rx(rec: dict) -> bool:
        return "HUMAN PRESCRIPTION DRUG" in rec.get("openfda", {}).get("product_type", [])

    def is_original(rec: dict) -> bool:
        """Prefer the manufacturer's own label over repackager/relabeler copies."""
        maker = " ".join(rec.get("openfda", {}).get("manufacturer_name", [])).upper()
        return not any(word in maker for word in REPACKAGER_HINTS)

    candidates = [r for r in results if is_match(r)]
    if not candidates:
        return None
    def single_brand(rec: dict) -> bool:  # avoid records that bundle several brands
        return len(set(n.upper() for n in rec.get("openfda", {}).get("brand_name", []))) == 1

    candidates.sort(
        key=lambda r: (is_rx(r), single_brand(r), is_original(r), r.get("effective_time", "")),
        reverse=True,
    )
    return candidates[0]


def fetch_labels(brands: Iterable[str], pause_s: float = 0.3) -> tuple[list[dict], list[str]]:
    """Fetch labels for many brands. Returns (labels, brands_not_found)."""
    import requests

    labels, missing = [], []
    with requests.Session() as session:
        for brand in brands:
            rec = fetch_label(brand, session=session)
            if rec is None:
                missing.append(brand)
            else:
                rec["_query_brand"] = brand.upper()
                labels.append(rec)
            time.sleep(pause_s)  # stay well inside openFDA's anonymous rate limit
    return labels, missing


# ---------------------------------------------------------------------------
# 2. Cleaning and chunking
# ---------------------------------------------------------------------------

_BULLETS = re.compile(r"[•●▪◦‣⁃·�]")
_SEE_REF = re.compile(r"\[\s*see[^\]]*\]", re.IGNORECASE)
_NUM_REF = re.compile(r"\(\s*\d+(?:\.\d+)*(?:\s*,\s*\d+(?:\.\d+)*)*\s*\)")
_SECTION_NUMBER = re.compile(r"^\s*\d+(?:\.\d+)*\s+")
# Words that make up standard label section headings ("4 CONTRAINDICATIONS", "5 WARNINGS AND PRECAUTIONS").
_HEADING_WORDS = {
    "INDICATIONS", "AND", "&", "USAGE", "DOSAGE", "ADMINISTRATION", "FORMS", "STRENGTHS",
    "CONTRAINDICATIONS", "WARNINGS", "PRECAUTIONS", "ADVERSE", "REACTIONS", "DRUG",
    "INTERACTIONS", "USE", "IN", "SPECIFIC", "POPULATIONS", "OVERDOSAGE", "MECHANISM", "OF", "ACTION",
}


def strip_heading(text: str) -> str:
    """Remove a leading section number and heading words, but never the drug name after them."""
    text = _SECTION_NUMBER.sub("", text)
    words = text.split(" ")
    i = 0
    while i < len(words) and words[i] in _HEADING_WORDS:
        i += 1
    return " ".join(words[i:]) if i else text


def clean_section_text(text: str) -> str:
    """Normalise one label section: drop numbering, cross-references and odd bullets."""
    text = _BULLETS.sub(" - ", text)
    text = _SEE_REF.sub("", text)
    text = _NUM_REF.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return strip_heading(text).strip()


def first_str(rec: dict, key: str, default: str = "") -> str:
    vals = rec.get("openfda", {}).get(key, [])
    return vals[0] if vals else default


def parse_label(raw: dict) -> list[dict]:
    """Turn one raw openFDA label into one row per available section."""
    rows = []
    has_new_warnings = bool(raw.get("warnings_and_cautions"))
    for key, title in SECTION_FIELDS:
        if key == "warnings" and has_new_warnings:
            continue  # older labels use "warnings"; newer ones "warnings_and_cautions"
        parts = raw.get(key) or []
        text = clean_section_text(" ".join(parts))
        if len(text.split()) < 5:
            continue
        rows.append({
            "set_id": raw.get("set_id", ""),
            "label_id": raw.get("id", ""),
            "effective_time": raw.get("effective_time", ""),
            "brand_name": raw.get("_query_brand") or first_str(raw, "brand_name").upper(),
            "generic_name": first_str(raw, "generic_name").upper(),
            "manufacturer": first_str(raw, "manufacturer_name"),
            "section_key": key,
            "section_title": title,
            "section_text": text,
        })
    return rows


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?;])\s+(?=[A-Z0-9\-])", text) if s.strip()]


def chunk_text(text: str, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    """Sentence-aware chunking with a word overlap between consecutive chunks."""
    sentences: list[str] = []
    for s in split_sentences(text):  # hard-split any very long "sentence" (e.g. tables)
        words = s.split()
        for i in range(0, len(words), max_words):
            sentences.append(" ".join(words[i:i + max_words]))

    chunks, current = [], []
    for sent in sentences:
        if current and len(" ".join(current + [sent]).split()) > max_words:
            chunks.append(" ".join(current))
            tail, count = [], 0
            for prev in reversed(current):  # carry the last sentences forward as overlap
                count += len(prev.split())
                if count > overlap_words:
                    break
                tail.insert(0, prev)
            current = tail
        current.append(sent)
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunks(sections: pd.DataFrame, max_words: int = 220, overlap_words: int = 40) -> pd.DataFrame:
    rows = []
    for rec in sections.to_dict("records"):
        for i, chunk in enumerate(chunk_text(rec["section_text"], max_words, overlap_words)):
            header = f"{rec['brand_name']} ({rec['generic_name']}) - {rec['section_title']}"
            rows.append({
                "chunk_id": f"{rec['set_id']}::{rec['section_key']}::{i}",
                "set_id": rec["set_id"],
                "brand_name": rec["brand_name"],
                "generic_name": rec["generic_name"],
                "section_key": rec["section_key"],
                "section_title": rec["section_title"],
                "chunk_index": i,
                "chunk_text": chunk,
                "embed_text": f"{header}: {chunk}",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Embeddings and retrieval
# ---------------------------------------------------------------------------

def make_databricks_embedder(endpoint: str, batch_size: int = 16) -> EmbedFn:
    """Embeddings from a Databricks Foundation Model API endpoint."""
    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")

    def embed(texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for i in range(0, len(texts), batch_size):
            resp = client.predict(endpoint=endpoint, inputs={"input": list(texts[i:i + batch_size])})
            vectors.extend(item["embedding"] for item in resp["data"])
        return np.asarray(vectors, dtype=np.float32)

    return embed


def make_tfidf_embedder(corpus: Sequence[str]) -> EmbedFn:
    """Fallback embedder (no model endpoint needed): TF-IDF fitted on the chunk corpus."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, stop_words="english")
    vectorizer.fit(corpus)

    def embed(texts: Sequence[str]) -> np.ndarray:
        return vectorizer.transform(list(texts)).toarray().astype(np.float32)

    return embed


def normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


@dataclass
class InMemoryRetriever:
    """Cosine-similarity search over chunk embeddings, with optional drug filter."""

    chunks: pd.DataFrame
    embeddings: np.ndarray
    _unit: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.chunks = self.chunks.reset_index(drop=True)
        self._unit = normalise(np.asarray(self.embeddings, dtype=np.float32))

    def search(self, query_vec: np.ndarray, k: int = 4, brands: Sequence[str] | None = None) -> pd.DataFrame:
        scores = self._unit @ normalise(query_vec.reshape(1, -1))[0]
        mask = np.ones(len(scores), dtype=bool)
        if brands:
            mask = self.chunks["brand_name"].isin([b.upper() for b in brands]).to_numpy()
        idx = np.where(mask)[0]
        top = idx[np.argsort(-scores[idx])[:k]]
        out = self.chunks.iloc[top].copy()
        out["score"] = scores[top]
        return out.reset_index(drop=True)


_SALT_WORDS = re.compile(
    r"\s+(HYDROCHLORIDE|HCL|SODIUM|POTASSIUM|CALCIUM|MAGNESIUM|CHLORIDE|FUMARATE|"
    r"MALEATE|MESYLATE|SUCCINATE|TARTRATE|BESYLATE|ALAFENAMIDE|ALAFENAMIDE FUMARATE)\b.*$"
)


def common_generic_name(generic: str) -> str:
    """'ATORVASTATIN CALCIUM' -> 'ATORVASTATIN' (the name people actually use)."""
    return _SALT_WORDS.sub("", generic.strip().upper())


class VectorSearchRetriever:
    """Same interface as InMemoryRetriever, backed by a Databricks Vector Search index."""

    COLUMNS = ["chunk_id", "brand_name", "generic_name", "section_key", "section_title", "chunk_text"]

    def __init__(self, endpoint_name: str, index_name: str):
        from databricks.vector_search.client import VectorSearchClient

        self.index = VectorSearchClient(disable_notice=True).get_index(endpoint_name, index_name)

    def search(self, query_vec: np.ndarray, k: int = 4, brands: Sequence[str] | None = None) -> pd.DataFrame:
        kwargs = {"query_vector": [float(x) for x in query_vec], "columns": self.COLUMNS, "num_results": k}
        if brands:
            kwargs["filters"] = {"brand_name": [b.upper() for b in brands]}
        result = self.index.similarity_search(**kwargs)["result"]
        cols = self.COLUMNS + ["score"]
        return pd.DataFrame(result.get("data_array", []), columns=cols)


def drug_lookup(chunks: pd.DataFrame) -> dict[str, str]:
    """Map every brand and generic name (upper case, with and without salt) to its brand name."""
    lookup = {}
    for brand, generic in chunks[["brand_name", "generic_name"]].drop_duplicates().itertuples(index=False):
        lookup[brand.upper()] = brand
        for part in re.split(r",?\s+AND\s+|,|/", generic.upper()):  # combination products
            if part.strip():
                lookup[part.strip()] = brand
                lookup[common_generic_name(part)] = brand
    return lookup


def detect_drugs(question: str, lookup: dict[str, str]) -> list[str]:
    """Return brand names whose brand or generic name appears in the question."""
    found = []
    q = question.upper()
    for name, brand in lookup.items():
        if re.search(rf"(?<![A-Z]){re.escape(name)}(?![A-Z])", q) and brand not in found:
            found.append(brand)
    return found


# ---------------------------------------------------------------------------
# 4. Generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful pharmaceutical information assistant. Answer ONLY from the FDA "
    "drug label excerpts provided. Cite the excerpts you use as [1], [2], etc. If the "
    "excerpts do not contain the answer, reply exactly: 'I could not find this in the "
    "provided FDA labels.' Keep answers concise and factual. This is not medical advice."
)

REFUSAL_NO_DRUG = (
    "I can only answer questions about the drugs in this label database, and I could "
    "not find one of them in your question."
)


def build_messages(question: str, contexts: pd.DataFrame) -> list[dict]:
    excerpts = "\n\n".join(
        f"[{i}] {row.brand_name} ({row.generic_name}) - {row.section_title}:\n{row.chunk_text}"
        for i, row in enumerate(contexts.itertuples(index=False), start=1)
    )
    user = f"FDA label excerpts:\n\n{excerpts}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def make_databricks_chat(endpoint: str, max_tokens: int = 400, temperature: float = 0.0) -> ChatFn:
    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")

    def chat(messages: list[dict]) -> str:
        resp = client.predict(
            endpoint=endpoint,
            inputs={"messages": messages, "max_tokens": max_tokens, "temperature": temperature},
        )
        return resp["choices"][0]["message"]["content"].strip()

    return chat


def extractive_chat(messages: list[dict]) -> str:
    """Fallback 'generator' when no LLM endpoint is available: returns the top excerpt."""
    user = messages[-1]["content"]
    match = re.search(r"\[1\][^\n]*\n(.+?)(?:\n\n\[2\]|\n\nQuestion:)", user, re.DOTALL)
    excerpt = match.group(1).strip() if match else ""
    return f"Most relevant label excerpt [1]: {excerpt[:700]}"


def answer_question(
    question: str,
    retriever: InMemoryRetriever,
    embed: EmbedFn,
    chat: ChatFn | None,
    lookup: dict[str, str],
    k: int = 4,
    use_drug_filter: bool = True,
) -> dict:
    """Full RAG step: detect drug -> retrieve -> generate (or refuse)."""
    start = time.perf_counter()
    brands = detect_drugs(question, lookup)
    if use_drug_filter and not brands:
        return {"question": question, "answer": REFUSAL_NO_DRUG, "refused": True,
                "detected_drugs": [], "contexts": pd.DataFrame(),
                "latency_s": time.perf_counter() - start}

    contexts = retriever.search(embed([question])[0], k=k, brands=brands if use_drug_filter else None)
    answer = (chat or extractive_chat)(build_messages(question, contexts))
    return {"question": question, "answer": answer, "refused": False, "detected_drugs": brands,
            "contexts": contexts, "latency_s": time.perf_counter() - start}


# ---------------------------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------------------------

QUESTION_TEMPLATES: dict[str, list[str]] = {
    "indications_and_usage": ["What is {drug} used to treat?", "What conditions is {drug} approved for?"],
    "dosage_and_administration": ["What is the recommended dose of {drug}?", "How should {drug} be taken?"],
    "contraindications": ["Who should not take {drug}?", "What are the contraindications for {drug}?"],
    "adverse_reactions": ["What are the most common side effects of {drug}?"],
    "drug_interactions": ["Which drugs interact with {drug}?"],
    "boxed_warning": ["Does {drug} have a boxed warning, and what does it say?"],
}

OUT_OF_SCOPE_QUESTIONS = [
    "What is the recommended dose of amoxicillin?",
    "What are the side effects of ibuprofen?",
    "Who should not take metoprolol?",
    "What is the capital of France?",
    "How do I reset my email password?",
]


def build_eval_set(sections: pd.DataFrame, n_questions: int = 40, seed: int = 7) -> pd.DataFrame:
    """Template questions with known gold (drug, section), plus out-of-scope questions."""
    rng = random.Random(seed)
    pool = []
    for rec in sections[["brand_name", "generic_name", "section_key"]].drop_duplicates().to_dict("records"):
        for template in QUESTION_TEMPLATES.get(rec["section_key"], []):
            # Ask by brand or generic name to test that both are recognised.
            first_ingredient = re.split(r",?\s+AND\s+|,", rec["generic_name"])[0]
            generic = common_generic_name(first_ingredient).lower()
            drug = rng.choice([rec["brand_name"].title(), generic])
            pool.append({"question": template.format(drug=drug), "gold_brand": rec["brand_name"],
                         "gold_section": rec["section_key"], "in_scope": True})
    rng.shuffle(pool)
    rows = pool[:n_questions] + [
        {"question": q, "gold_brand": None, "gold_section": None, "in_scope": False}
        for q in OUT_OF_SCOPE_QUESTIONS
    ]
    return pd.DataFrame(rows)


def retrieval_metrics(eval_df: pd.DataFrame, retriever: InMemoryRetriever, embed: EmbedFn,
                      lookup: dict[str, str], k: int = 4, use_drug_filter: bool = True) -> tuple[dict, pd.DataFrame]:
    """Hit rate@k and MRR on in-scope questions; refusal accuracy on all questions."""
    records = []
    for row in eval_df.to_dict("records"):
        brands = detect_drugs(row["question"], lookup)
        refused = use_drug_filter and not brands
        rank = None
        if not refused:
            hits = retriever.search(embed([row["question"]])[0], k=k, brands=brands if use_drug_filter else None)
            if row["in_scope"]:
                for i, h in enumerate(hits.itertuples(index=False), start=1):
                    if h.brand_name == row["gold_brand"] and h.section_key == row["gold_section"]:
                        rank = i
                        break
        records.append({**row, "detected_drugs": ", ".join(brands), "refused": refused, "gold_rank": rank})

    res = pd.DataFrame(records)
    ins = res[res["in_scope"]]
    oos = res[~res["in_scope"]]
    metrics = {
        f"hit_rate_at_{k}": float(ins["gold_rank"].notna().mean()),
        "mrr": float(ins["gold_rank"].apply(lambda r: 1.0 / r if pd.notna(r) else 0.0).mean()),
        "in_scope_answer_rate": float((~ins["refused"]).mean()),
        "out_of_scope_refusal_rate": float(oos["refused"].mean()) if len(oos) else float("nan"),
    }
    return metrics, res


JUDGE_PROMPT = (
    "You are grading a RAG answer. Given the excerpts and the answer, rate GROUNDEDNESS from 1 "
    "to 5: 5 = every claim is supported by the excerpts, 1 = mostly unsupported. Reply with JSON "
    'only: {{"score": <1-5>, "reason": "<one sentence>"}}\n\nExcerpts:\n{excerpts}\n\nAnswer:\n{answer}'
)


def judge_groundedness(answer: str, contexts: pd.DataFrame, chat: ChatFn) -> tuple[float | None, str]:
    excerpts = "\n".join(f"- {t}" for t in contexts["chunk_text"].tolist())
    raw = chat([{"role": "user", "content": JUDGE_PROMPT.format(excerpts=excerpts, answer=answer)}])
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        parsed = json.loads(match.group(0)) if match else {}
        return float(parsed.get("score")), str(parsed.get("reason", ""))
    except (ValueError, TypeError):
        return None, raw[:200]
