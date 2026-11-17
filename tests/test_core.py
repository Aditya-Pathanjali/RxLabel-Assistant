"""Local tests for the pure-Python logic (run with: python -m pytest -q)."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import drug_rag_core as core  # noqa: E402

DATA = ROOT / "data" / "drug_labels.jsonl"


def test_clean_section_text_keeps_drug_name_and_drops_references():
    raw = "4 CONTRAINDICATIONS ELIQUIS is contraindicated in patients with: • Active bleeding [see Warnings ( 5.2 )]"
    assert core.clean_section_text(raw) == "ELIQUIS is contraindicated in patients with: - Active bleeding"


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} has five words." for i in range(200))
    chunks = core.chunk_text(text, max_words=50, overlap_words=10)
    assert len(chunks) > 1
    assert all(len(c.split()) <= 60 for c in chunks)
    assert chunks[0].split()[-6:] == chunks[1].split()[:6]  # last sentence carried over


def test_detect_drugs_handles_brand_generic_and_salt():
    chunks = pd.DataFrame({"brand_name": ["LIPITOR", "ENTRESTO"],
                           "generic_name": ["ATORVASTATIN CALCIUM", "SACUBITRIL AND VALSARTAN"]})
    lookup = core.drug_lookup(chunks)
    assert core.detect_drugs("Side effects of atorvastatin?", lookup) == ["LIPITOR"]
    assert core.detect_drugs("Dose of valsartan", lookup) == ["ENTRESTO"]
    assert core.detect_drugs("What is Lipitor for?", lookup) == ["LIPITOR"]
    assert core.detect_drugs("What is the capital of France?", lookup) == []


def test_retriever_filter_limits_results_to_drug():
    chunks = pd.DataFrame({"brand_name": ["A", "B", "A"], "chunk_text": ["x", "y", "z"]})
    emb = np.array([[1, 0], [1, 0.01], [0, 1]], dtype=np.float32)
    hits = core.InMemoryRetriever(chunks, emb).search(np.array([1, 0]), k=2, brands=["A"])
    assert hits["brand_name"].tolist() == ["A", "A"]
    assert hits["score"].iloc[0] == pytest.approx(1.0)


def test_databricks_factories_parse_responses(monkeypatch):
    class FakeClient:
        def predict(self, endpoint, inputs):
            if "input" in inputs:
                return {"data": [{"embedding": [0.1, 0.2]} for _ in inputs["input"]]}
            return {"choices": [{"message": {"content": " Answer [1] "}}]}

    fake = types.ModuleType("mlflow.deployments")
    fake.get_deploy_client = lambda target: FakeClient()
    monkeypatch.setitem(sys.modules, "mlflow", types.ModuleType("mlflow"))
    monkeypatch.setitem(sys.modules, "mlflow.deployments", fake)

    assert core.make_databricks_embedder("e", batch_size=2)(["a", "b", "c"]).shape == (3, 2)
    assert core.make_databricks_embedder("e", batch_size=3, max_workers=3)([str(i) for i in range(10)]).shape == (10, 2)
    assert core.make_databricks_chat("c")([{"role": "user", "content": "hi"}]) == "Answer [1]"


def test_embedder_keeps_order_and_retries_rate_limits(monkeypatch):
    calls = {"n": 0}

    class FlakyClient:
        def predict(self, endpoint, inputs):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("429 Too Many Requests")
            return {"data": [{"embedding": [float(t), 0.0]} for t in inputs["input"]]}

    fake = types.ModuleType("mlflow.deployments")
    fake.get_deploy_client = lambda target: FlakyClient()
    monkeypatch.setitem(sys.modules, "mlflow", types.ModuleType("mlflow"))
    monkeypatch.setitem(sys.modules, "mlflow.deployments", fake)
    monkeypatch.setattr(core.time, "sleep", lambda s: None)

    vecs = core.make_databricks_embedder("e", batch_size=4, max_workers=1)([str(i) for i in range(10)])
    assert vecs[:, 0].tolist() == list(range(10))  # order preserved across batches
    assert calls["n"] == 4  # 3 batches + 1 retry


def test_judge_parses_json_and_survives_garbage():
    ctx = pd.DataFrame({"chunk_text": ["evidence"]})
    assert core.judge_groundedness("a", ctx, lambda m: 'Sure: {"score": 4, "reason": "ok"}') == (4.0, "ok")
    assert core.judge_groundedness("a", ctx, lambda m: "no json")[0] is None


@pytest.mark.skipif(not DATA.exists(), reason="run scripts/fetch_openfda_labels.py first")
def test_end_to_end_on_real_labels_with_tfidf():
    labels = [json.loads(line) for line in DATA.open(encoding="utf-8")]
    sections = pd.DataFrame([row for raw in labels for row in core.parse_label(raw)])
    chunks = core.build_chunks(sections)
    embed = core.make_tfidf_embedder(chunks["embed_text"].tolist())
    retriever = core.InMemoryRetriever(chunks, embed(chunks["embed_text"].tolist()))
    lookup = core.drug_lookup(chunks)

    result = core.answer_question("Who should not take Eliquis?", retriever, embed, None, lookup)
    assert result["detected_drugs"] == ["ELIQUIS"]
    assert set(result["contexts"]["brand_name"]) == {"ELIQUIS"}

    refused = core.answer_question("Dose of amoxicillin?", retriever, embed, None, lookup)
    assert refused["refused"] is True

    metrics, _ = core.retrieval_metrics(core.build_eval_set(sections), retriever, embed, lookup)
    assert metrics["out_of_scope_refusal_rate"] == 1.0
    assert 0.0 <= metrics["hit_rate_at_4"] <= 1.0
