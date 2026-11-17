"""Run notebooks 04 (ask questions) and 05 (MLflow evaluation) from a laptop.

Use this when Databricks serverless notebook compute is unavailable (e.g. Free Edition
capacity limits). Everything else still happens on Databricks:
  * Delta tables are read through a Databricks SQL warehouse (Statement Execution API)
  * embeddings and answers come from Databricks Foundation Model API endpoints
  * runs, metrics, tables and traces are logged to the Databricks MLflow experiment

Prerequisites: `databricks auth login` done, the Databricks CLI on PATH, and
    pip install mlflow-skinny databricks-sdk requests numpy pandas scikit-learn
Usage (from the project folder):
    python scripts/run_from_laptop.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
from databricks.sdk import WorkspaceClient  # noqa: E402
from databricks.sdk.service.sql import Disposition, Format, StatementState  # noqa: E402

import config as cfg  # noqa: E402
import drug_rag_core as core  # noqa: E402
import rag_setup  # noqa: E402

OUT_DIR = ROOT / "outputs"


# ---------------------------------------------------------------------------
# Minimal Spark stand-in backed by a SQL warehouse (only what rag_setup needs)
# ---------------------------------------------------------------------------

class WarehouseSession:
    def __init__(self, client: WorkspaceClient, warehouse_id: str):
        self.w, self.warehouse_id = client, warehouse_id
        self.catalog = types.SimpleNamespace(tableExists=self._table_exists)

    def query(self, statement: str) -> pd.DataFrame:
        sx = self.w.statement_execution
        resp = sx.execute_statement(statement=statement, warehouse_id=self.warehouse_id,
                                    disposition=Disposition.EXTERNAL_LINKS, format=Format.JSON_ARRAY,
                                    wait_timeout="50s")
        while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
            time.sleep(2)
            resp = sx.get_statement(resp.statement_id)
        if resp.status.state != StatementState.SUCCEEDED:
            raise RuntimeError(f"{resp.status.state}: {resp.status.error.message if resp.status.error else ''}")

        columns = resp.manifest.schema.columns
        rows, chunk = [], resp.result
        while chunk is not None:
            for link in chunk.external_links or []:
                rows.extend(requests.get(link.external_link, timeout=120).json())  # pre-signed URL
            nxt = (chunk.external_links[-1].next_chunk_index if chunk.external_links else None)
            chunk = sx.get_statement_result_chunk_n(resp.statement_id, nxt) if nxt is not None else None

        df = pd.DataFrame(rows, columns=[c.name for c in columns])
        for col in columns:  # JSON_ARRAY returns every value as a string
            kind = col.type_name.value if col.type_name else ""
            if kind == "ARRAY":
                df[col.name] = df[col.name].map(lambda v: json.loads(v) if v is not None else None)
            elif kind in ("INT", "LONG", "SHORT", "BYTE"):
                df[col.name] = pd.to_numeric(df[col.name]).astype("Int64")
            elif kind in ("DOUBLE", "FLOAT", "DECIMAL"):
                df[col.name] = pd.to_numeric(df[col.name])
            elif kind == "BOOLEAN":
                df[col.name] = df[col.name].map({"true": True, "false": False})
        return df

    def _table_exists(self, name: str) -> bool:
        try:
            self.query(f"DESCRIBE TABLE {name}")
            return True
        except Exception:
            return False

    def table(self, name: str, cols: str = "*"):
        session = self

        class _Table:
            def select(self, *c):
                return session.table(name, ", ".join(c))

            def toPandas(self):
                return session.query(f"SELECT {cols} FROM {name}")

        return _Table()


def cached(embed):
    """Cache query embeddings so repeated evaluation runs don't re-call the endpoint."""
    memo: dict[str, list[float]] = {}

    def wrapper(texts):
        missing = [t for t in texts if t not in memo]
        if missing:
            for t, v in zip(missing, embed(missing)):
                memo[t] = v
        import numpy as np
        return np.asarray([memo[t] for t in texts], dtype=np.float32)

    return wrapper


# ---------------------------------------------------------------------------
# 04 · Ask questions
# ---------------------------------------------------------------------------

DEMO_QUESTIONS = [
    "Who should not take Eliquis?",
    "What is the recommended starting dose of Revlimid for multiple myeloma?",
    "Does Opdivo have any serious immune-mediated adverse reactions?",
    "Which drugs interact with clopidogrel?",
    "What is the dose of amoxicillin for adults?",  # not in the database -> should refuse
]


def run_demo(rag: dict) -> list[dict]:
    @mlflow.trace(name="rxlabel_assistant")
    def ask(question: str) -> dict:
        res = core.answer_question(question, rag["retriever"], rag["embed"], rag["chat"], rag["lookup"], k=cfg.TOP_K)
        return {**res, "contexts": res["contexts"].to_dict("records")}

    results = []
    for q in DEMO_QUESTIONS:
        res = ask(q)
        cites = [f"[{i}] {c['brand_name']} - {c['section_title']} ({c['score']:.3f})"
                 for i, c in enumerate(res["contexts"], start=1)]
        print(f"\nQ: {q}\nDetected drug: {', '.join(res['detected_drugs']) or 'none'}\nA: {res['answer']}")
        for c in cites:
            print("   ", c)
        print(f"    latency {res['latency_s']:.1f}s")
        results.append({"question": q, "detected_drugs": res["detected_drugs"], "answer": res["answer"],
                        "refused": res["refused"], "citations": cites, "latency_s": round(res["latency_s"], 2)})
    return results


# ---------------------------------------------------------------------------
# 05 · MLflow evaluation (same logic as 05_evaluate_mlflow)
# ---------------------------------------------------------------------------

def run_retrieval_eval(session, eval_df, run_name, embedding_mode, use_filter, embed_cache) -> dict:
    rag = rag_setup.load_components(session, embedding_mode=embedding_mode)
    if rag["mode"].startswith("gte"):
        rag["embed"] = embed_cache.setdefault("gte", cached(rag["embed"]))
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({"embedding_mode": rag["mode"], "drug_filter": use_filter, "top_k": cfg.TOP_K,
                           "chunk_max_words": cfg.CHUNK_MAX_WORDS, "chunk_overlap_words": cfg.CHUNK_OVERLAP_WORDS,
                           "n_chunks": len(rag["chunks"]), "n_drugs": rag["chunks"]["brand_name"].nunique(),
                           "runner": "laptop"})
        metrics, details = core.retrieval_metrics(eval_df, rag["retriever"], rag["embed"], rag["lookup"],
                                                  k=cfg.TOP_K, use_drug_filter=use_filter)
        mlflow.log_metrics(metrics)
        mlflow.log_table(details.astype(str), artifact_file="retrieval_details.json")
    print(f"{run_name}: {metrics}")
    return {"run": run_name, **metrics}


def run_answer_eval(rag: dict, eval_df: pd.DataFrame, n: int = 15) -> tuple[dict, pd.DataFrame]:
    rows = []
    for q in eval_df[eval_df["in_scope"]]["question"].head(n):
        res = core.answer_question(q, rag["retriever"], rag["embed"], rag["chat"], rag["lookup"], k=cfg.TOP_K)
        score, reason = core.judge_groundedness(res["answer"], res["contexts"], rag["chat"])
        rows.append({"question": q, "answer": res["answer"], "groundedness": score,
                     "judge_reason": reason, "latency_s": round(res["latency_s"], 2)})
        print(f"  judged {len(rows)}/{n}: groundedness={score}")
    answers = pd.DataFrame(rows)
    metrics = {"mean_groundedness": float(answers["groundedness"].mean()),
               "share_groundedness_4_plus": float((answers["groundedness"] >= 4).mean()),
               "mean_latency_s": float(answers["latency_s"].mean()),
               "p95_latency_s": float(answers["latency_s"].quantile(0.95))}
    with mlflow.start_run(run_name=f"answers_{rag['mode']}"):
        mlflow.log_params({"embedding_mode": rag["mode"], "chat_endpoint": cfg.CHAT_ENDPOINT,
                           "n_questions": len(answers), "runner": "laptop"})
        mlflow.log_metrics(metrics)
        mlflow.log_table(answers, artifact_file="answers.json")
    return metrics, answers


def main() -> None:
    w = WorkspaceClient()
    user = w.current_user.me().user_name
    warehouse = next(iter(w.warehouses.list()))
    print(f"User {user} | SQL warehouse '{warehouse.name}'")
    session = WarehouseSession(w, warehouse.id)

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(f"/Users/{user}/{cfg.MLFLOW_EXPERIMENT_NAME}")

    print("\n=== 04 · Ask questions ===")
    rag = rag_setup.load_components(session)
    embed_cache = {"gte": cached(rag["embed"])} if rag["mode"].startswith("gte") else {}
    if embed_cache:
        rag["embed"] = embed_cache["gte"]
    demo = run_demo(rag)

    print("\n=== 05 · Evaluate with MLflow ===")
    sections = session.table(cfg.SILVER_TABLE).toPandas()
    eval_df = core.build_eval_set(sections, n_questions=40, seed=7)
    print(f"{int(eval_df['in_scope'].sum())} in-scope + {int((~eval_df['in_scope']).sum())} out-of-scope questions")

    retrieval = [run_retrieval_eval(session, eval_df, "tfidf_baseline", "tfidf", True, embed_cache)]
    try:
        retrieval.append(run_retrieval_eval(session, eval_df, "gte_no_filter", "gte", False, embed_cache))
        retrieval.append(run_retrieval_eval(session, eval_df, "gte_with_filter", "gte", True, embed_cache))
    except RuntimeError as exc:
        print("Skipping gte runs:", exc)

    answer_metrics, answers = ({}, pd.DataFrame())
    if rag["chat"] is not None:
        print("Generating and judging answers ...")
        answer_metrics, answers = run_answer_eval(rag, eval_df)

    OUT_DIR.mkdir(exist_ok=True)
    summary = {"retrieval_mode": rag["mode"], "chat_endpoint": cfg.CHAT_ENDPOINT if rag["chat"] else None,
               "n_in_scope": int(eval_df["in_scope"].sum()), "n_out_of_scope": int((~eval_df["in_scope"]).sum()),
               "retrieval_runs": retrieval, "answer_metrics": answer_metrics}
    (OUT_DIR / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "demo_answers.json").write_text(json.dumps(demo, indent=2), encoding="utf-8")
    answers.to_csv(OUT_DIR / "judged_answers.csv", index=False)
    print("\n=== Summary ===")
    print(pd.DataFrame(retrieval).round(3).to_string(index=False))
    print(json.dumps(answer_metrics, indent=2))


if __name__ == "__main__":
    main()
