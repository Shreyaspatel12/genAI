"""
agents/dataset_builder.py — Dataset Builder Agent (LangGraph).

Aggregates outputs from the Retriever, Extraction, and Reasoning agents
into a single structured JSON dataset ready for downstream use.

Output schema:
{
  "metadata": { run info, timestamps, sources, stats },
  "compounds": [ { ...extracted fields + reasoning answers } ],
  "reasoning_log": [ { question, type, answer } ],
  "errors": { ... }
}

Usage:
    from agents.dataset_builder import DatasetBuilderAgent
    agent = DatasetBuilderAgent()
    result = agent.run(
        query="aspirin",
        extraction_result=extraction_result,
        reasoning_results=[...],   # optional
    )
    print(result["output_path"])
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, END

from tools.models import ExtractionResult

logger = logging.getLogger(__name__)


class BuilderState(TypedDict):
    query:             str
    extraction_result: ExtractionResult
    reasoning_results: list[dict]
    dataset:           dict[str, Any]
    output_path:       str
    errors:            dict[str, str]


def validate_inputs(state: BuilderState) -> BuilderState:
    n = state["extraction_result"].total if state["extraction_result"] else 0
    logger.info("DatasetBuilder: %d compound(s), %d reasoning result(s) for '%s'",
                n, len(state.get("reasoning_results") or []), state["query"])
    if n == 0:
        state["errors"]["validation"] = "No extracted compounds to build dataset from"
    return state


def build_compound_records(state: BuilderState) -> BuilderState:
    if "validation" in state["errors"]:
        state["dataset"]["compounds"] = []
        return state

    compounds = []
    for i, c in enumerate(state["extraction_result"].compounds):
        compounds.append({
            "record_id":        f"{state['query'].replace(' ', '_')}_{i+1:03d}",
            "query":            state["query"],
            "molecule_name":    c.molecule_name,
            "chemical_formula": c.chemical_formula,
            "smiles":           c.smiles,
            "target_protein":   c.target_protein,
            "activity_value":   c.activity_value,
            "activity_units":   c.activity_units,
            "activity_type":    c.activity_type,
            # ── NEW: PubMed-enriched fields ──────────────────────
            "mechanism_of_action": c.mechanism_of_action,
            "disease_indication":  c.disease_indication,
            "pubmed_ids":          c.pubmed_ids,
            # ─────────────────────────────────────────────────────
            "confidence":       c.confidence,
            "notes":            c.notes,
            "source":           c.source,
            "source_id":        c.source_id,
            "extracted_at":     datetime.now(timezone.utc).isoformat(),
        })

    state["dataset"]["compounds"] = compounds
    logger.info("DatasetBuilder: built %d compound record(s)", len(compounds))
    return state


def attach_reasoning_log(state: BuilderState) -> BuilderState:
    log = [
        {
            "question":       r.get("question"),
            "question_type":  r.get("question_type"),
            "answer":         r.get("answer"),
            "compounds_used": [c.get("cid") for c in (r.get("compounds") or [])],
        }
        for r in (state.get("reasoning_results") or [])
    ]
    state["dataset"]["reasoning_log"] = log
    logger.info("DatasetBuilder: attached %d reasoning entry/entries", len(log))
    return state


def build_metadata(state: BuilderState) -> BuilderState:
    compounds = state["dataset"].get("compounds", [])
    total     = len(compounds)
    filled    = lambda f: sum(1 for c in compounds if c.get(f) is not None)

    state["dataset"]["metadata"] = {
        "query":            state["query"],
        "created_at":       datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "0.1.0",
        "sources_used":     list({c["source"] for c in compounds}),
        "stats": {
            "total_compounds":    total,
            "with_smiles":        filled("smiles"),
            "with_formula":       filled("chemical_formula"),
            "with_target":        filled("target_protein"),
            "with_activity":      filled("activity_value"),
            "with_mechanism":     filled("mechanism_of_action"),
            "with_disease":       filled("disease_indication"),
            "with_pubmed_refs":   sum(1 for c in compounds if c.get("pubmed_ids")),
            "avg_confidence":     round(
                sum(c["confidence"] for c in compounds) / total, 3
            ) if total else 0,
            "reasoning_entries":  len(state["dataset"].get("reasoning_log", [])),
            "extraction_errors":  len(state["extraction_result"].errors),
        },
    }
    return state


def save_dataset(state: BuilderState) -> BuilderState:
    out_dir  = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_q    = state["query"].replace(" ", "_").lower()
    out_path  = out_dir / f"dataset_{safe_q}_{timestamp}.json"

    final = {
        "metadata":      state["dataset"].get("metadata", {}),
        "compounds":     state["dataset"].get("compounds", []),
        "reasoning_log": state["dataset"].get("reasoning_log", []),
        "errors":        {**state["errors"], **state["extraction_result"].errors},
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)

    state["output_path"] = str(out_path)
    logger.info("DatasetBuilder: saved → %s", out_path)
    return state


def print_summary(state: BuilderState) -> BuilderState:
    stats = state["dataset"].get("metadata", {}).get("stats", {})
    logger.info("━━━ Dataset Summary ━━━")
    for k, v in stats.items():
        logger.info("  %-22s: %s", k, v)
    logger.info("  %-22s: %s", "output_path", state["output_path"])
    return state


def _build_graph() -> Any:
    g = StateGraph(BuilderState)
    for name, fn in [("validate", validate_inputs), ("compounds", build_compound_records),
                     ("reasoning", attach_reasoning_log), ("metadata", build_metadata),
                     ("save", save_dataset), ("summary", print_summary)]:
        g.add_node(name, fn)
    g.set_entry_point("validate")
    g.add_edge("validate",  "compounds")
    g.add_edge("compounds", "reasoning")
    g.add_edge("reasoning", "metadata")
    g.add_edge("metadata",  "save")
    g.add_edge("save",      "summary")
    g.add_edge("summary",   END)
    return g.compile()


class DatasetBuilderAgent:
    """
    Aggregates pipeline outputs into a structured JSON dataset.

    Example:
        builder = DatasetBuilderAgent()
        result  = builder.run(
            query="aspirin",
            extraction_result=extraction_result,
            reasoning_results=[r1, r2],
        )
        print(result["output_path"])
        print(result["stats"])
    """
    def __init__(self) -> None:
        self._graph = _build_graph()

    def run(
        self,
        query:             str,
        extraction_result: ExtractionResult,
        reasoning_results: Optional[list[dict]] = None,
    ) -> dict[str, Any]:
        final = self._graph.invoke({
            "query":             query,
            "extraction_result": extraction_result,
            "reasoning_results": reasoning_results or [],
            "dataset":           {},
            "output_path":       "",
            "errors":            {},
        })
        return {
            "output_path": final["output_path"],
            "stats":       final["dataset"].get("metadata", {}).get("stats", {}),
            "dataset":     final["dataset"],
            "errors":      final["errors"],
        }
