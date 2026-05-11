"""
agents/reasoner.py — Reasoning Agent (LangGraph + Claude).

Answers four classes of scientific chemical questions:

  1. SIMILARITY   — "What is structurally similar to aspirin?"
  2. COMPARATIVE  — "Compare ibuprofen and aspirin"
  3. TREND        — "How does solubility change with carbon chain length?"
  4. TARGET       — "What compounds inhibit COX-1?"

Architecture (LangGraph):
    classify_question
        ↓
    route  ──→  similarity_task
           ──→  comparative_task
           ──→  trend_task
           ──→  target_task
        ↓
    generate_answer
        ↓
    format_output

Each task node:
  - Fetches any needed data from PubChem
  - Computes hard metrics (Tanimoto, property deltas, etc.)
  - Packages structured context for the LLM
  The LLM then generates a scientifically reasoned explanation
  grounded in that context — it never reasons from memory alone.

Usage:
    from agents.reasoner import ReasoningAgent
    agent = ReasoningAgent()

    result = agent.run("What is structurally similar to aspirin?")
    print(result["answer"])

    result = agent.run("Compare ibuprofen and naproxen")
    print(result["answer"])
"""
from __future__ import annotations
import json
import logging
import math
from typing import Any, Literal, TypedDict

import anthropic
from langgraph.graph import StateGraph, END

from config import ANTHROPIC_API_KEY
from tools.pubchem_tool import search_compounds
from tools.models import RetrievedRecord

logger = logging.getLogger(__name__)

QuestionType = Literal["similarity", "comparative", "trend", "target", "unknown"]

# ── LangGraph state ────────────────────────────────────────────────────────────

class ReasonerState(TypedDict):
    question:       str
    question_type:  QuestionType
    context:        dict[str, Any]
    answer:         str
    compounds:      list[dict]
    client:         Any
    query_compound: str   # compound name passed from pipeline
    query_cid:      str   # PubChem CID passed from pipeline


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Classify question
# ══════════════════════════════════════════════════════════════════════════════

def classify_question(state: ReasonerState) -> ReasonerState:
    """
    Use Claude to classify the question type so we can route it correctly.
    Fast, cheap call with a tightly constrained prompt.
    """
    client = state["client"]
    q      = state["question"]

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",   # fast + cheap for classification
        max_tokens=20,
        system=(
            "Classify the chemical question into exactly one category. "
            "Reply with only one word: similarity, comparative, trend, or target."
            "\n- similarity: finding structurally similar compounds"
            "\n- comparative: comparing two or more specific compounds"
            "\n- trend: how a property changes across a series"
            "\n- target: finding compounds that act on a biological target"
        ),
        messages=[{"role": "user", "content": q}],
    )

    raw = response.content[0].text.strip().lower()
    valid: list[QuestionType] = ["similarity", "comparative", "trend", "target"]
    state["question_type"] = raw if raw in valid else "unknown"
    logger.info("ReasoningAgent: question_type='%s' for: %s", state["question_type"], q)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Route (conditional edge function)
# ══════════════════════════════════════════════════════════════════════════════

def route(state: ReasonerState) -> str:
    return state["question_type"]


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3a — Similarity task
# Fetches the query compound + similar compounds from PubChem similarity search
# Computes property-based similarity scores
# ══════════════════════════════════════════════════════════════════════════════

def similarity_task(state: ReasonerState) -> ReasonerState:
    """
    Find structurally similar compounds to the one mentioned in the question.
    Uses PubChem's fast 2D similarity search endpoint + property comparison.
    """
    import re, requests
    from tools.http_client import get_session
    from config import PUBCHEM_BASE

    session  = get_session()
    q        = state["question"]
    name     = state.get("query_compound") or ""
    base_cid = state.get("query_cid") or ""

    logger.info("Similarity: query_compound='%s' query_cid='%s'", name, base_cid)

    # If we already have the CID from the pipeline — skip all name searching
    # and go straight to the similarity search with the known CID
    if not base_cid:
        # No CID available — extract name and search
        if not name:
            name = _extract_compound_name(state["client"], q)
        logger.info("Similarity: searching PubChem for '%s'", name)
        base_records = search_compounds(name, max_results=1)
        if not base_records:
            state["context"] = {"error": f"Could not find compound: {name}"}
            state["compounds"] = []
            return state
        base_cid = base_records[0].record_id
        name     = base_records[0].title or name

    logger.info("Similarity: using CID=%s for '%s'", base_cid, name)

    # PubChem 2D similarity search
    try:
        sim_url = f"{PUBCHEM_BASE}/compound/fastsimilarity_2d/cid/{base_cid}/cids/JSON"
        resp    = session.get(sim_url, params={"Threshold": 70, "MaxRecords": 5}, timeout=15)
        resp.raise_for_status()
        similar_cids = [str(c) for c in resp.json().get("IdentifierList", {}).get("CID", [])][:5]
    except Exception as e:
        logger.warning("Similarity search failed: %s", e)
        similar_cids = []

    # Fetch properties for base + similar compounds
    all_cids    = [base_cid] + similar_cids
    all_records = _fetch_by_cids(all_cids, session)

    # Find base record from fetched results
    base_record = next((r for r in all_records if r.record_id == base_cid), None)
    base_props  = base_record.metadata if base_record else {}

    # Compute property deltas vs base
    comparisons = []
    for rec in all_records:
        if rec.record_id == base_cid:
            continue
        comparisons.append(_property_delta(base_props, rec.metadata, rec.record_id, rec.title))

    state["context"] = {
        "base_compound":     {"cid": base_cid, "name": name},
        "similar_compounds": comparisons,
        "similarity_threshold": "70% Tanimoto (2D fingerprint)",
    }
    state["compounds"] = [_record_to_dict(r) for r in all_records]
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3b — Comparative task
# Retrieves both compounds and builds a side-by-side property table
# ══════════════════════════════════════════════════════════════════════════════

def comparative_task(state: ReasonerState) -> ReasonerState:
    """
    Compare two or more compounds side by side on chemical properties.
    """
    from tools.http_client import get_session
    session = get_session()
    q       = state["question"]

    # Extract compound names
    names = _extract_compound_names(state["client"], q)
    logger.info("Comparative: comparing %s", names)

    records = []
    for name in names[:3]:   # cap at 3 compounds
        found = search_compounds(name, max_results=1)
        if found:
            records.append(found[0])
        else:
            logger.warning("Comparative: could not find '%s'", name)

    if len(records) < 2:
        # Fallback: try searching each name with common drug name variations
        logger.warning("Comparative: only %d record(s) found — trying fallback search", len(records))
        found_names = {r.title.lower() if r.title else "" for r in records}
        for name in names[:3]:
            if name.lower() in found_names:
                continue
            # Try with mesylate/hydrochloride salt suffix removed
            clean = name.split()[0]
            found = search_compounds(clean, max_results=1)
            if found and found[0] not in records:
                records.append(found[0])
                logger.info("Comparative: fallback found '%s' via '%s'", found[0].title, clean)

    if len(records) < 2:
        state["context"] = {"error": "Could not retrieve enough compounds to compare"}
        state["compounds"] = []
        return state

    # Build comparison table
    props = ["molecular_formula", "molecular_weight", "xlogp", "tpsa", "hbd", "hba"]
    table = {}
    for rec in records:
        table[rec.title or rec.record_id] = {p: rec.metadata.get(p) for p in props}

    # Property deltas (first vs rest)
    deltas = []
    base = records[0]
    for rec in records[1:]:
        deltas.append(_property_delta(base.metadata, rec.metadata,
                                      rec.record_id, rec.title or rec.record_id))

    state["context"] = {
        "compounds_compared": [r.title or r.record_id for r in records],
        "property_table":     table,
        "property_deltas":    deltas,
    }
    state["compounds"] = [_record_to_dict(r) for r in records]
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3c — Trend task
# Retrieves a homologous series and computes property trends
# ══════════════════════════════════════════════════════════════════════════════

def trend_task(state: ReasonerState) -> ReasonerState:
    """
    Analyse how a property changes across a chemical series.
    Fetches a homologous series from PubChem and computes the trend numerically.
    """
    q      = state["question"]
    client = state["client"]

    # Ask Claude to identify: what series + what property
    series_info = _extract_trend_info(client, q)
    compounds   = series_info.get("compounds", [])
    property_   = series_info.get("property", "solubility")
    logger.info("Trend: series=%s, property=%s", compounds, property_)

    records  = []
    for name in compounds[:6]:
        found = search_compounds(name, max_results=1)
        if found:
            records.append(found[0])

    if not records:
        state["context"] = {"error": "Could not retrieve compounds for trend analysis"}
        state["compounds"] = []
        return state

    # Build trend data points
    prop_map = {
        "solubility": "xlogp",   # logP is inverse proxy for aqueous solubility
        "lipophilicity": "xlogp",
        "polarity": "tpsa",
        "molecular_weight": "molecular_weight",
        "weight": "molecular_weight",
    }
    metric_key = prop_map.get(property_.lower(), "xlogp")

    trend_points = []
    for rec in records:
        val = rec.metadata.get(metric_key)
        trend_points.append({
            "compound":    rec.title or rec.record_id,
            "cid":         rec.record_id,
            metric_key:    val,
            "formula":     rec.metadata.get("molecular_formula"),
            "mw":          rec.metadata.get("molecular_weight"),
        })

    # Compute direction
    values = [p[metric_key] for p in trend_points if p[metric_key] is not None]
    trend_direction = _compute_trend(values)

    state["context"] = {
        "series":          compounds,
        "property_asked":  property_,
        "metric_used":     metric_key,
        "trend_points":    trend_points,
        "trend_direction": trend_direction,
    }
    state["compounds"] = [_record_to_dict(r) for r in records]
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3d — Target task
# Searches PubChem for compounds associated with a biological target
# ══════════════════════════════════════════════════════════════════════════════

def target_task(state: ReasonerState) -> ReasonerState:
    """
    Find compounds known to inhibit or interact with a given protein target.
    Uses PubChem bioassay keyword search as a proxy.
    """
    from tools.http_client import get_session
    from config import PUBCHEM_BASE
    session = get_session()
    q       = state["question"]

    target_name = _extract_target_name(state["client"], q)
    logger.info("Target: searching for compounds acting on '%s'", target_name)

    # Search PubChem compound descriptions for target name
    records = search_compounds(target_name + " inhibitor", max_results=5)
    if not records:
        records = search_compounds(target_name, max_results=5)

    state["context"] = {
        "target":   target_name,
        "strategy": "PubChem name search for '[target] inhibitor'",
        "note":     "For production use, integrate ChEMBL bioassay data for precise IC50 values.",
        "compounds_found": [
            {"cid": r.record_id, "name": r.title, "formula": r.metadata.get("molecular_formula")}
            for r in records
        ],
    }
    state["compounds"] = [_record_to_dict(r) for r in records]
    return state


def unknown_task(state: ReasonerState) -> ReasonerState:
    state["context"] = {"note": "Question type could not be classified. Attempting general reasoning."}
    state["compounds"] = []
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Generate answer
# Grounds LLM reasoning in the structured context gathered above
# ══════════════════════════════════════════════════════════════════════════════

_REASONING_SYSTEM = """You are an expert computational chemist providing scientifically grounded answers.

You will receive:
1. A chemical question
2. Structured data retrieved from PubChem (properties, similarity scores, trend data)

Your response must:
- Be grounded strictly in the provided data — do not invent values
- Include specific numbers (molecular weight, XLogP, TPSA, Tanimoto scores) from the data
- Explain the chemical reasoning behind the answer
- Be concise but precise — 3 to 6 sentences
- If data is limited, clearly state what is known vs what requires further experimental data

Format: plain prose, no bullet points, no headers.
"""

def generate_answer(state: ReasonerState) -> ReasonerState:
    client  = state["client"]
    context = state["context"]

    if "error" in context:
        state["answer"] = f"Could not complete analysis: {context['error']}"
        return state

    prompt = f"""Question: {state['question']}

Question type: {state['question_type']}

Retrieved chemical data:
{json.dumps(context, indent=2)}

Provide a scientifically reasoned answer grounded in the data above."""

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=600,
        system=_REASONING_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    state["answer"] = response.content[0].text.strip()
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Format output
# ══════════════════════════════════════════════════════════════════════════════

def format_output(state: ReasonerState) -> ReasonerState:
    logger.info("ReasoningAgent complete: %d word answer", len(state["answer"].split()))
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Graph assembly
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph() -> Any:
    g = StateGraph(ReasonerState)

    g.add_node("classify",    classify_question)
    g.add_node("similarity",  similarity_task)
    g.add_node("comparative", comparative_task)
    g.add_node("trend",       trend_task)
    g.add_node("target",      target_task)
    g.add_node("unknown",     unknown_task)
    g.add_node("generate",    generate_answer)
    g.add_node("format",      format_output)

    g.set_entry_point("classify")
    g.add_conditional_edges("classify", route, {
        "similarity":  "similarity",
        "comparative": "comparative",
        "trend":       "trend",
        "target":      "target",
        "unknown":     "unknown",
    })
    for task in ["similarity", "comparative", "trend", "target", "unknown"]:
        g.add_edge(task, "generate")
    g.add_edge("generate", "format")
    g.add_edge("format",   END)

    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def _extract_compound_name(client, question: str) -> str:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        system="Extract the single compound name from the question. Reply with only the compound name, nothing else.",
        messages=[{"role": "user", "content": question}],
    )
    return r.content[0].text.strip()


def _extract_compound_names(client, question: str) -> list[str]:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=40,
        system="Extract the compound names from the question. Reply with only a comma-separated list of names, nothing else.",
        messages=[{"role": "user", "content": question}],
    )
    return [n.strip() for n in r.content[0].text.strip().split(",") if n.strip()]


def _extract_target_name(client, question: str) -> str:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=20,
        system="Extract the biological target or protein name from the question. Reply with only the target name.",
        messages=[{"role": "user", "content": question}],
    )
    return r.content[0].text.strip()


def _extract_trend_info(client, question: str) -> dict:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=120,
        system=(
            "Extract the chemical series and property from the trend question. "
            "Reply with only a JSON object like: "
            '{"compounds": ["methanol","ethanol","propanol","butanol"], "property": "solubility"}'
        ),
        messages=[{"role": "user", "content": question}],
    )
    raw = r.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try:
        return json.loads(raw.strip())
    except Exception:
        return {"compounds": [], "property": "solubility"}


def _record_to_dict(rec: RetrievedRecord) -> dict:
    return {
        "cid":     rec.record_id,
        "name":    rec.title,
        "formula": rec.metadata.get("molecular_formula"),
        "mw":      rec.metadata.get("molecular_weight"),
        "xlogp":   rec.metadata.get("xlogp"),
        "tpsa":    rec.metadata.get("tpsa"),
        "hbd":     rec.metadata.get("hbd"),
        "hba":     rec.metadata.get("hba"),
        "smiles":  rec.metadata.get("smiles"),
    }


def _property_delta(base: dict, other: dict, other_id: str, other_name: str) -> dict:
    """Compute property differences between two compound metadata dicts."""
    result = {"cid": other_id, "name": other_name or other_id}
    for key in ["molecular_weight", "xlogp", "tpsa", "hbd", "hba"]:
        b = base.get(key)
        o = other.get(key)
        result[key] = o
        if b is not None and o is not None:
            try:
                result[f"{key}_delta"] = round(float(o) - float(b), 3)
            except (TypeError, ValueError):
                pass
    return result


def _compute_trend(values: list) -> str:
    if len(values) < 2:
        return "insufficient data"
    increasing = sum(1 for i in range(1, len(values)) if values[i] > values[i-1])
    decreasing = sum(1 for i in range(1, len(values)) if values[i] < values[i-1])
    if increasing > decreasing:
        return "increasing"
    elif decreasing > increasing:
        return "decreasing"
    return "no clear trend"


def _fetch_by_cids(cids: list[str], session) -> list[RetrievedRecord]:
    """Fetch compound records for a list of CIDs."""
    from tools.pubchem_tool import _fetch_properties
    from tools.models import DataSource
    if not cids:
        return []
    props = _fetch_properties(cids, session)
    records = []
    for cid, p in props.items():
        records.append(RetrievedRecord(
            source=DataSource.PUBCHEM,
            record_id=str(cid),
            title=p.get("IUPACName") or f"CID {cid}",
            url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            raw=p,
            metadata={
                "molecular_formula": p.get("MolecularFormula"),
                "molecular_weight":  p.get("MolecularWeight"),
                "smiles":            p.get("SMILES") or p.get("ConnectivitySMILES"),
                "inchikey":          p.get("InChIKey"),
                "xlogp":             p.get("XLogP"),
                "hbd":               p.get("HBondDonorCount"),
                "hba":               p.get("HBondAcceptorCount"),
                "tpsa":              p.get("TPSA"),
            },
        ))
    return records


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

class ReasoningAgent:
    """
    High-level wrapper around the reasoning LangGraph.

    Example:
        agent = ReasoningAgent()
        result = agent.run("What is structurally similar to aspirin?")
        print(result["answer"])
        print(result["question_type"])
        print(result["compounds"])   # raw evidence used
    """
    def __init__(self) -> None:
        self._graph  = _build_graph()
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def run(self, question: str, query_compound: str = "", query_cid: str = "") -> dict:
        final = self._graph.invoke({
            "question":       question,
            "question_type":  "unknown",
            "context":        {},
            "answer":         "",
            "compounds":      [],
            "client":         self._client,
            "query_compound": query_compound,
            "query_cid":      query_cid,
        })
        return {
            "question":      final["question"],
            "question_type": final["question_type"],
            "answer":        final["answer"],
            "compounds":     final["compounds"],
            "context":       final["context"],
        }