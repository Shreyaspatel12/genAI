"""
agents/retriever.py — RetrieverAgent (LangGraph).

Now supports BOTH PubChem and PubMed in a single run.

Flow:
    plan → fetch_pubchem → fetch_pubmed → aggregate

PubMed search query is automatically built as:
    "{compound_name} [compound name or target]"
    e.g. "Carboplatin cancer treatment mechanism"

Usage:
    from agents.retriever import RetrieverAgent
    from tools.models import RetrievalQuery, DataSource

    # PubChem only (original behaviour)
    result = RetrieverAgent().run(RetrievalQuery(
        query="aspirin", sources=[DataSource.PUBCHEM]
    ))

    # Both PubChem + PubMed
    result = RetrieverAgent().run(RetrievalQuery(
        query="aspirin", sources=[DataSource.PUBCHEM, DataSource.PUBMED]
    ))
"""
from __future__ import annotations
import logging
from typing import Any, TypedDict

from langgraph.graph import StateGraph, END

from tools.models import DataSource, RetrievalQuery, RetrievalResult, RetrievedRecord
from tools.pubchem_tool import search_compounds
from tools.pubmed_tool  import search_articles

logger = logging.getLogger(__name__)


class RetrieverState(TypedDict):
    query:   RetrievalQuery
    records: list[RetrievedRecord]
    errors:  dict[str, str]


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Plan
# ══════════════════════════════════════════════════════════════════════════════

def plan_retrieval(state: RetrieverState) -> RetrieverState:
    q = state["query"]
    logger.info("RetrieverAgent: query='%s' | max_results=%d | sources=%s",
                q.query, q.max_results, [s.value for s in q.sources])
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Fetch PubChem
# ══════════════════════════════════════════════════════════════════════════════

def fetch_pubchem(state: RetrieverState) -> RetrieverState:
    query = state["query"]

    if DataSource.PUBCHEM not in query.sources:
        return state

    try:
        results = search_compounds(query.query, query.max_results)
        logger.info("  [pubchem] → %d record(s)", len(results))
        state["records"].extend(results)
    except Exception as e:
        logger.error("  [pubchem] FAILED: %s", e)
        state["errors"]["pubchem"] = str(e)

    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Fetch PubMed
# ══════════════════════════════════════════════════════════════════════════════

def fetch_pubmed(state: RetrieverState) -> RetrieverState:
    query = state["query"]

    if DataSource.PUBMED not in query.sources:
        return state

    # Build a rich PubMed search query combining compound name + context
    # e.g. "Carboplatin" → "Carboplatin mechanism activity inhibitor"
    compound = query.query
    pubmed_query = (f'("{compound}"[tiab] OR "{compound}"[MeSH Terms]) '
                    f'AND (mechanism OR activity OR pharmacology OR "clinical trial")')

    try:
        results = search_articles(
            query=pubmed_query,
            max_results=query.max_results,
            date_from=query.date_from,
            date_to=query.date_to,
        )
        logger.info("  [pubmed] → %d record(s)", len(results))
        state["records"].extend(results)
    except Exception as e:
        logger.error("  [pubmed] FAILED: %s", e)
        state["errors"]["pubmed"] = str(e)

    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Aggregate
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(state: RetrieverState) -> RetrieverState:
    pubchem_count = sum(1 for r in state["records"] if r.source == DataSource.PUBCHEM)
    pubmed_count  = sum(1 for r in state["records"] if r.source == DataSource.PUBMED)
    logger.info(
        "RetrieverAgent complete: %d record(s) total "
        "(PubChem: %d, PubMed: %d) | %d error(s)",
        len(state["records"]), pubchem_count, pubmed_count, len(state["errors"])
    )
    return state


# ══════════════════════════════════════════════════════════════════════════════
# Graph assembly
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph() -> Any:
    g = StateGraph(RetrieverState)
    g.add_node("plan",          plan_retrieval)
    g.add_node("fetch_pubchem", fetch_pubchem)
    g.add_node("fetch_pubmed",  fetch_pubmed)
    g.add_node("aggregate",     aggregate)

    g.set_entry_point("plan")
    g.add_edge("plan",          "fetch_pubchem")
    g.add_edge("fetch_pubchem", "fetch_pubmed")
    g.add_edge("fetch_pubmed",  "aggregate")
    g.add_edge("aggregate",     END)
    return g.compile()


class RetrieverAgent:
    def __init__(self) -> None:
        self._graph = _build_graph()

    def run(self, query: RetrievalQuery) -> RetrievalResult:
        final = self._graph.invoke({
            "query":   query,
            "records": [],
            "errors":  {},
        })
        return RetrievalResult(
            query=query.query,
            records=final["records"],
            errors=final["errors"],
        )
