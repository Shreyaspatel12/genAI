"""
agents/filter.py — Relevance Filter Agent (LangGraph).

Scores and filters extracted compound records using 4 layered criteria:

  Layer 1 — Keyword Match      (free, instant)
  Layer 2 — Field Completeness (free, instant)
  Layer 3 — Confidence Threshold (free, uses extractor score)
  Layer 4 — LLM Relevance Score  (Claude, runs last to minimise API cost)

Each record gets a composite score 0.0–1.0. Records below the
minimum threshold are removed before reaching the DatasetBuilder.

Usage:
    from agents.filter import RelevanceFilterAgent
    agent = RelevanceFilterAgent(min_score=0.5, llm_threshold=0.6)
    result = agent.run(query="aspirin", compounds=extraction_result.compounds)

    print(result["kept"])    # list of ScoredCompound
    print(result["dropped"]) # list of ScoredCompound with reasons
    print(result["stats"])
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TypedDict

import anthropic
from langgraph.graph import StateGraph, END

from config import ANTHROPIC_API_KEY
from tools.models import ExtractedCompound

logger = logging.getLogger(__name__)

# ── Score weights (must sum to 1.0) ───────────────────────────────────────────
W_KEYWORD    = 0.20
W_COMPLETENESS = 0.25
W_CONFIDENCE = 0.20
W_LLM        = 0.35


# ── Scored compound wrapper ────────────────────────────────────────────────────

@dataclass
class ScoredCompound:
    compound:           ExtractedCompound
    keyword_score:      float = 0.0
    completeness_score: float = 0.0
    confidence_score:   float = 0.0
    llm_score:          float = 0.0
    composite_score:    float = 0.0
    llm_reason:         Optional[str] = None
    dropped:            bool  = False
    drop_reason:        Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "molecule_name":    self.compound.molecule_name,
            "source_id":        self.compound.source_id,
            "composite_score":  round(self.composite_score, 3),
            "keyword_score":    round(self.keyword_score, 3),
            "completeness_score": round(self.completeness_score, 3),
            "confidence_score": round(self.confidence_score, 3),
            "llm_score":        round(self.llm_score, 3),
            "llm_reason":       self.llm_reason,
            "dropped":          self.dropped,
            "drop_reason":      self.drop_reason,
        }


# ── LangGraph state ────────────────────────────────────────────────────────────

class FilterState(TypedDict):
    query:         str
    compounds:     list[ExtractedCompound]
    scored:        list[ScoredCompound]
    min_score:     float
    llm_threshold: float
    client:        Any
    stats:         dict[str, Any]


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Keyword match
# Checks if query terms appear in name, formula, SMILES, notes
# ══════════════════════════════════════════════════════════════════════════════

def score_keyword(state: FilterState) -> FilterState:
    """
    Score 1.0 if the query keyword appears in the record.
    Score 0.5 if a partial match is found.
    Score 0.0 if no match.
    Fast — no API calls.
    """
    query_terms = set(state["query"].lower().split())
    scored = []

    for c in state["compounds"]:
        # Build searchable text blob from all available fields
        blob = " ".join(filter(None, [
            c.molecule_name,
            c.chemical_formula,
            c.smiles,
            c.notes,
            c.target_protein,
        ])).lower()

        # Exact full query match
        if state["query"].lower() in blob:
            kw_score = 1.0
        # All query terms present (handles multi-word queries)
        elif all(t in blob for t in query_terms):
            kw_score = 0.9
        # Any query term present
        elif any(t in blob for t in query_terms):
            kw_score = 0.5
        else:
            kw_score = 0.0

        sc = ScoredCompound(compound=c, keyword_score=kw_score)
        scored.append(sc)
        logger.debug("Keyword [%s]: %.1f", c.source_id, kw_score)

    state["scored"] = scored
    logger.info("Filter Layer 1 (keyword): scored %d record(s)", len(scored))
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Field completeness
# Scores based on how many critical fields are populated
# ══════════════════════════════════════════════════════════════════════════════

# Fields and their weights within the completeness score
_FIELD_WEIGHTS = {
    "molecule_name":    0.20,
    "chemical_formula": 0.20,
    "smiles":           0.25,
    "target_protein":   0.20,
    "activity_value":   0.15,
}

def score_completeness(state: FilterState) -> FilterState:
    """
    Score based on fraction of critical fields populated.
    SMILES and formula weighted highest as they are structurally essential.
    """
    for sc in state["scored"]:
        c = sc.compound
        score = 0.0
        for field_name, weight in _FIELD_WEIGHTS.items():
            if getattr(c, field_name, None) is not None:
                score += weight
        sc.completeness_score = round(score, 3)
        logger.debug("Completeness [%s]: %.2f", c.source_id, score)

    logger.info("Filter Layer 2 (completeness): scored %d record(s)", len(state["scored"]))
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Confidence threshold
# Uses the confidence score already produced by the ExtractionAgent
# ══════════════════════════════════════════════════════════════════════════════

def score_confidence(state: FilterState) -> FilterState:
    """
    Normalise the extractor's confidence (0–1) directly as this layer's score.
    Also hard-drops records with confidence < 0.3 (not worth LLM evaluation).
    """
    dropped_early = 0
    for sc in state["scored"]:
        sc.confidence_score = sc.compound.confidence or 0.0

        if sc.confidence_score < 0.3:
            sc.dropped     = True
            sc.drop_reason = f"Confidence too low: {sc.confidence_score:.2f} < 0.30"
            dropped_early += 1

    logger.info("Filter Layer 3 (confidence): %d hard-dropped (conf < 0.3)", dropped_early)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — LLM relevance score
# Claude evaluates scientific relevance — only runs on non-dropped records
# ══════════════════════════════════════════════════════════════════════════════

_LLM_SYSTEM = """You are a scientific relevance evaluator for a chemical data pipeline.

Given a query and a compound record, score the relevance of the record to the query.

Reply with ONLY a JSON object in this exact format:
{"score": 0.85, "reason": "one sentence explanation"}

Scoring guide:
- 0.9–1.0 : Directly relevant — compound IS the query or a primary match
- 0.7–0.9 : Highly relevant — closely related compound or direct structural analog
- 0.5–0.7 : Moderately relevant — same drug class or related mechanism
- 0.3–0.5 : Weakly relevant — tangential relationship only
- 0.0–0.3 : Not relevant — unrelated compound

Be strict. A score above 0.6 means the record should be kept.
"""

def score_llm(state: FilterState) -> FilterState:
    """
    Use Claude Haiku to evaluate scientific relevance.
    Only evaluates records that passed the confidence check (not already dropped).
    """
    client    = state["client"]
    to_eval   = [sc for sc in state["scored"] if not sc.dropped]
    skipped   = len(state["scored"]) - len(to_eval)

    logger.info("Filter Layer 4 (LLM): evaluating %d record(s), skipping %d dropped",
                len(to_eval), skipped)

    for sc in to_eval:
        c = sc.compound
        prompt = f"""Query: "{state['query']}"

Compound record:
- Name: {c.molecule_name or 'unknown'}
- Formula: {c.chemical_formula or 'unknown'}
- SMILES: {c.smiles or 'unknown'}
- Target: {c.target_protein or 'not specified'}
- Notes: {c.notes or 'none'}
- Source ID: {c.source_id}

Score the relevance of this record to the query."""

        try:
            resp = client.messages.create(
                model="claude-haiku-4-5-20251001",   # cheap + fast for scoring
                max_tokens=80,
                system=_LLM_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"): raw = raw[4:]
            import json
            data = json.loads(raw.strip())
            sc.llm_score  = float(data.get("score", 0.5))
            sc.llm_reason = data.get("reason", "")
            logger.info("  LLM [%s]: %.2f — %s", c.source_id, sc.llm_score, sc.llm_reason)

        except Exception as e:
            logger.warning("  LLM scoring failed for %s: %s — defaulting to 0.5", c.source_id, e)
            sc.llm_score  = 0.5
            sc.llm_reason = f"LLM scoring failed: {e}"

    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Compute composite score and apply final filter
# ══════════════════════════════════════════════════════════════════════════════

def compute_and_filter(state: FilterState) -> FilterState:
    """
    Compute weighted composite score and drop records below min_score.
    """
    min_score = state["min_score"]

    for sc in state["scored"]:
        if sc.dropped:
            sc.composite_score = 0.0
            continue

        sc.composite_score = (
            W_KEYWORD      * sc.keyword_score      +
            W_COMPLETENESS * sc.completeness_score +
            W_CONFIDENCE   * sc.confidence_score   +
            W_LLM          * sc.llm_score
        )

        if sc.composite_score < min_score:
            sc.dropped     = True
            sc.drop_reason = (
                f"Composite score {sc.composite_score:.3f} < threshold {min_score:.2f} "
                f"(kw={sc.keyword_score:.2f}, complete={sc.completeness_score:.2f}, "
                f"conf={sc.confidence_score:.2f}, llm={sc.llm_score:.2f})"
            )

    kept    = [sc for sc in state["scored"] if not sc.dropped]
    dropped = [sc for sc in state["scored"] if sc.dropped]

    # Sort kept by score descending
    kept.sort(key=lambda s: s.composite_score, reverse=True)

    state["stats"] = {
        "total_input":   len(state["scored"]),
        "kept":          len(kept),
        "dropped":       len(dropped),
        "retention_pct": round(len(kept) / max(len(state["scored"]), 1) * 100, 1),
        "avg_score_kept": round(
            sum(s.composite_score for s in kept) / max(len(kept), 1), 3
        ),
        "score_breakdown": {
            "weight_keyword":      W_KEYWORD,
            "weight_completeness": W_COMPLETENESS,
            "weight_confidence":   W_CONFIDENCE,
            "weight_llm":          W_LLM,
        },
    }

    logger.info("Filter complete: kept %d / %d (%.0f%%) | avg score: %.3f",
                len(kept), len(state["scored"]),
                state["stats"]["retention_pct"],
                state["stats"]["avg_score_kept"])

    for sc in dropped:
        logger.info("  DROPPED [%s]: %s", sc.compound.source_id, sc.drop_reason)

    return state


# ══════════════════════════════════════════════════════════════════════════════
# Graph assembly
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph() -> Any:
    g = StateGraph(FilterState)
    for name, fn in [
        ("keyword",      score_keyword),
        ("completeness", score_completeness),
        ("confidence",   score_confidence),
        ("llm",          score_llm),
        ("filter",       compute_and_filter),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("keyword")
    g.add_edge("keyword",      "completeness")
    g.add_edge("completeness", "confidence")
    g.add_edge("confidence",   "llm")
    g.add_edge("llm",          "filter")
    g.add_edge("filter",       END)
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

class RelevanceFilterAgent:
    """
    Scores and filters extracted compounds using 4 layered criteria.

    Args:
        min_score     : composite score threshold to keep a record (default 0.4)
        llm_threshold : LLM score below which a record is flagged (informational)

    Example:
        agent  = RelevanceFilterAgent(min_score=0.4)
        result = agent.run(query="aspirin", compounds=extraction_result.compounds)

        for sc in result["kept"]:
            print(sc.compound.molecule_name, sc.composite_score)
    """
    def __init__(self, min_score: float = 0.4, llm_threshold: float = 0.6) -> None:
        self._graph         = _build_graph()
        self._client        = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.min_score      = min_score
        self.llm_threshold  = llm_threshold

    def run(
        self,
        query:     str,
        compounds: list[ExtractedCompound],
    ) -> dict[str, Any]:

        final = self._graph.invoke({
            "query":         query,
            "compounds":     compounds,
            "scored":        [],
            "min_score":     self.min_score,
            "llm_threshold": self.llm_threshold,
            "client":        self._client,
            "stats":         {},
        })

        kept    = [sc for sc in final["scored"] if not sc.dropped]
        dropped = [sc for sc in final["scored"] if sc.dropped]

        return {
            "kept":          kept,
            "dropped":       dropped,
            "kept_compounds": [sc.compound for sc in kept],
            "stats":         final["stats"],
            "scores":        [sc.to_dict() for sc in final["scored"]],
        }
