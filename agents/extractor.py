"""
agents/extractor.py — Extraction Agent (LangGraph + Claude).

Updated to handle BOTH PubChem and PubMed records.

PubChem records  → extracts: name, formula, SMILES (structural data)
PubMed records   → extracts: target, activity, mechanism, disease (biological data)

After individual extraction, a MERGE step combines PubChem structural
data with PubMed biological data into unified ExtractedCompound records.

Architecture:
    prepare → extract_records → validate → merge_pubmed → aggregate
"""
from __future__ import annotations
import json
import logging
from typing import Any, TypedDict

import anthropic
from langgraph.graph import StateGraph, END

from config import ANTHROPIC_API_KEY
from tools.models import DataSource, RetrievedRecord, ExtractedCompound, ExtractionResult

logger = logging.getLogger(__name__)


# Prompts — separate prompts for PubChem vs PubMed

_SYSTEM_PUBCHEM = """You are a precise chemical data extraction engine.

Given a PubChem compound record, extract ONLY these fields:
- molecule_name      : common name or IUPAC name
- chemical_formula   : molecular formula (e.g. C9H8O4)
- smiles             : SMILES string
- target_protein     : biological target if mentioned, else null
- activity_value     : numeric bioactivity value if mentioned, else null
- activity_units     : units e.g. µM, nM, else null
- activity_type      : assay type e.g. IC50, Ki, else null
- mechanism_of_action: how it works if mentioned, else null
- disease_indication : disease treated if mentioned, else null
- confidence         : float 0.0-1.0 reflecting extraction confidence
- notes              : brief note if uncertain, else null

Rules:
- Return ONLY a valid JSON object. No markdown, no explanation.
- Never invent values. If not present in source, return null.
- For activity_value, extract the number only (not units).
"""

_SYSTEM_PUBMED = """You are a biomedical literature extraction engine specialising in pharmacology.

Given a PubMed research paper title and abstract, extract ONLY these fields:
- molecule_name      : name of the main compound studied
- chemical_formula   : molecular formula if mentioned, else null
- smiles             : SMILES string if mentioned, else null
- target_protein     : the protein, enzyme, or receptor the compound acts on (e.g. COX-2, EGFR, BCR-ABL)
- activity_value     : the most prominent numeric bioactivity value (e.g. 0.3)
- activity_units     : units for activity (e.g. µM, nM, mg/kg)
- activity_type      : assay type (e.g. IC50, Ki, EC50, MIC, GI50)
- mechanism_of_action: clear one-sentence description of HOW the compound works at molecular level
                       (e.g. "inhibits DNA replication by forming intrastrand crosslinks")
- disease_indication : specific disease or condition treated (e.g. "ovarian cancer", "Type 2 diabetes")
- confidence         : float 0.0-1.0 — how confident are you in this extraction?
                       1.0 = all fields explicitly stated, 0.5 = some inferred from context
- notes              : any important caveats, multiple compounds, or uncertain fields

Rules:
- Return ONLY a valid JSON object. No markdown, no explanation, no code fences.
- Never invent values. If a field is not stated in the abstract, return null.
- mechanism_of_action and disease_indication are the most important fields — prioritise these.
- If the paper studies multiple compounds, extract the primary/most prominent one.
- For activity_value, extract the number only (not units).
"""

_USER_TEMPLATE = """Extract chemical/biomedical data from this record:

SOURCE: {source}
ID: {record_id}
TITLE: {title}

METADATA:
{metadata}

ABSTRACT / DESCRIPTION:
{abstract}

Return a JSON object with fields: molecule_name, chemical_formula, smiles,
target_protein, activity_value, activity_units, activity_type,
mechanism_of_action, disease_indication, confidence, notes.
"""


class ExtractionState(TypedDict):
    query:     str
    records:   list[RetrievedRecord]
    compounds: list[ExtractedCompound]
    errors:    dict[str, str]
    client:    Any


def prepare(state: ExtractionState) -> ExtractionState:
    state["client"]    = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    state["compounds"] = []
    state["errors"]    = {}
    pubchem = sum(1 for r in state["records"] if r.source == DataSource.PUBCHEM)
    pubmed  = sum(1 for r in state["records"] if r.source == DataSource.PUBMED)
    logger.info("ExtractionAgent: %d record(s) — PubChem: %d, PubMed: %d | query: '%s'",
                len(state["records"]), pubchem, pubmed, state["query"])
    return state


def extract_records(state: ExtractionState) -> ExtractionState:
    client    = state["client"]
    compounds = state["compounds"]
    errors    = state["errors"]

    for record in state["records"]:
        try:
            # Use different system prompt depending on source
            system = _SYSTEM_PUBMED if record.source == DataSource.PUBMED else _SYSTEM_PUBCHEM
            compound = _extract_one(client, record, system)
            compounds.append(compound)
            logger.info(
                "  [%s] %s → %s | target: %s | mechanism: %s | disease: %s",
                record.source.value,
                record.record_id,
                compound.molecule_name or "?",
                compound.target_protein or "—",
                (compound.mechanism_of_action or "—")[:50],
                compound.disease_indication or "—",
            )
        except Exception as e:
            logger.error("  [%s] %s → FAILED: %s", record.source.value, record.record_id, e)
            errors[record.record_id] = str(e)

    return state


def validate(state: ExtractionState) -> ExtractionState:
    """
    Cross-check PubChem records — structural fields from PubChem metadata
    are authoritative and override whatever Claude extracted.
    """
    for compound in state["compounds"]:
        original = next(
            (r for r in state["records"] if r.record_id == compound.source_id), None
        )
        if original is None or original.source != DataSource.PUBCHEM:
            continue

        meta = original.metadata

        # Fill missing formula from PubChem metadata
        if meta.get("molecular_formula") and not compound.chemical_formula:
            compound.chemical_formula = meta["molecular_formula"]

        # Fill missing SMILES from PubChem metadata
        if meta.get("smiles") and not compound.smiles:
            compound.smiles = meta["smiles"]

        # If formula mismatch — PubChem always wins
        if (meta.get("molecular_formula") and compound.chemical_formula
                and meta["molecular_formula"] != compound.chemical_formula):
            logger.warning("Validator: formula mismatch for %s — using PubChem value", compound.source_id)
            compound.chemical_formula = meta["molecular_formula"]
            compound.notes = (compound.notes or "") + " [formula corrected by validator]"

        # Remove implausible activity values
        if compound.activity_value is not None and compound.activity_value < 0:
            compound.activity_value = None
            compound.notes = (compound.notes or "") + " [negative activity removed]"

    return state


def merge_pubmed(state: ExtractionState) -> ExtractionState:
    """
    This is the key new step.

    PubChem records have: name, formula, SMILES  (structural — good)
    PubMed records have:  target, activity, mechanism, disease  (biological — good)

    This node merges them: for each PubChem compound, find relevant PubMed
    extractions for the same compound and fill in any missing biological fields.

    Matching is done by compound name (case-insensitive substring match).
    The PMID of each contributing paper is recorded in pubmed_ids.
    """
    pubchem_compounds = [c for c in state["compounds"] if c.source == DataSource.PUBCHEM.value]
    pubmed_compounds  = [c for c in state["compounds"] if c.source == DataSource.PUBMED.value]

    if not pubmed_compounds:
        logger.info("Merge: no PubMed records to merge")
        return state

    logger.info("Merge: merging %d PubMed extraction(s) into %d PubChem record(s)",
                len(pubmed_compounds), len(pubchem_compounds))

    for pc in pubchem_compounds:
        pc_name = (pc.molecule_name or "").lower()

        for pm in pubmed_compounds:
            pm_name = (pm.molecule_name or "").lower()

            pc_words   = set(w for w in pc_name.split() if len(w) > 3)
            pm_words   = set(w for w in pm_name.split() if len(w) > 3)
            query_word = state["query"].lower()

            matched = (
                pc_name in pm_name or
                pm_name in pc_name or
                bool(pc_words & pm_words) or
                query_word in pc_name or
                query_word in pm_name
            )

            if not matched:
                continue

            # Fill in missing biological fields from PubMed
            if not pc.target_protein and pm.target_protein:
                pc.target_protein = pm.target_protein
                logger.info("  Merged target_protein: %s → %s", pc.molecule_name, pm.target_protein)

            if not pc.activity_value and pm.activity_value:
                pc.activity_value = pm.activity_value
                pc.activity_units = pm.activity_units
                pc.activity_type  = pm.activity_type
                logger.info("  Merged activity: %s %s", pm.activity_value, pm.activity_units)

            if not pc.mechanism_of_action and pm.mechanism_of_action:
                pc.mechanism_of_action = pm.mechanism_of_action
                logger.info("  Merged mechanism: %s", pm.mechanism_of_action[:60])

            if not pc.disease_indication and pm.disease_indication:
                pc.disease_indication = pm.disease_indication
                logger.info("  Merged disease: %s", pm.disease_indication)

            # Record which PMID contributed this data
            if pm.source_id not in pc.pubmed_ids:
                pc.pubmed_ids.append(pm.source_id)

    
    state["compounds"] = pubchem_compounds if pubchem_compounds else state["compounds"]

    enriched = sum(1 for c in pubchem_compounds if c.pubmed_ids)
    logger.info("Merge complete: %d / %d PubChem compound(s) enriched with PubMed data",
                enriched, len(pubchem_compounds))

    return state


def aggregate(state: ExtractionState) -> ExtractionState:
    filled_target    = sum(1 for c in state["compounds"] if c.target_protein)
    filled_mechanism = sum(1 for c in state["compounds"] if c.mechanism_of_action)
    filled_disease   = sum(1 for c in state["compounds"] if c.disease_indication)
    filled_activity  = sum(1 for c in state["compounds"] if c.activity_value)

    logger.info(
        "ExtractionAgent complete: %d compound(s) | "
        "target: %d | mechanism: %d | disease: %d | activity: %d | errors: %d",
        len(state["compounds"]),
        filled_target, filled_mechanism, filled_disease, filled_activity,
        len(state["errors"]),
    )
    return state


def _build_graph() -> Any:
    g = StateGraph(ExtractionState)
    g.add_node("prepare",         prepare)
    g.add_node("extract_records", extract_records)
    g.add_node("validate",        validate)
    g.add_node("merge_pubmed",    merge_pubmed)
    g.add_node("aggregate",       aggregate)

    g.set_entry_point("prepare")
    g.add_edge("prepare",         "extract_records")
    g.add_edge("extract_records", "validate")
    g.add_edge("validate",        "merge_pubmed")
    g.add_edge("merge_pubmed",    "aggregate")
    g.add_edge("aggregate",       END)
    return g.compile()


def _extract_one(
    client: anthropic.Anthropic,
    record: RetrievedRecord,
    system: str,
) -> ExtractedCompound:
    prompt = _USER_TEMPLATE.format(
        source=record.source.value,
        record_id=record.record_id,
        title=record.title or "(no title)",
        metadata=json.dumps(record.metadata, indent=2),
        abstract=record.abstract or "(no abstract)",
    )

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=600,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    data = json.loads(raw_text)

    return ExtractedCompound(
        molecule_name=data.get("molecule_name"),
        chemical_formula=data.get("chemical_formula"),
        smiles=data.get("smiles"),
        target_protein=data.get("target_protein"),
        activity_value=_safe_float(data.get("activity_value")),
        activity_units=data.get("activity_units"),
        activity_type=data.get("activity_type"),
        mechanism_of_action=data.get("mechanism_of_action"),
        disease_indication=data.get("disease_indication"),
        confidence=float(data.get("confidence", 1.0)),
        notes=data.get("notes"),
        source_id=record.record_id,
        source=record.source.value,
    )


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


class ExtractionAgent:
    """
    Extracts structured chemical data from PubChem + PubMed records.
    """
    def __init__(self) -> None:
        self._graph = _build_graph()

    def run(self, query: str, records: list[RetrievedRecord]) -> ExtractionResult:
        final = self._graph.invoke({
            "query":     query,
            "records":   records,
            "compounds": [],
            "errors":    {},
            "client":    None,
        })
        return ExtractionResult(
            query=query,
            compounds=final["compounds"],
            errors=final["errors"],
        )
