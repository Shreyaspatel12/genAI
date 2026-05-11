"""
agents/structural_agent.py — ChemCrow-style Structural Chemistry Agent.

Handles structural chemistry questions that the standard Reasoning Agent
cannot answer — specifically:
  - Structural motif analysis (does this compound have N-OH, C=S, etc.)
  - Structure-activity relationship (SAR) questions
  - SMILES-based property analysis across multiple compounds
  - Functional group identification

How it works:
  1. Fetches real SMILES from PubChem for known KDM4/target inhibitors
  2. Uses RDKit to detect structural motifs programmatically (no hallucination)
  3. Passes real structural analysis to Claude Opus for scientific explanation

This replaces ChemCrow for structural questions — same idea, no dependency conflicts.

Usage:
    agent  = StructuralAgent()
    result = agent.run(
        question="What structural motifs are effective in KDM4 inhibitors?",
        compounds=["ML324", "QC6352", "TACH107", "JIB-04"]
    )
    print(result["answer"])
"""
from __future__ import annotations
import logging
from typing import Any, Optional, TypedDict

import anthropic
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from langgraph.graph import StateGraph, END

from config import ANTHROPIC_API_KEY
from tools.pubchem_tool import search_compounds

# ── ChemMCP tools (extracted from OSU-NLP-Group/ChemMCP) ─────────────────────
from tools.chemmcp_tools import (
    functional_groups,       # identify all functional groups in a molecule
    functional_groups_dict,  # same but returns dict for programmatic use
    smiles_check,            # validate a SMILES string
    molecule_weight,         # calculate exact molecular weight
    tanimoto_similarity,     # Tanimoto similarity between two molecules
    smiles_to_formula,       # convert SMILES to molecular formula
    name_to_smiles,          # convert compound name to SMILES via PubChem
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Structural motif definitions — SMARTS patterns
# SMARTS is a language for describing chemical substructures
# Each pattern is checked against each compound's SMILES
# ══════════════════════════════════════════════════════════════════════════════

MOTIFS = {
    # The motifs the chemist asked about
    "carbonyl_adjacent_COOH":  "[CX3](=O)[CX3](=O)[OH]",   # C=O next to COOH
    "thioketone_CS":           "[CX3]=[SX1]",               # C=S group
    "long_carbon_chain_5plus": "[CH2][CH2][CH2][CH2][CH2]", # 5+ carbon chain
    "sulfonate_ester_SO2OR":   "[SX4](=O)(=O)[OX2][#6]",   # SO2-O-R
    "hydroxylamine_NOH":       "[NX3][OX2H]",               # N-OH group

    # Additional common KDM4 inhibitor motifs
    "hydroxamic_acid":         "[CX3](=O)[NX3][OX2H]",     # C(=O)N-OH — iron chelator
    "carboxylic_acid":         "[CX3](=O)[OX2H1]",          # COOH
    "pyridine":                "n1ccccc1",                   # pyridine ring
    "quinoline":               "c1ccc2ncccc2c1",             # quinoline
    "iron_chelator_NOO":       "[NX2]=[CX3]-[OX2H]",       # N=C-OH chelator
    "triazole":                "c1nnn[nH]1",                 # triazole ring
    "aromatic_amine":          "[NX3H][c]",                  # NH on aromatic
}


# ══════════════════════════════════════════════════════════════════════════════
# LangGraph state
# ══════════════════════════════════════════════════════════════════════════════

class StructuralState(TypedDict):
    question:          str
    compound_names:    list[str]
    smiles_data:       dict[str, str]        # name → SMILES
    properties:        dict[str, dict]       # name → property dict
    motif_results:     dict[str, dict]       # name → {motif: bool}
    context:           str                   # formatted context for Claude
    answer:            str
    client:            Any
    errors:            list[str]


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 — Fetch SMILES from PubChem for each compound
# ══════════════════════════════════════════════════════════════════════════════

def fetch_structures(state: StructuralState) -> StructuralState:
    """
    For each compound name, search PubChem and retrieve its SMILES.
    This gives us real, validated structures — not hallucinated ones.
    """
    logger.info("=" * 55)
    logger.info("  StructuralAgent starting — ChemMCP tools active")
    logger.info("  Tools: Name2Smiles | MoleculeSmilesCheck |")
    logger.info("         FunctionalGroups | MoleculeWeight  |")
    logger.info("         MoleculeSimilarity | Smiles2Formula ")
    logger.info("=" * 55)
    smiles_data = {}
    errors      = list(state["errors"])

    for name in state["compound_names"]:
        try:
            # Try ChemMCP name_to_smiles first (uses PubChem isomeric SMILES)
            smiles = name_to_smiles(name)
            if smiles:
                # Validate with ChemMCP smiles_check before accepting
                validity = smiles_check(smiles)
                if validity == "valid":
                    smiles_data[name] = smiles
                    logger.info("  [ChemMCP] %s → %s", name, smiles[:50])
                else:
                    errors.append(f"SMILES for {name} failed validation: {validity}")
            else:
                # Fallback to existing search_compounds
                records = search_compounds(name, max_results=1)
                if records:
                    smiles = records[0].metadata.get("smiles")
                    if smiles and smiles_check(smiles) == "valid":
                        smiles_data[name] = smiles
                        logger.info("  [PubChem fallback] %s → %s", name, smiles[:50])
                    else:
                        errors.append(f"No valid SMILES for {name}")
                else:
                    errors.append(f"{name} not found in PubChem")
        except Exception as e:
            errors.append(f"Fetch failed for {name}: {e}")
            logger.error("  Failed to fetch %s: %s", name, e)

    state["smiles_data"] = smiles_data
    state["errors"]      = errors
    logger.info("fetch_structures: got SMILES for %d / %d compounds",
                len(smiles_data), len(state["compound_names"]))
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 — Compute properties with RDKit
# ══════════════════════════════════════════════════════════════════════════════

def compute_properties(state: StructuralState) -> StructuralState:
    """
    Use RDKit to compute key physicochemical properties for each compound.
    These are calculated directly from the SMILES — 100% accurate.
    """
    properties = {}

    for name, smiles in state["smiles_data"].items():
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            state["errors"].append(f"RDKit could not parse SMILES for {name}")
            continue

        props = {
            "molecular_weight":    round(Descriptors.MolWt(mol), 2),
            "logp":                round(Descriptors.MolLogP(mol), 2),
            "hbd":                 rdMolDescriptors.CalcNumHBD(mol),   # H-bond donors
            "hba":                 rdMolDescriptors.CalcNumHBA(mol),   # H-bond acceptors
            "tpsa":                round(Descriptors.TPSA(mol), 2),
            "rotatable_bonds":     rdMolDescriptors.CalcNumRotatableBonds(mol),
            "aromatic_rings":      rdMolDescriptors.CalcNumAromaticRings(mol),
            "heavy_atom_count":    mol.GetNumHeavyAtoms(),
            "smiles":              smiles,
        }
        properties[name] = props
        logger.info("  Properties for %s: MW=%.1f, LogP=%.2f, TPSA=%.1f",
                    name, props["molecular_weight"], props["logp"], props["tpsa"])

    state["properties"] = properties
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 3 — Detect structural motifs with RDKit SMARTS matching
# ══════════════════════════════════════════════════════════════════════════════

def detect_motifs(state: StructuralState) -> StructuralState:
    """
    Use ChemMCP functional_groups_dict (RDKit SMARTS) to detect all
    functional groups and structural motifs in each compound.
    100% programmatic — no AI guessing, no hallucination.
    """
    motif_results = {}

    for name, smiles in state["smiles_data"].items():
        # ChemMCP tool: returns dict of {group_name: True/False} for all groups
        results = functional_groups_dict(smiles)

        # Also run the human-readable version for the context block
        fg_description = functional_groups(smiles)

        # Store both
        motif_results[name] = results
        motif_results[f"{name}__description"] = fg_description

        found = [m for m, v in results.items() if v]
        logger.info("  [ChemMCP] Motifs in %s: %s", name, found[:5] or "none")

    state["motif_results"] = motif_results
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 — Build context for Claude
# ══════════════════════════════════════════════════════════════════════════════

def build_context(state: StructuralState) -> StructuralState:
    """
    Format all the real structural data into a clean context block
    that Claude can reason over.
    """
    lines = []
    lines.append(f"QUESTION: {state['question']}\n")
    lines.append("=" * 60)
    lines.append("REAL STRUCTURAL DATA (computed by RDKit from PubChem SMILES)\n")

    for name in state["compound_names"]:
        if name not in state["smiles_data"]:
            lines.append(f"\n{name}: NOT FOUND IN PUBCHEM")
            continue

        lines.append(f"\n{'─'*40}")
        lines.append(f"COMPOUND: {name}")
        lines.append(f"SMILES: {state['smiles_data'][name]}")

        # Properties
        if name in state["properties"]:
            p = state["properties"][name]
            lines.append(f"MW: {p['molecular_weight']}  |  LogP: {p['logp']}  |  "
                        f"TPSA: {p['tpsa']}  |  HBD: {p['hbd']}  |  HBA: {p['hba']}  |  "
                        f"Aromatic rings: {p['aromatic_rings']}")

        # Motifs — ChemMCP human-readable description
        fg_desc_key = f"{name}__description"
        if fg_desc_key in state["motif_results"]:
            lines.append(f"FUNCTIONAL GROUPS (ChemMCP): {state['motif_results'][fg_desc_key]}")

        # Detailed motif table
        if name in state["motif_results"]:
            present_groups = [m for m, v in state["motif_results"][name].items()
                              if v and not m.startswith("__")]
            absent_asked = [m for m, v in state["motif_results"][name].items()
                            if not v and any(kw in m for kw in
                            ["hydroxyl", "carbonyl", "thio", "sulfo", "amine",
                             "carboxyl", "hydroxamic", "pyridine", "N-OH"])]
            if present_groups:
                lines.append(f"PRESENT GROUPS: {', '.join(present_groups)}")
            if absent_asked:
                lines.append(f"KEY ABSENT GROUPS: {', '.join(absent_asked)}")

    # Tanimoto similarity between all compound pairs (ChemMCP tool)
    names_with_smiles = [n for n in state["compound_names"] if n in state["smiles_data"]]
    if len(names_with_smiles) >= 2:
        lines.append(f"\n{'─'*40}")
        lines.append("PAIRWISE SIMILARITY (ChemMCP Tanimoto):")
        for i in range(len(names_with_smiles)):
            for j in range(i+1, len(names_with_smiles)):
                n1, n2 = names_with_smiles[i], names_with_smiles[j]
                sim = tanimoto_similarity(state["smiles_data"][n1], state["smiles_data"][n2])
                lines.append(f"  {n1} vs {n2}: {sim}")

    if state["errors"]:
        lines.append(f"\nNOTES: {'; '.join(state['errors'])}")

    state["context"] = "\n".join(lines)
    return state


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 — Claude generates scientific answer from real data
# ══════════════════════════════════════════════════════════════════════════════

_SYSTEM = """You are an expert medicinal chemist specialising in structure-activity relationships.

You are given real structural data computed programmatically by RDKit from PubChem SMILES strings.
Every motif detection result is accurate — generated by SMARTS pattern matching, not guessing.

Your job:
1. Analyse the motif data to answer the question
2. Explain WHY each motif matters chemically (e.g. iron chelation, H-bonding, membrane permeability)
3. Identify which motifs are consistently present across compounds — these are likely important
4. Be honest if data is limited — say so clearly

Rules:
- Only reason over the data provided — never invent structures or IC50 values
- If a compound was not found in PubChem, note this and explain the limitation
- Keep the answer focused and scientifically precise
"""

def generate_answer(state: StructuralState) -> StructuralState:
    """Send real structural data to Claude Opus for scientific interpretation."""
    client = state["client"]

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            system=_SYSTEM,
            messages=[{"role": "user", "content": state["context"]}],
        )
        state["answer"] = response.content[0].text.strip()
        logger.info("StructuralAgent: generated %d word answer",
                    len(state["answer"].split()))
    except Exception as e:
        state["answer"] = f"Answer generation failed: {e}"
        logger.error("StructuralAgent answer generation failed: %s", e)

    return state


# ══════════════════════════════════════════════════════════════════════════════
# Graph assembly
# ══════════════════════════════════════════════════════════════════════════════

def _build_graph() -> Any:
    g = StateGraph(StructuralState)
    for name, fn in [
        ("fetch",      fetch_structures),
        ("properties", compute_properties),
        ("motifs",     detect_motifs),
        ("context",    build_context),
        ("answer",     generate_answer),
    ]:
        g.add_node(name, fn)

    g.set_entry_point("fetch")
    g.add_edge("fetch",      "properties")
    g.add_edge("properties", "motifs")
    g.add_edge("motifs",     "context")
    g.add_edge("context",    "answer")
    g.add_edge("answer",     END)
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

class StructuralAgent:
    """
    ChemCrow-style structural chemistry agent.
    Uses RDKit for motif detection + Claude Opus for scientific interpretation.

    Example:
        agent  = StructuralAgent()
        result = agent.run(
            question="What structural motifs are effective in KDM4 inhibitors?",
            compounds=["ML324", "QC6352", "JIB-04"]
        )
        print(result["answer"])
        print(result["motif_results"])
    """
    def __init__(self) -> None:
        self._graph  = _build_graph()
        self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    def run(
        self,
        question:  str,
        compounds: list[str],
    ) -> dict[str, Any]:

        final = self._graph.invoke({
            "question":       question,
            "compound_names": compounds,
            "smiles_data":    {},
            "properties":     {},
            "motif_results":  {},
            "context":        "",
            "answer":         "",
            "client":         self._client,
            "errors":         [],
        })

        return {
            "question":      final["question"],
            "answer":        final["answer"],
            "smiles_data":   final["smiles_data"],
            "properties":    final["properties"],
            "motif_results": final["motif_results"],
            "errors":        final["errors"],
        }