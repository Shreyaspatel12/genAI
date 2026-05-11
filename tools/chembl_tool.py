"""
tools/chembl_tool.py — ChEMBL REST API tool for bioactivity data.

ChEMBL is a large open-access bioactivity database containing IC50, Ki, EC50
and other activity measurements for compounds against biological targets.

This fills the biggest gap in the pipeline — PubChem has structure data but
no bioassay measurements. ChEMBL has millions of measured activity values.

No API key required. Free and open.
Base URL: https://www.ebi.ac.uk/chembl/api/data/

Functions:
  get_activity_by_compound(name, max_results)  → IC50/Ki values for a compound
  get_activity_by_target(target_name, max)     → all compounds tested on a target
  get_compound_chembl_id(name)                 → look up ChEMBL ID for a compound
  get_sar_table(target_name, max_results)      → SAR table: compound + IC50 + SMILES

Usage:
    from tools.chembl_tool import get_activity_by_compound, get_sar_table

    # Get all IC50 values for imatinib
    activities = get_activity_by_compound("imatinib", max_results=10)

    # Get SAR table for BCR-ABL
    sar = get_sar_table("BCR-ABL", max_results=20)
"""
from __future__ import annotations
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
TIMEOUT = 15


# ══════════════════════════════════════════════════════════════════════════════
# Helper — raw GET request to ChEMBL REST API
# ══════════════════════════════════════════════════════════════════════════════

def _chembl_get(endpoint: str, params: dict) -> Optional[dict]:
    """Make a GET request to ChEMBL REST API and return JSON."""
    params["format"] = "json"
    try:
        resp = requests.get(f"{CHEMBL_BASE}/{endpoint}", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("[ChEMBL] Request failed for %s: %s", endpoint, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Function 1 — Get ChEMBL ID for a compound name
# ══════════════════════════════════════════════════════════════════════════════

def get_compound_chembl_id(name: str) -> Optional[str]:
    """
    Look up the ChEMBL ID for a compound by name.

    Example:
        get_compound_chembl_id("imatinib") → "CHEMBL941"
    """
    logger.info("[ChEMBL:Molecule] looking up: %s", name)

    # Try preferred name first
    data = _chembl_get("molecule", {
        "pref_name__iexact": name,
        "limit": 1,
    })
    if data and data.get("molecules"):
        cid = data["molecules"][0]["molecule_chembl_id"]
        logger.info("[ChEMBL:Molecule] found: %s → %s", name, cid)
        return cid

    # Try synonym search
    data = _chembl_get("molecule", {
        "molecule_synonyms__molecule_synonym__iexact": name,
        "limit": 1,
    })
    if data and data.get("molecules"):
        cid = data["molecules"][0]["molecule_chembl_id"]
        logger.info("[ChEMBL:Molecule] found via synonym: %s → %s", name, cid)
        return cid

    logger.warning("[ChEMBL:Molecule] not found: %s", name)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Function 2 — Get bioactivity data for a compound
# ══════════════════════════════════════════════════════════════════════════════

def get_activity_by_compound(
    name: str,
    max_results: int = 10,
    activity_type: str = "IC50",
) -> list[dict]:
    """
    Get bioactivity measurements for a compound from ChEMBL.

    Returns a list of activity records, each containing:
      - target_name     : what the compound was tested against
      - activity_type   : IC50, Ki, EC50, etc.
      - value           : numeric value
      - units           : nM, µM, etc.
      - assay_description
      - document_chembl_id : source paper

    Example:
        activities = get_activity_by_compound("imatinib", max_results=5)
        for a in activities:
            print(f"{a['target_name']}: {a['value']} {a['units']}")
    """
    logger.info("[ChEMBL:Activity] fetching %s data for: %s", activity_type, name)

    chembl_id = get_compound_chembl_id(name)
    if not chembl_id:
        return []

    data = _chembl_get("activity", {
        "molecule_chembl_id": chembl_id,
        "standard_type":      activity_type,
        "limit":              max_results,
    })

    if not data or not data.get("activities"):
        logger.warning("[ChEMBL:Activity] no %s data for %s (%s)", activity_type, name, chembl_id)
        return []

    results = []
    for a in data["activities"]:
        results.append({
            "compound":          name,
            "chembl_id":         chembl_id,
            "target_name":       a.get("target_pref_name") or a.get("target_chembl_id"),
            "target_organism":   a.get("target_organism"),
            "activity_type":     a.get("standard_type"),
            "value":             a.get("standard_value"),
            "units":             a.get("standard_units"),
            "relation":          a.get("standard_relation"),  # =, <, >
            "assay_description": a.get("assay_description"),
            "assay_type":        a.get("assay_type"),
            "document":          a.get("document_chembl_id"),
        })

    logger.info("[ChEMBL:Activity] found %d %s record(s) for %s",
                len(results), activity_type, name)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Function 3 — Get SAR table for a target
# Returns: compound + IC50 + SMILES — perfect for SAR analysis
# ══════════════════════════════════════════════════════════════════════════════

def get_sar_table(
    target_name: str,
    max_results: int = 20,
    activity_type: str = "IC50",
) -> list[dict]:
    """
    Get a SAR table for a biological target from ChEMBL.

    Returns a list of compounds tested against this target, each with:
      - compound_name   : preferred name
      - chembl_id       : ChEMBL compound ID
      - smiles          : canonical SMILES
      - activity_type   : IC50, Ki, etc.
      - value           : numeric value in nM
      - units           : standard units
      - assay_description

    Example:
        sar = get_sar_table("BCR-ABL", max_results=10)
        for row in sar:
            print(f"{row['compound_name']}: {row['value']} {row['units']}")
    """
    logger.info("[ChEMBL:SAR] building SAR table for target: %s", target_name)

    # Step 1 — find target ChEMBL ID
    # Try exact name first, then progressively broader searches
    target = None
    for search_params in [
        {"pref_name__iexact":    target_name, "limit": 1},
        {"pref_name__icontains": target_name, "limit": 3},
        {"pref_name__icontains": target_name.split()[0], "limit": 3},
    ]:
        data = _chembl_get("target", {**search_params, "target_type": "SINGLE PROTEIN"})
        if data and data.get("targets"):
            target = data["targets"][0]
            break

    # If still not found, try without target_type restriction
    if not target:
        data = _chembl_get("target", {
            "pref_name__icontains": target_name,
            "limit": 3,
        })
        if data and data.get("targets"):
            target = data["targets"][0]

    if not target:
        logger.warning("[ChEMBL:SAR] target not found: %s", target_name)
        return []

    target_id   = target["target_chembl_id"]
    target_full = target.get("pref_name", target_name)
    logger.info("[ChEMBL:SAR] target: %s (%s)", target_full, target_id)

    # Step 2 — get activities for this target
    data = _chembl_get("activity", {
        "target_chembl_id": target_id,
        "standard_type":    activity_type,
        "limit":            max_results,
    })

    if not data or not data.get("activities"):
        logger.warning("[ChEMBL:SAR] no %s data for target %s", activity_type, target_id)
        return []

    # Step 3 — enrich with SMILES for each unique compound
    seen = set()
    results = []
    for a in data["activities"]:
        mol_id = a.get("molecule_chembl_id")
        if mol_id in seen:
            continue
        seen.add(mol_id)

        # Fetch SMILES for this compound
        smiles = None
        mol_data = _chembl_get(f"molecule/{mol_id}", {})
        if mol_data:
            struct = mol_data.get("molecule_structures") or {}
            smiles = struct.get("canonical_smiles")

        results.append({
            "compound_name":    a.get("molecule_pref_name") or mol_id,
            "chembl_id":        mol_id,
            "smiles":           smiles,
            "target":           target_full,
            "target_id":        target_id,
            "activity_type":    a.get("standard_type"),
            "value":            a.get("standard_value"),
            "units":            a.get("standard_units"),
            "relation":         a.get("standard_relation"),
            "assay_description": a.get("assay_description"),
        })

    logger.info("[ChEMBL:SAR] built SAR table: %d compound(s) for %s",
                len(results), target_full)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Function 4 — Format SAR table as readable text for Claude
# ══════════════════════════════════════════════════════════════════════════════

def format_sar_for_llm(sar_table: list[dict], target_name: str) -> str:
    """
    Format a SAR table into a clean text block for Claude to reason over.
    """
    if not sar_table:
        return f"No SAR data found for {target_name} in ChEMBL."

    lines = [f"SAR DATA FROM ChEMBL — Target: {target_name}",
             f"Total compounds: {len(sar_table)}", "─" * 50]

    for i, row in enumerate(sar_table, 1):
        value_str = f"{row['relation'] or ''}{row['value']} {row['units']}" \
                    if row['value'] else "value not reported"
        lines.append(
            f"{i:2d}. {row['compound_name'] or row['chembl_id']:<30} "
            f"{row['activity_type']}: {value_str}"
        )
        if row.get("smiles"):
            lines.append(f"    SMILES: {row['smiles'][:60]}")

    return "\n".join(lines)