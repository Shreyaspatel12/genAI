"""
tools/pubchem_tool.py — PubChem PUG REST API wrapper.

Endpoints used:
  - Compound search by name/SMILES/formula  →  /compound/name/{query}/JSON
  - Full compound property table            →  /compound/cid/{cid}/property/.../JSON
  - Bioassay activity data                  →  /compound/cid/{cid}/assaysummary/JSON

No API key required. Rate limit: 5 requests/sec.
Docs: https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest
"""
from __future__ import annotations
import logging
from typing import Any

from config import PUBCHEM_BASE
from tools.http_client import get_session, safe_get
from tools.models import DataSource, RetrievedRecord

logger = logging.getLogger(__name__)

# Properties to fetch for each compound
# Note: PubChem returns "SMILES" and "ConnectivitySMILES" — NOT IsomericSMILES/CanonicalSMILES
_PROPERTIES = ",".join([
    "MolecularFormula",
    "MolecularWeight",
    "IUPACName",
    "SMILES",
    "ConnectivitySMILES",
    "InChIKey",
    "XLogP",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "TPSA",
])


def search_compounds(query: str, max_results: int = 10) -> list[RetrievedRecord]:
    """
    Search PubChem for compounds matching a name, SMILES, or formula string.
    Returns a list of RetrievedRecord with structured chemical properties.
    """
    session = get_session()
    records: list[RetrievedRecord] = []

    # Step 1 — get matching CIDs
    cids = _fetch_cids(query, max_results, session)
    if not cids:
        logger.info("PubChem: no CIDs found for query '%s'", query)
        return records

    logger.info("PubChem: found %d CID(s) for '%s'", len(cids), query)

    # Step 2 — fetch property table for all CIDs in one request
    props = _fetch_properties(cids, session)

    # Step 3 — build normalised records
    for cid_str, prop_data in props.items():
        cid = str(cid_str)
        record = RetrievedRecord(
            source=DataSource.PUBCHEM,
            record_id=cid,
            title=prop_data.get("IUPACName") or f"CID {cid}",
            url=f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            raw=prop_data,
            metadata={
                "molecular_formula": prop_data.get("MolecularFormula"),
                "molecular_weight":  prop_data.get("MolecularWeight"),
                "smiles":            prop_data.get("SMILES") or prop_data.get("ConnectivitySMILES"),
                "inchikey":          prop_data.get("InChIKey"),
                "xlogp":             prop_data.get("XLogP"),
                "hbd":               prop_data.get("HBondDonorCount"),
                "hba":               prop_data.get("HBondAcceptorCount"),
                "tpsa":              prop_data.get("TPSA"),
            },
        )
        records.append(record)

    return records


# ── private helpers ────────────────────────────────────────────────────────────

def _fetch_cids(query: str, max_results: int, session) -> list[str]:
    """Return up to max_results CIDs for a name search."""
    url = f"{PUBCHEM_BASE}/compound/name/{query}/cids/JSON"
    try:
        data = safe_get(url, session=session)
        cids = data.get("IdentifierList", {}).get("CID", [])
        return [str(c) for c in cids[:max_results]]
    except RuntimeError as e:
        # Try formula fallback
        logger.debug("PubChem name search failed (%s), trying formula search", e)
        try:
            url2 = f"{PUBCHEM_BASE}/compound/formula/{query}/cids/JSON"
            data2 = safe_get(url2, session=session)
            cids2 = data2.get("IdentifierList", {}).get("CID", [])
            return [str(c) for c in cids2[:max_results]]
        except RuntimeError:
            return []


def _fetch_properties(cids: list[str], session) -> dict[str, dict[str, Any]]:
    """
    Batch-fetch compound properties for multiple CIDs.
    Uses POST instead of GET to avoid URL length limits when cids > 1.
    Returns {cid: {prop: value}}.
    """
    url = f"{PUBCHEM_BASE}/compound/cid/property/{_PROPERTIES}/JSON"
    try:
        # POST sends CIDs in the body — works reliably for any number of CIDs
        resp = session.post(
            url,
            data={"cid": ",".join(cids)},
            timeout=15,
        )
        resp.raise_for_status()
        data  = resp.json()
        props = data.get("PropertyTable", {}).get("Properties", [])
        return {str(p["CID"]): p for p in props}
    except Exception as e:
        logger.warning("PubChem property fetch failed: %s", e)
        return {}
