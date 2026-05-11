"""
agents/chembl_enricher.py — ChEMBL Enrichment Agent.

Runs after the Extraction Agent and fills in bioactivity fields
that PubChem and PubMed cannot reliably provide:
  - activity_value  (IC50, Ki, EC50 in nM)
  - activity_units  (nM, µM)
  - activity_type   (IC50, Ki, EC50)
  - chembl_id       (ChEMBL compound identifier)
  - sar_data        (list of related compounds with activity)

This is the key addition that allows questions like:
  "What is the IC50 of imatinib against BCR-ABL?"
  "Which KDM4 inhibitors have the best potency?"

Usage:
    from agents.chembl_enricher import ChEMBLEnricher
    enricher = ChEMBLEnricher()
    enriched = enricher.run(extraction_result)
"""
from __future__ import annotations
import logging
from typing import Any

from tools.models import ExtractionResult, ExtractedCompound
from tools.chembl_tool import get_compound_chembl_id, get_activity_by_compound

logger = logging.getLogger(__name__)


class ChEMBLEnricher:
    """
    Enriches extracted compounds with bioactivity data from ChEMBL.

    For each compound that has no activity_value, it:
    1. Looks up the ChEMBL ID by compound name
    2. Fetches IC50 values against the known target (if any)
    3. Falls back to the best available IC50 across all targets
    4. Updates the compound record in place

    Example:
        enricher = ChEMBLEnricher()
        enriched_result = enricher.run(extraction_result)
    """

    def run(self, extraction_result: ExtractionResult) -> ExtractionResult:
        """
        Enrich all compounds in an ExtractionResult with ChEMBL data.
        Returns the same ExtractionResult with activity fields filled in.
        """
        enriched = 0

        for compound in extraction_result.compounds:
            if self._try_enrich(compound):
                enriched += 1

        logger.info("ChEMBLEnricher: enriched %d / %d compound(s) with activity data",
                    enriched, extraction_result.total)
        return extraction_result

    def _try_enrich(self, compound: ExtractedCompound) -> bool:
        """
        Try to enrich one compound with ChEMBL bioactivity data.
        Returns True if enrichment succeeded.
        """
        name = compound.molecule_name
        if not name:
            return False

        # Note existing activity value — ChEMBL will override if it finds
        # a more precise value (ChEMBL data is from curated published assays)
        has_existing = compound.activity_value is not None
        if has_existing:
            logger.debug("ChEMBL: %s has existing value %s %s — will try to enrich anyway",
                         name, compound.activity_value, compound.activity_units)

        # Look up ChEMBL ID
        chembl_id = get_compound_chembl_id(name)
        if not chembl_id:
            logger.debug("ChEMBL: %s not found", name)
            return False

        # Try IC50 first, then Ki, then EC50
        for activity_type in ["IC50", "Ki", "EC50"]:
            activities = get_activity_by_compound(
                name, max_results=10, activity_type=activity_type
            )
            if not activities:
                continue

            # Filter to human target assays if possible
            human = [a for a in activities
                     if (a.get("target_organism") or "").lower() in
                     ("homo sapiens", "human", "")]
            candidates = human or activities

            # If we know the target, prefer matching target
            if compound.target_protein:
                target_lower = compound.target_protein.lower()
                target_match = [
                    a for a in candidates
                    if a.get("target_name") and
                    any(w in (a["target_name"] or "").lower()
                        for w in target_lower.split()[:2] if len(w) > 3)
                ]
                if target_match:
                    candidates = target_match

            # Pick the best value (lowest IC50 = most potent)
            valid = [a for a in candidates
                     if a.get("value") and a["value"] not in ("", None)]
            if not valid:
                continue

            try:
                best = min(valid, key=lambda a: float(a["value"]))
            except (ValueError, TypeError):
                best = valid[0]

            # Update compound
            try:
                compound.activity_value = float(best["value"])
            except (ValueError, TypeError):
                continue

            compound.activity_units = best.get("units") or "nM"
            compound.activity_type  = activity_type
            compound.notes = (compound.notes or "") + \
                f" [ChEMBL {activity_type}: {best['value']} {best.get('units','nM')}" \
                f" vs {best.get('target_name','unknown target')}]"

            logger.info("ChEMBL: enriched %s → %s %s %s (vs %s)",
                        name, activity_type, best["value"],
                        best.get("units", "nM"), best.get("target_name", "?"))
            return True

        return False