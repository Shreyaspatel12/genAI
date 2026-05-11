"""
tools/models.py — Shared Pydantic models used across all agents.
Updated to include PubMed-specific fields:
  - mechanism_of_action
  - disease_indication
  - pubmed_ids (which papers support this record)
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class DataSource(str, Enum):
    PUBCHEM  = "pubchem"
    PUBMED   = "pubmed"
    OPENFDA  = "openfda"
    BIORXIV  = "biorxiv"


class RetrievedRecord(BaseModel):
    source:    DataSource
    record_id: str
    title:     Optional[str]       = None
    abstract:  Optional[str]       = None
    raw:       dict[str, Any]      = Field(default_factory=dict)
    url:       Optional[str]       = None
    metadata:  dict[str, Any]      = Field(default_factory=dict)


class RetrievalQuery(BaseModel):
    query:       str
    max_results: int               = 10
    sources:     list[DataSource]  = Field(default_factory=lambda: list(DataSource))
    date_from:   Optional[str]     = None
    date_to:     Optional[str]     = None


class RetrievalResult(BaseModel):
    query:   str
    records: list[RetrievedRecord] = Field(default_factory=list)
    errors:  dict[str, str]        = Field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.records)


class ExtractedCompound(BaseModel):
    """
    Structured chemical record produced by the Extraction Agent.

    NEW FIELDS populated by PubMed:
      mechanism_of_action  — how the drug works at the molecular level
      disease_indication   — what disease/condition it treats
      pubmed_ids           — list of PMIDs that support this record
    """
    # Identity
    molecule_name:       Optional[str]   = Field(None)
    chemical_formula:    Optional[str]   = Field(None)
    smiles:              Optional[str]   = Field(None)

    # Bioactivity
    target_protein:      Optional[str]   = Field(None)
    activity_value:      Optional[float] = Field(None)
    activity_units:      Optional[str]   = Field(None)
    activity_type:       Optional[str]   = Field(None)

    # NEW: Biology & Clinical (from PubMed)
    mechanism_of_action: Optional[str]   = Field(None)
    disease_indication:  Optional[str]   = Field(None)
    pubmed_ids:          list[str]       = Field(default_factory=list)

    # Provenance
    source_id:           str
    source:              str
    confidence:          float           = Field(default=1.0, ge=0.0, le=1.0)
    notes:               Optional[str]   = Field(None)


class ExtractionResult(BaseModel):
    query:     str
    compounds: list[ExtractedCompound] = Field(default_factory=list)
    errors:    dict[str, str]          = Field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.compounds)
