"""
tools/pubmed_tool.py — PubMed E-utilities API wrapper.

Endpoints used:
  - esearch  →  find PMIDs matching a query
  - efetch   →  retrieve full abstracts for those PMIDs

Free to use. No API key required but registering for one at
ncbi.nlm.nih.gov/account raises rate limit from 3 → 10 req/sec.
Set PUBMED_API_KEY in your .env file if you have one.

Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""
from __future__ import annotations
import logging
import xml.etree.ElementTree as ET
from typing import Optional

from config import PUBMED_BASE, PUBMED_API_KEY
from tools.http_client import get_session, safe_get
from tools.models import DataSource, RetrievedRecord

logger = logging.getLogger(__name__)

_DB = "pubmed"


def search_articles(
    query:       str,
    max_results: int = 10,
    date_from:   Optional[str] = None,   # "YYYY/MM/DD"
    date_to:     Optional[str] = None,
) -> list[RetrievedRecord]:
    """
    Search PubMed for articles matching a query string.
    Returns a list of RetrievedRecord with title + abstract.

    Example:
        records = search_articles("Carboplatin mechanism activity", max_results=10)
    """
    session = get_session()

    # Step 1 — esearch: find matching PMIDs
    pmids = _esearch(query, max_results, date_from, date_to, session)
    if not pmids:
        logger.info("PubMed: no results for query '%s'", query)
        return []

    logger.info("PubMed: found %d PMID(s) for '%s'", len(pmids), query)

    # Step 2 — efetch: fetch full records for those PMIDs
    return _efetch(pmids, session)


# ── private helpers ────────────────────────────────────────────────────────────

def _base_params(extra: dict | None = None) -> dict:
    """Build base query params, injecting API key if available."""
    params = {"db": _DB, "retmode": "json", "retmax": 0}
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY
    if extra:
        params.update(extra)
    return params


def _esearch(
    query:       str,
    max_results: int,
    date_from:   Optional[str],
    date_to:     Optional[str],
    session,
) -> list[str]:
    """Call esearch to get list of PMIDs for a query."""
    params = _base_params({
        "term":       query,
        "retmax":     max_results,
        "usehistory": "n",
    })
    if date_from:
        params["mindate"] = date_from.replace("-", "/")
    if date_to:
        params["maxdate"] = date_to.replace("-", "/")
    if date_from or date_to:
        params["datetype"] = "pdat"

    try:
        data = safe_get(f"{PUBMED_BASE}/esearch.fcgi", params=params, session=session)
        return data.get("esearchresult", {}).get("idlist", [])
    except RuntimeError as e:
        logger.error("PubMed esearch failed: %s", e)
        return []


def _efetch(pmids: list[str], session) -> list[RetrievedRecord]:
    """Fetch full XML records for a list of PMIDs and parse into RetrievedRecords."""
    params: dict = {
        "db":      _DB,
        "id":      ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if PUBMED_API_KEY:
        params["api_key"] = PUBMED_API_KEY

    try:
        from config import REQUEST_TIMEOUT
        resp = session.get(
            f"{PUBMED_BASE}/efetch.fcgi",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_pubmed_xml(resp.text)
    except Exception as e:
        logger.error("PubMed efetch failed: %s", e)
        return []


def _parse_pubmed_xml(xml_text: str) -> list[RetrievedRecord]:
    """Parse PubMed XML response into a list of RetrievedRecord objects."""
    records: list[RetrievedRecord] = []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.error("PubMed XML parse error: %s", e)
        return records

    for article in root.findall(".//PubmedArticle"):

        # PMID
        pmid_el = article.find(".//PMID")
        pmid    = pmid_el.text if pmid_el is not None else "unknown"

        # Title
        title_el = article.find(".//ArticleTitle")
        title    = "".join(title_el.itertext()) if title_el is not None else None

        # Abstract (may have multiple sections e.g. Background, Methods, Results)
        abstract_parts = article.findall(".//AbstractText")
        abstract = " ".join(
            "".join(p.itertext()) for p in abstract_parts
        ).strip() or None

        # Authors
        authors = [
            f"{a.findtext('LastName', '')} {a.findtext('ForeName', '')}".strip()
            for a in article.findall(".//Author")
            if a.findtext("LastName")
        ]

        # Journal + year
        journal = (
            article.findtext(".//Journal/Title") or
            article.findtext(".//ISOAbbreviation")
        )
        year = (
            article.findtext(".//PubDate/Year") or
            article.findtext(".//PubDate/MedlineDate", "")[:4]
        )

        # MeSH terms (controlled vocabulary — useful for target/disease matching)
        mesh = [
            m.text for m in article.findall(".//DescriptorName") if m.text
        ]

        records.append(RetrievedRecord(
            source    = DataSource.PUBMED,
            record_id = pmid,
            title     = title,
            abstract  = abstract,
            url       = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            raw       = {"pmid": pmid, "title": title, "abstract": abstract},
            metadata  = {
                "authors": authors,
                "journal": journal,
                "year":    year,
                "mesh":    mesh,
            },
        ))

    logger.info("PubMed: parsed %d article(s) from XML", len(records))
    return records
