"""
config.py — Centralised settings loader.
All agents import from here; never read os.environ directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY:     str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL:       str = os.getenv("OPENAI_MODEL", "gpt-4o")
ANTHROPIC_API_KEY:  str = os.getenv("ANTHROPIC_API_KEY", "")

# ── API base URLs ─────────────────────────────────────────────────────────────
PUBCHEM_BASE:  str = os.getenv("PUBCHEM_BASE",  "https://pubchem.ncbi.nlm.nih.gov/rest/pug")
OPENFDA_BASE:  str = os.getenv("OPENFDA_BASE",  "https://api.fda.gov")
PUBMED_BASE:   str = os.getenv("PUBMED_BASE",   "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
BIORXIV_BASE:  str = os.getenv("BIORXIV_BASE",  "https://api.biorxiv.org")

# ── Optional API keys ─────────────────────────────────────────────────────────
PUBMED_API_KEY:   str = os.getenv("PUBMED_API_KEY",   "")
DRUGBANK_API_KEY: str = os.getenv("DRUGBANK_API_KEY", "")

# ── Retrieval defaults ────────────────────────────────────────────────────────
DEFAULT_MAX_RESULTS: int = int(os.getenv("DEFAULT_MAX_RESULTS", "10"))
REQUEST_TIMEOUT:     int = int(os.getenv("REQUEST_TIMEOUT", "15"))
