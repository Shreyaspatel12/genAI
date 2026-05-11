# ChemAgent Pipeline

An agentic AI pipeline that automatically retrieves, extracts, and reasons over chemical compound data from multiple scientific databases. Give it a compound name — it returns a complete structured record with molecular structure, biological target, mechanism of action, disease indication, and measured activity values.

---

## Architecture

```
Input: compound name
          │
          ▼
┌─────────────────────┐
│   Retriever Agent   │  ← PubChem (structure) + PubMed (papers)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Extraction Agent   │  ← Claude Opus reads records, pulls structured fields
└──────────┬──────────┘       PubChem + PubMed data merged into one record
           │
           ▼
┌─────────────────────┐
│  ChEMBL Enricher    │  ← Fills in real IC50/Ki values from curated bioassays
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Filter Agent      │  ← 4-layer quality gate (keyword → completeness →
└──────────┬──────────┘       confidence → LLM relevance score)
           │
           ▼
┌─────────────────────┐
│  Reasoning Agent    │  ← Answers scientific questions using real PubChem data
│  Structural Agent   │  ← Structural motif analysis using ChemMCP + RDKit
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Dataset Builder    │  ← Packages everything into a final JSON file
└─────────────────────┘
          │
          ▼
Output: outputs/dataset_<compound>_<timestamp>.json
```

---

## Data Sources

| Source | What it provides | Cost |
|--------|-----------------|------|
| PubChem | SMILES, molecular formula, physicochemical properties | Free |
| PubMed | Target protein, mechanism, disease (from paper abstracts) | Free |
| ChEMBL | Curated IC50/Ki activity values from published bioassays | Free |

---

## Output Fields

Each compound record contains:

```json
{
  "molecule_name": "Imatinib",
  "chemical_formula": "C29H31N7O",
  "smiles": "CC1=C(C=C(C=C1)NC(=O)...",
  "target_protein": "BCR-ABL tyrosine kinase",
  "mechanism_of_action": "Tyrosine kinase inhibitor (BCR-ABL, c-KIT, PDGFR)",
  "disease_indication": "Chronic myeloid leukemia (CML)",
  "activity_value": 38.0,
  "activity_units": "nM",
  "activity_type": "IC50",
  "pubmed_ids": ["42093005", "42080642"],
  "confidence": 0.95
}
```

---

## Installation

```bash
# Clone the repo
git clone https://github.com/Shreyaspatel12/genAI.git
cd genAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=your_key_here
```

---

## Running the Pipeline

### Basic usage — PubChem only

```bash
python pipeline.py --query "aspirin"
```

### With PubMed — populates target, mechanism, disease

```bash
python pipeline.py --query "aspirin" --pubmed
```

### With scientific questions

```bash
python pipeline.py --query "aspirin" --pubmed --questions "What is the mechanism of aspirin?"
```

---

## Examples

### Example 1 — Simple compound lookup

```bash
python pipeline.py --query "aspirin" --pubmed
```

**Output:**
```
STEP 1 · Retrieve     → 1 PubChem record, 3 PubMed papers
STEP 2 · Extract      → molecule: 2-acetyloxybenzoic acid | target: COX-1/COX-2 | confidence: 0.85
STEP 2b · ChEMBL      → activity: null (aspirin uses % inhibition not IC50)
STEP 3 · Filter       → kept 1/1 (score: 0.895)
STEP 5 · Dataset      → saved outputs/dataset_aspirin_YYYYMMDD.json
```

---

### Example 2 — Drug with activity data

```bash
python pipeline.py --query "imatinib" --pubmed
```

**Output:**
```
STEP 1 · Retrieve     → 1 PubChem record, 3 PubMed papers
STEP 2 · Extract      → molecule: Imatinib | target: BCR-ABL | confidence: 0.95
STEP 2b · ChEMBL      → IC50 = 38.0 nM vs Tyrosine-protein kinase ABL1
STEP 3 · Filter       → kept 1/1 (score: 0.972)
STEP 5 · Dataset      → saved outputs/dataset_imatinib_YYYYMMDD.json
```

---

### Example 3 — Structural chemistry question

```bash
python pipeline.py --query "KDM4 inhibitor" --pubmed --questions "Which structural motifs are effective in KDM4 inhibitors: pyridine, C=S groups, N-OH groups, or long carbon chains?"
```

**Output:**
```
STEP 4 · Reason       → Routing to StructuralAgent (ChemCrow-style)
                         [ChemMCP:Name2Smiles]     looking up: ML324
                         [ChemMCP:FunctionalGroups] analysing SMILES...
                         [ChemMCP:MoleculeSimilarity] Tanimoto computed

Answer: Pyridine is present in 100% of known KDM4 inhibitors —
it acts as a bidentate Fe(II) chelator mimicking the 2-oxoglutarate
cofactor. C=S groups and N-OH groups are absent and not effective.
```

---

## Command Reference

```bash
# Basic
python pipeline.py --query "compound name"

# With PubMed (recommended)
python pipeline.py --query "compound name" --pubmed

# More results per source
python pipeline.py --query "compound name" --pubmed --max 5

# With scientific questions
python pipeline.py --query "compound name" --pubmed --questions "your question here"

# Multiple questions
python pipeline.py --query "imatinib" --pubmed \
  --questions "What is the mechanism of imatinib?" \
              "Compare imatinib and dasatinib"
```

---

## Project Structure

```
DataCollect/
├── pipeline.py              ← Main entry point — run this
├── config.py                ← Loads API keys from .env
├── requirements.txt
├── README.md
│
├── agents/
│   ├── retriever.py         ← Fetches from PubChem + PubMed
│   ├── extractor.py         ← Claude extracts structured fields
│   ├── chembl_enricher.py   ← Fills IC50 values from ChEMBL
│   ├── filter.py            ← 4-layer quality filter
│   ├── reasoner.py          ← Scientific Q&A with real data
│   ├── structural_agent.py  ← Structural motif analysis (ChemMCP)
│   └── dataset_builder.py   ← Packages final JSON output
│
└── tools/
    ├── models.py             ← Shared data blueprints
    ├── http_client.py        ← Safe HTTP with retry/backoff
    ├── pubchem_tool.py       ← PubChem API calls
    ├── pubmed_tool.py        ← PubMed E-utilities API calls
    ├── chembl_tool.py        ← ChEMBL REST API calls
    └── chemmcp_tools.py      ← Chemistry tools (RDKit + ChemMCP)
```

---

## Requirements

```
anthropic
langgraph
pydantic
requests
python-dotenv
rdkit
chembl_webresource_client
sentence-transformers
```

---

## Two Core Principles

**1. No single database has everything**
PubChem has structure but not biology. PubMed has biology but as unstructured text. ChEMBL has measured activity values. The pipeline chains all three so each fills what the others cannot.

**2. Claude never answers from memory alone**
Every agent fetches real data from a database first, then passes it to Claude to explain. This prevents hallucination and ensures every value in the output is traceable to a real source.

---

## Built With

- [Anthropic Claude](https://anthropic.com) — Claude Opus for extraction and reasoning, Claude Haiku for classification and scoring
- [LangGraph](https://github.com/langchain-ai/langgraph) — Agent orchestration
- [RDKit](https://www.rdkit.org) — Cheminformatics and structural analysis
- [ChemMCP](https://github.com/OSU-NLP-Group/ChemMCP) — Chemistry tools (MIT licence)
- [PubChem](https://pubchem.ncbi.nlm.nih.gov) — Chemical structure database
- [PubMed](https://pubmed.ncbi.nlm.nih.gov) — Biomedical literature
- [ChEMBL](https://www.ebi.ac.uk/chembl) — Bioactivity database
