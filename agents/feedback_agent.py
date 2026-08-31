"""
agents/feedback_agent.py — Enhanced Feedback Agent for ChemAgent Pipeline.

What this agent does:
  1. Takes 100 SMILES + target protein name
  2. Runs DrugCLIP docking to score all molecules
  3. Computes drug-likeness (Lipinski) for each molecule using RDKit
  4. Fetches PubMed literature for the target protein to ground the reasoning
  5. LLM writes feedback covering: docking analysis + drug-likeness + literature support
  6. Handles follow-up clarification questions

Supports multiple LLM providers:
  - Claude  (Anthropic)  — default
  - GPT-4   (OpenAI)
  - Gemini  (Google)

Usage:
    python agents/feedback_agent.py \
        --smiles_file compounds.txt \
        --target abl1 \
        --drugclip_dir ~/data_storage/DrugCLIP \
        --llm claude        # or: openai, gemini
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Protein target map
PROTEIN_TARGET_MAP = {
    "abl1": "abl1", "bcr-abl": "abl1", "bcr_abl": "abl1",
    "egfr": "egfr", "src": "src", "jak2": "jak2", "braf": "braf",
    "cdk2": "cdk2", "fak1": "fak1", "igf1r": "igf1r", "kit": "kit",
    "lck": "lck", "met": "met", "vgfr2": "vgfr2", "fgfr1": "fgfr1",
    "hdac2": "hdac2", "hdac8": "hdac8", "parp1": "parp1",
    "cxcr4": "cxcr4", "hivpr": "hivpr", "hivrt": "hivrt",
    "bace1": "bace1", "ace": "ace", "andr": "andr", "androgen": "andr",
    "esr1": "esr1", "estrogen": "esr1", "esr2": "esr2",
}


# LLM Provider abstraction

class LLMProvider:
    """
    Unified interface for Claude, OpenAI GPT, and Google Gemini.
    Pass llm='claude', 'openai', or 'gemini' to FeedbackAgent.
    """

    def __init__(self, provider: str = "claude"):
        self.provider = provider.lower().strip()
        self._client  = None
        self._init_client()

    def _init_client(self):
        if self.provider == "claude":
            import anthropic
            from config import ANTHROPIC_API_KEY
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        elif self.provider == "openai":
            try:
                import openai
                api_key = os.environ.get("OPENAI_API_KEY") or self._read_env("OPENAI_API_KEY")
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("pip install openai to use OpenAI provider")

        elif self.provider == "gemini":
            try:
                from google import genai
                api_key = os.environ.get("GEMINI_API_KEY") or self._read_env("GEMINI_API_KEY")
                self._client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError("pip install google-genai to use Gemini provider")

        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}. Use claude, openai, or gemini.")

    def _read_env(self, key: str) -> str:
        """Read a key from .env file if not in environment."""
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip()
        raise ValueError(f"{key} not found in environment or .env file")

    def chat(self, system: str, user: str, max_tokens: int = 2000) -> str:
        """Send a chat message and return the response text."""

        if self.provider == "claude":
            response = self._client.messages.create(
                model="claude-opus-4-5",
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}]
            )
            return response.content[0].text

        elif self.provider == "openai":
            response = self._client.chat.completions.create(
                model="gpt-4o",
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user}
                ]
            )
            return response.choices[0].message.content

        elif self.provider == "gemini":
            prompt = system + "\n\n" + user
            response = self._client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
            return response.text

        return ""


# Drug-likeness calculator (RDKit)

def compute_drug_likeness(smiles: str) -> dict:
    """
    Compute Lipinski Rule of Five and other drug-likeness properties.
    Returns a dict with properties and a pass/fail for each rule.

    Properties computed:
      MW     : molecular weight
      LogP   : lipophilicity
      HBD    : hydrogen bond donors
      HBA    : hydrogen bond acceptors
      TPSA   : topological polar surface area
      RotBonds: rotatable bonds
      Lipinski: True if passes all 5 rules
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, rdMolDescriptors
        from rdkit.Chem.rdMolDescriptors import CalcTPSA

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"valid": False, "error": "Invalid SMILES"}

        mw       = round(Descriptors.MolWt(mol), 2)
        logp     = round(Descriptors.MolLogP(mol), 2)
        hbd      = rdMolDescriptors.CalcNumHBD(mol)
        hba      = rdMolDescriptors.CalcNumHBA(mol)
        tpsa     = round(CalcTPSA(mol), 2)
        rot      = rdMolDescriptors.CalcNumRotatableBonds(mol)

        lipinski = (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10)
        veber    = (tpsa <= 140 and rot <= 10)

        return {
            "valid":        True,
            "MW":           mw,
            "LogP":         logp,
            "HBD":          hbd,
            "HBA":          hba,
            "TPSA":         tpsa,
            "RotBonds":     rot,
            "Lipinski":     lipinski,
            "Veber":        veber,
            "drug_like":    lipinski and veber,
            "violations": {
                "MW>500":   mw > 500,
                "LogP>5":   logp > 5,
                "HBD>5":    hbd > 5,
                "HBA>10":   hba > 10,
                "TPSA>140": tpsa > 140,
                "Rot>10":   rot > 10,
            }
        }
    except Exception as e:
        return {"valid": False, "error": str(e)}


# PubMed literature fetcher

def fetch_pubmed_context(target_protein: str, max_papers: int = 5) -> str:
    """
    Fetch recent PubMed abstracts about the target protein's inhibitors.
    Returns a formatted string of paper titles and abstracts.
    """
    try:
        from tools.pubmed_tool import search_articles
        query = f'("{target_protein} inhibitor"[tiab] OR "{target_protein} binding"[tiab]) AND (structure activity OR pharmacophore OR docking)'
        records = search_articles(query, max_results=max_papers)
        if not records:
            return f"No PubMed papers found for {target_protein}."
        context = []
        for r in records[:max_papers]:
            title    = r.title or r.metadata.get("title", "Unknown title")
            abstract = (r.abstract[:400] if hasattr(r, "abstract") and r.abstract
                       else r.raw.get("abstract", "No abstract available")[:400])
            context.append(f"TITLE: {title}\nABSTRACT: {abstract}")
        return "\n\n---\n\n".join(context)
    except Exception as e:
        logger.warning("PubMed fetch failed: %s", e)
        return f"PubMed literature not available: {e}"


# Main Feedback Agent

class FeedbackAgent:
    """
    Enhanced Feedback Agent with:
    - DrugCLIP docking scores
    - Drug-likeness (Lipinski) evaluation
    - PubMed literature grounding
    - Multi-LLM support (Claude / OpenAI / Gemini)
    """

    def __init__(
        self,
        target_protein: str,
        drugclip_dir: str,
        dude_data_dir: Optional[str] = None,
        top_n: int = 10,
        llm: str = "claude",
    ):
        self.target_protein = target_protein.lower().strip()
        self.drugclip_dir   = os.path.realpath(os.path.expanduser(drugclip_dir))
        self.top_n          = top_n
        self.llm            = LLMProvider(llm)

        self.pocket_path  = self._resolve_pocket_path(dude_data_dir)
        if not self.pocket_path:
            raise ValueError(f"Pocket file not found for target '{target_protein}'")

        self.weights_path = os.path.join(self.drugclip_dir, "data", "checkpoint_best.pt")
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

        # Store results for follow-up questions
        self._docking_results:   list[dict] = []
        self._druglikeness_data: list[dict] = []
        self._last_feedback:     str = ""

        logger.info("FeedbackAgent: target=%s | llm=%s | pocket=%s",
                    self.target_protein, llm, self.pocket_path)

    def _resolve_pocket_path(self, dude_data_dir):
        base = dude_data_dir or os.path.join(
            self.drugclip_dir, "data", "dude", "data", "protein", "DUD-E", "raw", "all"
        )
        base = os.path.expanduser(base)
        key  = PROTEIN_TARGET_MAP.get(self.target_protein, self.target_protein)
        for candidate in [
            os.path.join(base, key, "pocket.lmdb"),
            os.path.join(base, self.target_protein, "pocket.lmdb"),
        ]:
            if os.path.exists(candidate):
                return candidate
        return None

    # SMILES → LMDB 

    def _smiles_to_lmdb(self, smiles_list: list[str], output_path: str) -> int:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for smi in smiles_list:
                f.write(smi.strip() + "\n")
            tmp = f.name
        try:
            converter   = os.path.join(self.drugclip_dir, "smiles_to_lmdb.py")
            drugclip_py = os.path.expanduser("~/.conda/envs/drugclip/bin/python")
            python_exe  = drugclip_py if os.path.exists(drugclip_py) else sys.executable
            result = subprocess.run(
                [python_exe, converter, "--smiles_file", tmp, "--output", output_path],
                capture_output=True, text=True, cwd=self.drugclip_dir
            )
            if result.returncode != 0:
                logger.error("SMILES conversion failed: %s", result.stderr)
                return 0
            for line in result.stdout.split("\n"):
                if "Successfully converted:" in line:
                    return int(line.split(":")[1].strip())
            return len(smiles_list)
        finally:
            os.unlink(tmp)

    # DrugCLIP docking

    def _run_docking(self, mols_lmdb: str, output_dir: str) -> list[dict]:
        os.makedirs(output_dir, exist_ok=True)
        emb_dir = os.path.join(output_dir, "emb")
        os.makedirs(emb_dir, exist_ok=True)

        env = os.environ.copy()
        env["PYTHONPATH"] = self.drugclip_dir + ":" + env.get("PYTHONPATH", "")

        drugclip_py = os.path.expanduser("~/.conda/envs/drugclip/bin/python")
        python_exe  = drugclip_py if os.path.exists(drugclip_py) else sys.executable

        cmd = [
            python_exe,
            os.path.join(self.drugclip_dir, "unimol", "retrieval.py"),
            "--user-dir", "./unimol", "./data",
            "--valid-subset", "test",
            "--results-path", output_dir,
            "--num-workers", "4", "--ddp-backend=c10d",
            "--batch-size", "8",
            "--task", "drugclip", "--loss", "in_batch_softmax", "--arch", "drugclip",
            "--max-pocket-atoms", "256",
            "--fp16", "--fp16-init-scale", "4", "--fp16-scale-window", "256",
            "--seed", "1",
            "--path", self.weights_path,
            "--log-interval", "100", "--log-format", "simple",
            "--mol-path", mols_lmdb,
            "--pocket-path", self.pocket_path,
            "--emb-dir", emb_dir,
        ]

        logger.info("FeedbackAgent: running DrugCLIP docking...")
        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=self.drugclip_dir, env=env)
        if result.returncode != 0:
            logger.error("DrugCLIP failed: %s", result.stderr[-1000:])
            return []

        ranked_file = os.path.join(emb_dir, "ranked_compounds.txt")
        if not os.path.exists(ranked_file):
            return []

        scores = []
        with open(ranked_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        scores.append({"smiles": parts[0], "score": float(parts[1])})
                    except ValueError:
                        pass
        scores.sort(key=lambda x: x["score"], reverse=True)
        logger.info("FeedbackAgent: docking complete — %d scores", len(scores))
        return scores

    # Drug-likeness

    def _compute_drug_likeness(self, scores: list[dict]) -> list[dict]:
        """Add Lipinski properties to each docked molecule."""
        enriched = []
        for s in scores:
            dl = compute_drug_likeness(s["smiles"])
            enriched.append({**s, "drug_likeness": dl})
        return enriched

    # Score analysis

    def _analyse(self, enriched: list[dict]) -> dict:
        if not enriched:
            return {}
        n   = len(enriched)
        top = enriched[:self.top_n]
        bot = enriched[max(0, n - self.top_n):]

        scores = [e["score"] for e in enriched]
        drug_like_count = sum(
            1 for e in enriched
            if e.get("drug_likeness", {}).get("drug_like", False)
        )

        return {
            "total_molecules":  n,
            "target_protein":   self.target_protein,
            "score_range": {
                "highest": round(scores[0], 4),
                "lowest":  round(scores[-1], 4),
                "mean":    round(sum(scores) / n, 4),
            },
            "positive_count":   sum(1 for s in scores if s > 0),
            "negative_count":   sum(1 for s in scores if s <= 0),
            "drug_like_count":  drug_like_count,
            "top_binders":      top,
            "bottom_binders":   bot,
        }

    # LLM feedback generation

    def _generate_feedback(self, analysis: dict, pubmed_context: str) -> str:
        system_prompt = """You are an expert medicinal chemist reviewing virtual screening results.
You have DrugCLIP docking scores, drug-likeness data, and PubMed literature about the target.

Your feedback must cover THREE sections:

## 1. Docking Analysis
- Which structural features correlate with high docking scores
- Why certain scaffolds bind well and others fail
- Specific chemical reasoning for the top and bottom scorers

## 2. Drug-likeness Assessment
- Which top-scoring molecules also pass Lipinski / Veber criteria
- Which have violations (MW > 500, LogP > 5, HBD > 5, HBA > 10, TPSA > 140, RotBonds > 10)
- Highlight molecules that score well in docking BUT fail drug-likeness — these need optimization
- Highlight molecules that are drug-like BUT score poorly in docking — scaffold may need modification

## 3. Recommendations
- Based on the docking results AND the PubMed literature provided, give specific suggestions
- What structural modifications would improve both docking and drug-likeness?
- What should be prioritised in the next round of molecule design?

Use the PubMed literature to support your reasoning where relevant.
Be specific, scientific, and practical. Do not invent data."""

        user_message = f"""TARGET PROTEIN: {analysis['target_protein'].upper()}

SCREENING SUMMARY:
- Total molecules: {analysis['total_molecules']}
- Score range: {analysis['score_range']['lowest']} to {analysis['score_range']['highest']} (mean: {analysis['score_range']['mean']})
- Predicted binders (positive score): {analysis['positive_count']}
- Drug-like molecules (Lipinski + Veber): {analysis['drug_like_count']}

TOP {len(analysis['top_binders'])} MOLECULES (best docking scores):
{json.dumps(analysis['top_binders'], indent=2)}

BOTTOM {len(analysis['bottom_binders'])} MOLECULES (worst docking scores):
{json.dumps(analysis['bottom_binders'], indent=2)}

PUBMED LITERATURE ON {analysis['target_protein'].upper()} INHIBITORS:
{pubmed_context}

Please provide your medicinal chemistry feedback covering all three sections."""

        return self.llm.chat(system_prompt, user_message, max_tokens=2000)

    # Public API

    def run(self, smiles_list: list[str]) -> str:
        if not smiles_list:
            return "No SMILES provided."

        logger.info("FeedbackAgent.run: %d molecules → %s", len(smiles_list), self.target_protein)

        with tempfile.TemporaryDirectory() as tmpdir:
            mols_lmdb  = os.path.join(tmpdir, "mols.lmdb")
            output_dir = os.path.join(tmpdir, "docking")

            # Step 1 — Convert SMILES
            logger.info("Step 1: Converting SMILES to LMDB...")
            n = self._smiles_to_lmdb(smiles_list, mols_lmdb)
            if n == 0:
                return "Failed to convert SMILES to 3D format."
            logger.info("  Converted %d / %d", n, len(smiles_list))

            # Step 2 — DrugCLIP docking
            logger.info("Step 2: Running DrugCLIP docking...")
            scores = self._run_docking(mols_lmdb, output_dir)
            if not scores:
                return "DrugCLIP docking failed."

        # Step 3 — Drug-likeness
        logger.info("Step 3: Computing drug-likeness...")
        enriched = self._compute_drug_likeness(scores)
        self._docking_results   = enriched
        self._druglikeness_data = enriched

        # Step 4 — PubMed literature
        logger.info("Step 4: Fetching PubMed literature...")
        pubmed_context = fetch_pubmed_context(self.target_protein)

        # Step 5 — Analyse
        logger.info("Step 5: Analysing results...")
        analysis = self._analyse(enriched)

        # Step 6 — Generate feedback
        logger.info("Step 6: Generating feedback with %s...", self.llm.provider)
        feedback = self._generate_feedback(analysis, pubmed_context)
        self._last_feedback = feedback
        return feedback

    def run_from_file(self, smiles_file: str) -> str:
        with open(smiles_file) as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        return self.run(lines)

    def answer(self, question: str) -> str:
        if not self._docking_results:
            return "No results available. Please run the feedback agent first."

        system = """You are an expert medicinal chemist. Answer the follow-up question
based on the docking scores, drug-likeness data, and previous feedback provided.
Be specific and reference actual data where relevant."""

        user = f"""Docking results (top 20):
{json.dumps(self._docking_results[:20], indent=2)}

Previous feedback:
{self._last_feedback[:800]}

Question: {question}"""

        return self.llm.chat(system, user, max_tokens=800)

    def get_scores(self) -> list[dict]:
        return self._docking_results


# Standalone usage

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Feedback Agent — DrugCLIP + PubMed + LLM")
    parser.add_argument("--smiles_file",   required=True)
    parser.add_argument("--target",        required=True)
    parser.add_argument("--drugclip_dir",  default=os.path.expanduser("~/data_storage/DrugCLIP"))
    parser.add_argument("--top_n",         type=int, default=10)
    parser.add_argument("--llm",           default="claude",
                        choices=["claude", "openai", "gemini"],
                        help="LLM provider to use (default: claude)")
    args = parser.parse_args()

    agent = FeedbackAgent(
        target_protein = args.target,
        drugclip_dir   = args.drugclip_dir,
        top_n          = args.top_n,
        llm            = args.llm,
    )

    n_mols = sum(1 for l in open(args.smiles_file) if l.strip() and not l.startswith("#"))
    print(f"\nRunning feedback for {n_mols} molecules against {args.target.upper()} using {args.llm.upper()}...\n")

    feedback = agent.run_from_file(args.smiles_file)

    print("\n" + "═" * 60)
    print(f"  MEDICINAL CHEMISTRY FEEDBACK  [{args.llm.upper()}]")
    print("═" * 60)
    print(feedback)
    print("═" * 60)

    print("\nAsk follow-up questions (type 'quit' to exit):")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("quit", "exit", "q"):
            break
        if q:
            print(f"\nAnswer: {agent.answer(q)}")