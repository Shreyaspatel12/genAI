"""
pipeline.py — Full end-to-end ChemAgent pipeline (PubMed enabled).

Run:
    # PubChem only (original)
    python pipeline.py --query "aspirin"

    # PubChem + PubMed (new — fills target, mechanism, disease)
    python pipeline.py --query "aspirin" --pubmed

    # With reasoning questions
    python pipeline.py --query "Carboplatin" --pubmed --questions "What is the use of Carboplatin?"
"""
from __future__ import annotations
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from agents.retriever       import RetrieverAgent
from agents.extractor       import ExtractionAgent
from agents.chembl_enricher import ChEMBLEnricher
from agents.filter          import RelevanceFilterAgent
from agents.reasoner        import ReasoningAgent
from agents.dataset_builder import DatasetBuilderAgent
from tools.models           import RetrievalQuery, DataSource

GREEN = "\033[92m"; YELLOW = "\033[93m"; BOLD = "\033[1m"; RESET = "\033[0m"

def section(title): print(f"\n{BOLD}{'━'*55}\n  {title}\n{'━'*55}{RESET}")
def ok(m):   print(f"  {GREEN}✓{RESET}  {m}")
def info(m): print(f"  {YELLOW}→{RESET}  {m}")


def run_pipeline(query: str, max_results: int, questions: list[str], use_pubmed: bool) -> None:

    # Decide which sources to use
    sources = [DataSource.PUBCHEM]
    if use_pubmed:
        sources.append(DataSource.PUBMED)

    # ── Step 1: Retrieve ─────────────────────────────────────────
    section("STEP 1 · Retrieve")
    retriever        = RetrieverAgent()
    retrieval_result = retriever.run(RetrievalQuery(
        query=query,
        max_results=max_results,
        sources=sources,
    ))
    pubchem_n = sum(1 for r in retrieval_result.records if r.source == DataSource.PUBCHEM)
    pubmed_n  = sum(1 for r in retrieval_result.records if r.source == DataSource.PUBMED)
    ok(f"{retrieval_result.total} record(s) retrieved  "
       f"(PubChem: {pubchem_n}  |  PubMed: {pubmed_n})")

    for r in retrieval_result.records:
        src = r.source.value.upper()
        info(f"[{src}] {r.record_id}  {(r.title or '')[:70]}")

    if retrieval_result.total == 0:
        print("\n  No records retrieved. Check your query and try again.")
        sys.exit(1)

    # ── Step 2: Extract ──────────────────────────────────────────
    section("STEP 2 · Extract")
    extractor         = ExtractionAgent()
    extraction_result = extractor.run(query=query, records=retrieval_result.records)
    ok(f"{extraction_result.total} compound(s) extracted")
    for c in extraction_result.compounds:
        info(f"{c.molecule_name or c.source_id}")
        info(f"   Formula    : {c.chemical_formula or '—'}")
        info(f"   Target     : {c.target_protein or '—'}")
        info(f"   Mechanism  : {(c.mechanism_of_action or '—')[:70]}")
        info(f"   Disease    : {c.disease_indication or '—'}")
        info(f"   Activity   : {c.activity_value or '—'} {c.activity_units or ''}")
        info(f"   PubMed IDs : {c.pubmed_ids or '—'}")
        info(f"   Confidence : {c.confidence}")

    # ── Step 2b: ChEMBL Enrichment ───────────────────────────────
    section("STEP 2b · ChEMBL Enrichment")
    enricher = ChEMBLEnricher()
    extraction_result = enricher.run(extraction_result)
    enriched_count = sum(1 for c in extraction_result.compounds if c.activity_value)
    ok(f"{enriched_count} / {extraction_result.total} compound(s) enriched with ChEMBL activity data")
    for c in extraction_result.compounds:
        if c.activity_value:
            info(f"{c.molecule_name or c.source_id}: "
                 f"{c.activity_type} = {c.activity_value} {c.activity_units}")

    # ── Step 3: Filter ───────────────────────────────────────────
    section("STEP 3 · Filter")
    filter_agent  = RelevanceFilterAgent(min_score=0.4)
    filter_result = filter_agent.run(query=query, compounds=extraction_result.compounds)
    stats = filter_result["stats"]
    ok(f"Kept {stats['kept']} / {stats['total_input']} compound(s)  "
       f"(avg score: {stats['avg_score_kept']})")

    filtered_extraction          = extraction_result
    filtered_extraction.compounds = filter_result["kept_compounds"]

    # ── Step 4: Reason ───────────────────────────────────────────
    reasoning_results = []
    if questions:
        section("STEP 4 · Reason")
        reasoner  = ReasoningAgent()
        from agents.structural_agent import StructuralAgent
        structural = StructuralAgent()

        # Keywords that indicate a structural chemistry question
        # Keywords that mean structural GROUP analysis (→ StructuralAgent)
        structural_keywords = [
            "motif", "scaffold", "functional group", "smiles",
            "fragment", "pharmacophore", "sar",
            "C=S", "N-OH", "COOH", "SO2", "chelat",
            "compare", "versus", "vs", "difference between",
            "structural properties", "structural features",
            "structural difference", "what features",
            "structural motifs", "binding mode", "warhead",
        ]

        # Keywords that mean find similar compounds (→ ReasoningAgent)
        similarity_keywords = [
            "similar to", "structurally similar", "analogs",
            "analogues", "what compounds are similar",
            "compounds like", "related compounds",
        ]

        for q in questions:
            info(f"Q: {q}")

            # Decide which agent to use
            # Similarity search takes priority — route back to ReasoningAgent
            is_similarity  = any(kw.lower() in q.lower() for kw in similarity_keywords)
            is_structural  = any(kw.lower() in q.lower() for kw in structural_keywords)

            # If it looks like a similarity search, don't use StructuralAgent
            if is_similarity:
                is_structural = False
                info("→ Routing to ReasoningAgent (similarity search)")

            if is_structural:
                info("→ Routing to StructuralAgent (ChemCrow-style)")

                # Step 1: get compound names from extracted results
                compound_names = [
                    c.molecule_name for c in filtered_extraction.compounds
                    if c.molecule_name
                ]

                # Step 2: also extract any compound names mentioned in the
                # question itself that are NOT already in compound_names
                # This handles "compare imatinib and dasatinib" where
                # dasatinib was never retrieved by the pipeline
                import re
                q_lower = q.lower()
                extra_candidates = []
                # Simple heuristic: capitalised words or known drug suffixes
                # that look like compound names
                tokens = re.findall(r"[A-Za-z][a-z0-9]+-?[a-z0-9]*", q)
                stop_words = {"compare", "structural", "properties", "motifs",
                              "inhibitors", "between", "versus", "and", "the",
                              "what", "which", "are", "how", "does", "as",
                              "features", "make", "effective", "present", "in"}
                for token in tokens:
                    if (token.lower() not in stop_words and
                        len(token) > 4 and
                        token.lower() not in [c.lower() for c in compound_names]):
                        extra_candidates.append(token)

                # Combine — pipeline compounds first, then question extras
                all_compounds = compound_names.copy()
                for ec in extra_candidates:
                    if ec not in all_compounds:
                        all_compounds.append(ec)

                if not all_compounds:
                    all_compounds = [query]

                info(f"Compounds to analyse: {all_compounds}")
                sr = structural.run(question=q, compounds=all_compounds)
                reasoning_results.append({
                    "question":      q,
                    "question_type": "structural",
                    "answer":        sr["answer"],
                    "compounds":     [],
                })
                print(f"\n  {BOLD}Answer (structural):{RESET}")
                for line in sr["answer"].split("\n"):
                    if line.strip():
                        print(f"  {line}")
            else:
                # Pass the source_id (CID) of the first compound if available
                query_cid = ""
                if filtered_extraction.compounds:
                    query_cid = filtered_extraction.compounds[0].source_id or ""
                result = reasoner.run(q, query_compound=query, query_cid=query_cid)
                reasoning_results.append(result)
                print(f"\n  {BOLD}Answer ({result['question_type']}):{RESET}")
                for line in result["answer"].split(". "):
                    if line.strip():
                        print(f"  {line.strip()}.")
            print()
    else:
        section("STEP 4 · Reason")
        info("No questions — skipping (use --questions to add)")

    # ── Step 5: Build Dataset ────────────────────────────────────
    section("STEP 5 · Build Dataset")
    builder = DatasetBuilderAgent()
    result  = builder.run(
        query=query,
        extraction_result=filtered_extraction,
        reasoning_results=reasoning_results,
    )
    ok(f"Dataset saved → {result['output_path']}")
    stats = result["stats"]
    print()
    print(f"  {'─'*42}")
    print(f"  {'Total compounds':<24}: {stats.get('total_compounds', 0)}")
    print(f"  {'With SMILES':<24}: {stats.get('with_smiles', 0)}")
    print(f"  {'With target':<24}: {stats.get('with_target', 0)}")
    print(f"  {'With activity':<24}: {stats.get('with_activity', 0)}")
    print(f"  {'With mechanism':<24}: {stats.get('with_mechanism', 0)}")
    print(f"  {'With disease':<24}: {stats.get('with_disease', 0)}")
    print(f"  {'With PubMed refs':<24}: {stats.get('with_pubmed_refs', 0)}")
    print(f"  {'Avg confidence':<24}: {stats.get('avg_confidence', 0)}")
    print(f"  {'─'*42}")
    print(f"\n{BOLD}  Pipeline complete.{RESET}")
    print(f"  Output: {result['output_path']}\n")


def parse_args():
    p = argparse.ArgumentParser(description="ChemAgent — Full Pipeline")
    p.add_argument("--query",     required=True,  help="Compound or drug name")
    p.add_argument("--max",       type=int, default=3, help="Max results per source")
    p.add_argument("--pubmed",    action="store_true", help="Also search PubMed papers")
    p.add_argument("--questions", nargs="*", default=[],
                   help='Reasoning questions e.g. "What is similar to aspirin?"')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.query, args.max, args.questions, args.pubmed)