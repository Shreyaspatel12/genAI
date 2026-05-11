"""
tools/chemmcp_tools.py — Selected ChemMCP tools extracted for direct use.

These are the core chemistry functions from ChemMCP (OSU-NLP-Group/ChemMCP)
extracted and adapted to work without the full MCP server setup.

Tools included:
  - functional_groups(smiles)     → identify all functional groups in a molecule
  - smiles_check(smiles)          → validate a SMILES string
  - molecule_weight(smiles)       → calculate molecular weight
  - tanimoto_similarity(s1, s2)   → calculate structural similarity score
  - smiles_to_formula(smiles)     → convert SMILES to molecular formula
  - name_to_smiles(name)          → convert compound name to SMILES via PubChem

Credit: ChemMCP by OSU-NLP-Group (https://github.com/OSU-NLP-Group/ChemMCP)
License: MIT
"""
from __future__ import annotations
import logging
import requests
from typing import Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Functional Groups — from ChemMCP functional_groups.py
# Full list from RDKit's FunctionalGroups.txt
# ══════════════════════════════════════════════════════════════════════════════

_FUNCTIONAL_GROUPS = {
    "furan":                    "o1cccc1",
    "aldehydes":                "[CX3H1](=O)[#6]",
    "esters":                   "[#6][CX3](=O)[OX2H0][#6]",
    "ketones":                  "[#6][CX3](=O)[#6]",
    "amides":                   "C(=O)-N",
    "thiol groups":             "[SH]",
    "alcohol groups":           "[OH]",
    "methylamide":              "*-[N;D2]-[C;D3](=O)-[C;D1;H3]",
    "carboxylic acids":         "*-C(=O)[O;D1]",
    "carbonyl methylester":     "*-C(=O)[O;D2]-[C;D1;H3]",
    "terminal aldehyde":        "*-C(=O)-[C;D1]",
    "amide":                    "*-C(=O)-[N;D1]",
    "isocyanate":               "*-[N;D2]=[C;D2]=[O;D1]",
    "isothiocyanate":           "*-[N;D2]=[C;D2]=[S;D1]",
    "nitro":                    "*-[N;D3](=[O;D1])[O;D1]",
    "nitroso":                  "*-[N;R0]=[O;D1]",
    "oximes":                   "*=[N;R0]-[O;D1]",
    "imines":                   "*-[N;R0]=[C;D1;H2]",
    "terminal azo":             "*-[N;D2]=[N;D2]-[C;D1;H3]",
    "hydrazines":               "*-[N;D2]=[N;D1]",
    "diazo":                    "*-[N;D2]#[N;D1]",
    "cyano":                    "*-[C;D2]#[N;D1]",
    "primary sulfonamide":      "*-[S;D4](=[O;D1])(=[O;D1])-[N;D1]",
    "methyl sulfonamide":       "*-[N;D2]-[S;D4](=[O;D1])(=[O;D1])-[C;D1;H3]",
    "sulfonic acid":            "*-[S;D4](=O)(=O)-[O;D1]",
    "methyl ester sulfonyl":    "*-[S;D4](=O)(=O)-[O;D2]-[C;D1;H3]",
    "methyl sulfonyl":          "*-[S;D4](=O)(=O)-[C;D1;H3]",
    "sulfonyl chloride":        "*-[S;D4](=O)(=O)-[Cl]",
    "methyl sulfinyl":          "*-[S;D3](=O)-[C;D1]",
    "methyl thio":              "*-[S;D2]-[C;D1;H3]",
    "thiols":                   "*-[S;D1]",
    "thio carbonyls":           "*=[S;D1]",
    "halogens":                 "*-[#9,#17,#35,#53]",
    "t-butyl":                  "*-[C;D4]([C;D1])([C;D1])-[C;D1]",
    "tri fluoromethyl":         "*-[C;D4](F)(F)F",
    "acetylenes":               "*-[C;D2]#[C;D1;H]",
    "cyclopropyl":              "*-[C;D3]1-[C;D2]-[C;D2]1",
    "ethoxy":                   "*-[O;D2]-[C;D2]-[C;D1;H3]",
    "methoxy":                  "*-[O;D2]-[C;D1;H3]",
    "side-chain hydroxyls":     "*-[O;D1]",
    "primary amines":           "*-[N;D1]",
    "nitriles":                 "*#[N;D1]",
    # Extra groups relevant to KDM4 / medicinal chemistry
    "hydroxamic acid":          "[CX3](=O)[NX3][OX2H]",
    "pyridine":                 "n1ccccc1",
    "quinoline":                "c1ccc2ncccc2c1",
    "hydroxylamine (N-OH)":     "[NX3][OX2H]",
    "sulfonate ester (SO2OR)":  "[SX4](=O)(=O)[OX2][#6]",
    "thioketone (C=S)":         "[CX3]=[SX1]",
}


def functional_groups(smiles: str) -> str:
    """
    Identify all functional groups present in a molecule.
    Adapted from ChemMCP FunctionalGroups tool.

    Args:
        smiles: SMILES string of the molecule

    Returns:
        Human-readable string listing all detected functional groups
    """
    logger.info("[ChemMCP:FunctionalGroups] analysing SMILES: %s", smiles[:40])
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return f"Invalid SMILES string: {smiles}"

    found = []
    for name, smarts in _FUNCTIONAL_GROUPS.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            if pattern and mol.HasSubstructMatch(pattern):
                found.append(name)
        except Exception:
            continue

    if len(found) > 1:
        return f"This molecule contains: {', '.join(found[:-1])}, and {found[-1]}."
    elif len(found) == 1:
        return f"This molecule contains: {found[0]}."
    else:
        return "No common functional groups detected in this molecule."


def functional_groups_dict(smiles: str) -> dict[str, bool]:
    """
    Same as functional_groups() but returns a dict of {group_name: True/False}
    for all groups — useful for programmatic analysis.
    """
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return {}

    results = {}
    for name, smarts in _FUNCTIONAL_GROUPS.items():
        try:
            pattern = Chem.MolFromSmarts(smarts)
            results[name] = bool(pattern and mol.HasSubstructMatch(pattern))
        except Exception:
            results[name] = False
    return results


# ══════════════════════════════════════════════════════════════════════════════
# SMILES Validation — from ChemMCP molecule_smiles_check.py
# ══════════════════════════════════════════════════════════════════════════════

def smiles_check(smiles: str) -> str:
    """
    Check if a SMILES string is chemically valid.
    Adapted from ChemMCP MoleculeSmilesCheck tool.

    Returns:
        "valid" or "invalid" with explanation
    """
    logger.info("[ChemMCP:MoleculeSmilesCheck] validating SMILES: %s", smiles[:40])
    if ">" in smiles:
        return "invalid — contains '>' which suggests a reaction SMARTS, not a molecule SMILES"
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return f"invalid — RDKit could not parse: {smiles}"
        return "valid"
    except Exception as e:
        return f"invalid — {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Molecular Weight — from ChemMCP molecule_weight.py
# ══════════════════════════════════════════════════════════════════════════════

def molecule_weight(smiles: str) -> Optional[float]:
    """
    Calculate exact molecular weight from SMILES.
    Adapted from ChemMCP MoleculeWeight tool.
    """
    logger.info("[ChemMCP:MoleculeWeight] calculating MW for: %s", smiles[:40])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return round(rdMolDescriptors.CalcExactMolWt(mol), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Tanimoto Similarity — from ChemMCP molecule_similarity.py
# ══════════════════════════════════════════════════════════════════════════════

def tanimoto_similarity(smiles1: str, smiles2: str) -> str:
    """
    Calculate Tanimoto structural similarity between two molecules.
    Adapted from ChemMCP MoleculeSimilarity tool.

    Returns:
        Human-readable similarity description with score
    """
    logger.info("[ChemMCP:MoleculeSimilarity] Tanimoto: %s vs %s",
                smiles1[:30], smiles2[:30])
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return "Error: one or both SMILES strings are invalid"

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
        score = DataStructs.TanimotoSimilarity(fp1, fp2)

        if score == 1.0:
            return "The two molecules are identical (Tanimoto = 1.0)"

        labels = {0.9: "very similar", 0.8: "similar", 0.7: "somewhat similar",
                  0.6: "not very similar", 0: "not similar"}
        label = labels[max(k for k in labels if k <= round(score, 1))]

        return (f"Tanimoto similarity = {round(score, 4)} "
                f"— the two molecules are {label}")
    except Exception as e:
        return f"Similarity calculation failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# SMILES to Formula — from ChemMCP smiles2formula.py
# ══════════════════════════════════════════════════════════════════════════════

def smiles_to_formula(smiles: str) -> Optional[str]:
    """
    Convert a SMILES string to molecular formula.
    Adapted from ChemMCP Smiles2Formula tool.
    """
    logger.info("[ChemMCP:Smiles2Formula] converting SMILES: %s", smiles[:40])
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return rdMolDescriptors.CalcMolFormula(mol)


# ══════════════════════════════════════════════════════════════════════════════
# Name to SMILES — from ChemMCP name2smiles.py (via PubChem)
# ══════════════════════════════════════════════════════════════════════════════

def name_to_smiles(name: str) -> Optional[str]:
    """
    Convert a chemical name to SMILES using PubChem API.
    Adapted from ChemMCP Name2Smiles tool.

    Args:
        name: Common or IUPAC name of the compound

    Returns:
        SMILES string or None if not found
    """
    try:
        logger.info("[ChemMCP:Name2Smiles] looking up: %s", name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/IsomericSMILES,CanonicalSMILES/JSON"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            props = data["PropertyTable"]["Properties"][0]
            # Try IsomericSMILES first, fall back to CanonicalSMILES
            smiles = props.get("IsomericSMILES") or props.get("CanonicalSMILES")
            if smiles:
                logger.info("name_to_smiles: %s → %s", name, smiles[:50])
                return smiles
            return None
        else:
            logger.warning("name_to_smiles: PubChem returned %d for '%s'", resp.status_code, name)
            return None
    except Exception as e:
        logger.error("name_to_smiles failed for '%s': %s", name, e)
        return None