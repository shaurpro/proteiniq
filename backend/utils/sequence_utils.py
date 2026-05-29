# backend/utils/sequence_utils.py
"""
Utility functions for sequence parsing, validation, and encoding.
"""

import re
from typing import Optional

AA_ALPHABET   = "ACDEFGHIKLMNPQRSTVWY"
AA_PROPERTIES = {
    "A": {"charge": 0,  "hydro": 1.8,  "size": "small",  "polar": False},
    "C": {"charge": 0,  "hydro": 2.5,  "size": "small",  "polar": False},
    "D": {"charge": -1, "hydro": -3.5, "size": "medium", "polar": True},
    "E": {"charge": -1, "hydro": -3.5, "size": "large",  "polar": True},
    "F": {"charge": 0,  "hydro": 2.8,  "size": "large",  "polar": False},
    "G": {"charge": 0,  "hydro": -0.4, "size": "tiny",   "polar": False},
    "H": {"charge": +1, "hydro": -3.2, "size": "large",  "polar": True},
    "I": {"charge": 0,  "hydro": 4.5,  "size": "medium", "polar": False},
    "K": {"charge": +1, "hydro": -3.9, "size": "large",  "polar": True},
    "L": {"charge": 0,  "hydro": 3.8,  "size": "medium", "polar": False},
    "M": {"charge": 0,  "hydro": 1.9,  "size": "large",  "polar": False},
    "N": {"charge": 0,  "hydro": -3.5, "size": "medium", "polar": True},
    "P": {"charge": 0,  "hydro": -1.6, "size": "small",  "polar": False},
    "Q": {"charge": 0,  "hydro": -3.5, "size": "large",  "polar": True},
    "R": {"charge": +1, "hydro": -4.5, "size": "large",  "polar": True},
    "S": {"charge": 0,  "hydro": -0.8, "size": "small",  "polar": True},
    "T": {"charge": 0,  "hydro": -0.7, "size": "medium", "polar": True},
    "V": {"charge": 0,  "hydro": 4.2,  "size": "small",  "polar": False},
    "W": {"charge": 0,  "hydro": -0.9, "size": "large",  "polar": False},
    "Y": {"charge": 0,  "hydro": -1.3, "size": "large",  "polar": True},
}


def clean_sequence(raw: str) -> str:
    """Remove whitespace, numbers, FASTA headers, convert to uppercase."""
    lines = raw.strip().splitlines()
    seq   = "".join(l for l in lines if not l.startswith(">"))
    return re.sub(r"[^A-Za-z]", "", seq).upper()


def is_valid_sequence(seq: str) -> bool:
    return bool(seq) and all(aa in AA_ALPHABET for aa in seq)


def parse_fasta(text: str) -> list[dict]:
    """Parse multi-FASTA into list of {header, sequence} dicts."""
    records = []
    current = None
    for line in text.strip().splitlines():
        if line.startswith(">"):
            if current:
                records.append(current)
            current = {"header": line[1:].strip(), "sequence": ""}
        elif current:
            current["sequence"] += line.strip().upper()
    if current:
        records.append(current)
    return records


def sequence_composition(seq: str) -> dict:
    """Return amino acid composition as percentage."""
    total = len(seq)
    return {aa: round(100 * seq.count(aa) / total, 1) for aa in AA_ALPHABET}


def mean_hydrophobicity(seq: str) -> float:
    """Kyte-Doolittle mean hydrophobicity."""
    scores = [AA_PROPERTIES.get(aa, {}).get("hydro", 0.0) for aa in seq]
    return round(sum(scores) / max(len(scores), 1), 3)


def net_charge(seq: str, pH: float = 7.0) -> float:
    """Simplified net charge at given pH (ignores pKa shifts)."""
    pos = sum(1 for aa in seq if AA_PROPERTIES.get(aa, {}).get("charge", 0) > 0)
    neg = sum(1 for aa in seq if AA_PROPERTIES.get(aa, {}).get("charge", 0) < 0)
    return float(pos - neg)


def molecular_weight(seq: str) -> float:
    """Approximate molecular weight in Da."""
    MW = {"A":89,"C":121,"D":133,"E":147,"F":165,"G":75,"H":155,"I":131,
          "K":146,"L":131,"M":149,"N":132,"P":115,"Q":146,"R":174,"S":105,
          "T":119,"V":117,"W":204,"Y":181}
    return round(sum(MW.get(aa, 128) for aa in seq) - (len(seq)-1)*18.0, 1)
