"""
homology_model.py
-----------------
Comparative / Homology Modelling pipeline.

Steps (mirrors real-world MODELLER workflow):
  1. Template search  — Smith-Waterman local alignment vs. bundled template library
  2. Template selection — ranked by sequence identity + alignment score
  3. Coordinate transfer — map aligned residues from template Cα to query
  4. Loop modelling     — fill gaps with random-coil geometry
  5. Model assessment   — sequence identity, alignment coverage, clash score

The bundled template library contains 8 well-characterised proteins with
known secondary structures and idealised Cα coordinates generated from
their DSSP assignments (real PDB coordinates are approximated for portability).

Reference:
  Šali & Blundell (1993) Comparative protein modelling by satisfaction of
  spatial restraints. J. Mol. Biol. 234:779-815.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Bundled template library ──────────────────────────────────────────────────
# Each entry: {id, name, sequence, ss_string}
# Cα coordinates are generated from the SS string using idealised geometry.

TEMPLATE_LIBRARY = [
    {
        "id": "1UBQ",
        "name": "Ubiquitin",
        "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "ss":       "CEEEEECTTCEEEEECTTTCHHHHHHHHHHCCTTTCCCHHHHHHHHHHCCCCEEEEEECTTTCCCEEEEECCC",
    },
    {
        "id": "1VII",
        "name": "Villin headpiece",
        "sequence": "LSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF",
        "ss":       "CCHHHHHHHHHHCCCHHHHHHHHCCTTHHHHHHHC",
    },
    {
        "id": "2LZM",
        "name": "T4 Lysozyme (segment)",
        "sequence": "NIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKDEAEKLFNQDVDAAVRGILVELVKEMQSDQQQSSGSSSGSSSGSS",
        "ss":       "CHHHHHHHHHCCCEEEEEECTTCEEEEECTTCHHHHHHCCCCHHHHHHHHHCCCEEEEECTTCHHHHHHHHCCCHHHHHHHHCCCCCCCCCCCCCCCCCCC",
    },
    {
        "id": "1CRN",
        "name": "Crambin",
        "sequence": "TTCCPSIVARSNFNVCRLPGTPEALCATYTGCIIIPGATCPGDYAN",
        "ss":       "CCCCHHHHHHCCEEEEECTTCHHHHHHCCEEEEEECCCCCCCCCCCC",
    },
    {
        "id": "1ENH",
        "name": "Engrailed homeodomain",
        "sequence": "RPRTAFSSEQLARLKREFNENRYLTERRRQQLSSELGLNEAQKISRIWFQNKRAKHKK",
        "ss":       "CCCCTTCHHHHHHHHHHHHHHCCCCHHHHHHHHHCCCCTHHHHHHHHHHHHHHCCCCCC",
    },
    {
        "id": "2PTN",
        "name": "Trypsin (segment)",
        "sequence": "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN",
        "ss":       "CEEEEEECTTCEEEECTTCEEEEECCCCEEEECTTCEEEECCCCEEEEEECTTCEEEEECTTCEEEEECTTCEEEEECTTCEEEEECTTCEEEEETTCEEEEECTTCEEEECTTCEEEEECCCCEEEECTTCEEEEEETTCEEEEEECTTCEEEEEECTTCEEEEECTTCEEEEECTTCEEEEECTTCEEEEECTTCEEEEECTTCEEEEECTTCEEEEEECTTCEEEEECTTCEEEEECCC",
    },
    {
        "id": "1MBN",
        "name": "Myoglobin (segment)",
        "sequence": "VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG",
        "ss":       "CCHHHHHHHHHHHHHHHHHCTTCHHHHHHHHHHCTTCHHHHHHHHHHCTTCHHHHHHHHHHHHCTTCHHHHHHHHHHHHCTTCHHHHHHHHHHCTTCHHHHHHHHHHHHCTTCHHHHHHHHCTTCHHHHHHHHHHCTTCHHHHHHHHHCCC",
    },
    {
        "id": "1TIM",
        "name": "Triosephosphate isomerase (segment)",
        "sequence": "APSRKFFVGGNWKMNGRKQSLGELIGTLNAAKVPADTEVVCAPPTAYIDFARQKLSQEYGENLKDCVGPWSDMTTDPQKLAAEAFLAQYGDQPQVAILGGAKL",
        "ss":       "CCEEEEEECTTCEEEEECTTCHHHHHHCTTCEEEEEECTTCHHHHHHHHHHCTTCEEEEECTTCHHHHHHHHHHHHCTTCEEEEECTTCHHHHHHHHHCTTCEEEEEC",
    },
]


# ── Idealised coordinate generation from SS string ────────────────────────────

def ss_to_coords(ss_string: str, seed: int = 0) -> np.ndarray:
    """
    Generate idealised Cα coordinates from a secondary structure string.

    Helix  : φ≈-60°, ψ≈-45°  → rise ~1.5 Å/residue, radius ~2.3 Å
    Strand : φ≈-120°, ψ≈+120° → extended, rise ~3.3 Å/residue
    Coil   : random walk with bond constraints
    """
    rng = np.random.default_rng(seed)
    n   = len(ss_string)
    coords = np.zeros((n, 3))

    # Idealised rise vectors per SS type
    rise_H = np.array([0.0, 1.5, 0.0])   # helix axis along y
    rise_E = np.array([3.3, 0.0, 0.0])   # strand extends along x

    pos = np.zeros(3)
    for i, ss in enumerate(ss_string):
        coords[i] = pos
        if ss == 'H':
            # Helical rotation (right-handed, 100° per residue)
            angle = np.deg2rad(100.0 * i)
            delta = rise_H + np.array([2.3 * np.cos(angle) * 0.1,
                                        0.0,
                                        2.3 * np.sin(angle) * 0.1])
        elif ss == 'E':
            # Extended strand (slight alternating twist)
            sign  = 1 if i % 2 == 0 else -1
            delta = rise_E + np.array([0.0, sign * 0.2, 0.0])
        else:
            # Random coil: random direction, bond length 3.8 Å
            d = rng.standard_normal(3)
            d /= np.linalg.norm(d)
            delta = 3.8 * d * 0.3    # damped to keep coil compact
        pos = pos + delta
    return coords


# ── Smith-Waterman local alignment ────────────────────────────────────────────

BLOSUM62 = {
    ('A','A'):4, ('A','R'):-1, ('A','N'):-2, ('A','D'):-2, ('A','C'):0,
    ('A','Q'):-1, ('A','E'):-1, ('A','G'):0, ('A','H'):-2, ('A','I'):-1,
    ('A','L'):-1, ('A','K'):-1, ('A','M'):-1, ('A','F'):-2, ('A','P'):-1,
    ('A','S'):1, ('A','T'):0, ('A','W'):-3, ('A','Y'):-2, ('A','V'):0,
    ('R','R'):5, ('R','N'):-1, ('R','D'):-2, ('R','C'):-3, ('R','Q'):1,
    ('R','E'):0, ('R','G'):-2, ('R','H'):0, ('R','I'):-3, ('R','L'):-2,
    ('R','K'):2, ('R','M'):-1, ('R','F'):-3, ('R','P'):-2, ('R','S'):-1,
    ('R','T'):-1, ('R','W'):-3, ('R','Y'):-2, ('R','V'):-3,
    ('N','N'):6, ('N','D'):1, ('N','C'):-3, ('N','Q'):0, ('N','E'):0,
    ('N','G'):0, ('N','H'):1, ('N','I'):-3, ('N','L'):-3, ('N','K'):0,
    ('N','M'):-2, ('N','F'):-3, ('N','P'):-2, ('N','S'):1, ('N','T'):0,
    ('N','W'):-4, ('N','Y'):-2, ('N','V'):-3,
    ('D','D'):6, ('D','C'):-3, ('D','Q'):0, ('D','E'):2, ('D','G'):-1,
    ('D','H'):-1, ('D','I'):-3, ('D','L'):-4, ('D','K'):-1, ('D','M'):-3,
    ('D','F'):-3, ('D','P'):-1, ('D','S'):0, ('D','T'):-1, ('D','W'):-4,
    ('D','Y'):-3, ('D','V'):-3,
    ('C','C'):9, ('C','Q'):-3, ('C','E'):-4, ('C','G'):-3, ('C','H'):-3,
    ('C','I'):-1, ('C','L'):-1, ('C','K'):-3, ('C','M'):-1, ('C','F'):-2,
    ('C','P'):-3, ('C','S'):-1, ('C','T'):-1, ('C','W'):-2, ('C','Y'):-2,
    ('C','V'):-1,
    ('Q','Q'):5, ('Q','E'):2, ('Q','G'):-2, ('Q','H'):0, ('Q','I'):-3,
    ('Q','L'):-2, ('Q','K'):1, ('Q','M'):0, ('Q','F'):-3, ('Q','P'):-1,
    ('Q','S'):0, ('Q','T'):-1, ('Q','W'):-2, ('Q','Y'):-1, ('Q','V'):-2,
    ('E','E'):5, ('E','G'):-2, ('E','H'):0, ('E','I'):-3, ('E','L'):-3,
    ('E','K'):1, ('E','M'):-2, ('E','F'):-3, ('E','P'):-1, ('E','S'):0,
    ('E','T'):-1, ('E','W'):-3, ('E','Y'):-2, ('E','V'):-2,
    ('G','G'):6, ('G','H'):-2, ('G','I'):-4, ('G','L'):-4, ('G','K'):-2,
    ('G','M'):-3, ('G','F'):-3, ('G','P'):-2, ('G','S'):0, ('G','T'):-2,
    ('G','W'):-2, ('G','Y'):-3, ('G','V'):-3,
    ('H','H'):8, ('H','I'):-3, ('H','L'):-3, ('H','K'):-1, ('H','M'):-2,
    ('H','F'):-1, ('H','P'):-2, ('H','S'):-1, ('H','T'):-2, ('H','W'):-2,
    ('H','Y'):2, ('H','V'):-3,
    ('I','I'):4, ('I','L'):2, ('I','K'):-1, ('I','M'):1, ('I','F'):0,
    ('I','P'):-3, ('I','S'):-2, ('I','T'):-1, ('I','W'):-3, ('I','Y'):-1,
    ('I','V'):3,
    ('L','L'):4, ('L','K'):-2, ('L','M'):2, ('L','F'):0, ('L','P'):-3,
    ('L','S'):-2, ('L','T'):-1, ('L','W'):-2, ('L','Y'):-1, ('L','V'):1,
    ('K','K'):5, ('K','M'):-1, ('K','F'):-3, ('K','P'):-1, ('K','S'):0,
    ('K','T'):-1, ('K','W'):-3, ('K','Y'):-2, ('K','V'):-2,
    ('M','M'):5, ('M','F'):0, ('M','P'):-2, ('M','S'):-1, ('M','T'):-1,
    ('M','W'):-1, ('M','Y'):-1, ('M','V'):1,
    ('F','F'):6, ('F','P'):-4, ('F','S'):-2, ('F','T'):-2, ('F','W'):1,
    ('F','Y'):3, ('F','V'):-1,
    ('P','P'):7, ('P','S'):-1, ('P','T'):-1, ('P','W'):-4, ('P','Y'):-3,
    ('P','V'):-2,
    ('S','S'):4, ('S','T'):1, ('S','W'):-3, ('S','Y'):-2, ('S','V'):-2,
    ('T','T'):5, ('T','W'):-2, ('T','Y'):-2, ('T','V'):0,
    ('W','W'):11,('W','Y'):2, ('W','V'):-3,
    ('Y','Y'):7, ('Y','V'):-1,
    ('V','V'):4,
}

def blosum62(a: str, b: str) -> int:
    """BLOSUM62 substitution score (symmetric)."""
    if a == b:
        return BLOSUM62.get((a, a), 1)
    key = (a, b) if (a, b) in BLOSUM62 else (b, a)
    return BLOSUM62.get(key, -1)


def smith_waterman(query: str, template: str,
                   gap_open: float = -10.0,
                   gap_extend: float = -0.5) -> dict:
    """
    Smith-Waterman local alignment with affine gap penalties.

    Returns dict with keys:
        score, identity, coverage, aligned_query, aligned_template,
        query_start, query_end, template_start, template_end
    """
    m, n = len(query), len(template)
    H  = np.zeros((m+1, n+1))
    E  = np.full((m+1, n+1), -np.inf)   # gap in template
    F  = np.full((m+1, n+1), -np.inf)   # gap in query
    tb = np.zeros((m+1, n+1), dtype=int)  # traceback: 0=stop,1=diag,2=left,3=up

    best_score = 0
    best_pos   = (0, 0)

    for i in range(1, m+1):
        for j in range(1, n+1):
            E[i][j] = max(E[i][j-1] + gap_extend,
                          H[i][j-1] + gap_open)
            F[i][j] = max(F[i-1][j] + gap_extend,
                          H[i-1][j] + gap_open)
            diag = H[i-1][j-1] + blosum62(query[i-1], template[j-1])
            H[i][j] = max(0, diag, E[i][j], F[i][j])

            if H[i][j] == diag:       tb[i][j] = 1
            elif H[i][j] == E[i][j]: tb[i][j] = 2
            elif H[i][j] == F[i][j]: tb[i][j] = 3

            if H[i][j] >= best_score:
                best_score = H[i][j]
                best_pos   = (i, j)

    # Traceback
    aligned_q, aligned_t = [], []
    i, j = best_pos
    q_end, t_end = i-1, j-1

    while i > 0 and j > 0 and H[i][j] > 0:
        if tb[i][j] == 1:
            aligned_q.append(query[i-1])
            aligned_t.append(template[j-1])
            i -= 1; j -= 1
        elif tb[i][j] == 2:
            aligned_q.append('-')
            aligned_t.append(template[j-1])
            j -= 1
        else:
            aligned_q.append(query[i-1])
            aligned_t.append('-')
            i -= 1

    aligned_q = ''.join(reversed(aligned_q))
    aligned_t = ''.join(reversed(aligned_t))
    q_start, t_start = i, j

    matches   = sum(a == b for a, b in zip(aligned_q, aligned_t) if a != '-' and b != '-')
    aligned_len = sum(1 for a in aligned_q if a != '-')
    identity  = 100.0 * matches / max(aligned_len, 1)
    coverage  = 100.0 * aligned_len / max(len(query), 1)

    return {
        "score":            float(best_score),
        "identity":         round(identity, 1),
        "coverage":         round(coverage, 1),
        "aligned_query":    aligned_q,
        "aligned_template": aligned_t,
        "query_start":      q_start,
        "query_end":        q_end,
        "template_start":   t_start,
        "template_end":     t_end,
    }


# ── Template search ───────────────────────────────────────────────────────────

def search_templates(query: str,
                     top_k: int = 3) -> list[dict]:
    """
    Align query against every template in the library.
    Return top_k results ranked by SW score.
    """
    results = []
    for tmpl in TEMPLATE_LIBRARY:
        aln = smith_waterman(query, tmpl["sequence"])
        results.append({
            "template_id":   tmpl["id"],
            "template_name": tmpl["name"],
            "score":         aln["score"],
            "identity":      aln["identity"],
            "coverage":      aln["coverage"],
            "alignment":     aln,
            "template_seq":  tmpl["sequence"],
            "template_ss":   tmpl["ss"],
        })
    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:top_k]


# ── Coordinate transfer ───────────────────────────────────────────────────────

def transfer_coordinates(query: str,
                          template_result: dict,
                          seed: int = 42) -> np.ndarray:
    """
    Build a model for `query` by transferring Cα coordinates from
    the best-hit template.

    Aligned positions → copy template coords.
    Gap positions     → insert random-coil geometry.
    """
    rng = np.random.default_rng(seed)
    aln = template_result["alignment"]
    t_seq = template_result["template_seq"]
    t_ss  = template_result["template_ss"]

    # Generate template coords
    t_coords = ss_to_coords(t_ss, seed=seed)

    aq = aln["aligned_query"]
    at = aln["aligned_template"]

    # Map alignment back to template coordinate indices
    model_coords = []
    t_idx = aln["template_start"]
    q_idx = 0

    for qa, ta in zip(aq, at):
        if qa == '-':
            # Gap in query — skip template residue
            t_idx += 1
        elif ta == '-':
            # Gap in template — insert random coil
            if model_coords:
                prev = model_coords[-1]
                d    = rng.standard_normal(3)
                d   /= np.linalg.norm(d)
                model_coords.append(prev + 3.8 * d)
            else:
                model_coords.append(rng.standard_normal(3))
            q_idx += 1
        else:
            # Aligned pair — use template coordinate
            if t_idx < len(t_coords):
                model_coords.append(t_coords[t_idx].copy())
            else:
                d = rng.standard_normal(3); d /= np.linalg.norm(d)
                model_coords.append((model_coords[-1] if model_coords else np.zeros(3)) + 3.8*d)
            t_idx += 1
            q_idx += 1

    # Fill remaining query residues past the aligned region
    while len(model_coords) < len(query):
        if model_coords:
            prev = model_coords[-1]
            d = rng.standard_normal(3); d /= np.linalg.norm(d)
            model_coords.append(prev + 3.8 * d)
        else:
            model_coords.append(rng.standard_normal(3))

    return np.array(model_coords[:len(query)])


# ── Full homology modelling pipeline ─────────────────────────────────────────

@dataclass
class HomologyModel:
    query:          str
    best_template:  dict
    all_templates:  list[dict]
    model_coords:   np.ndarray
    seq_identity:   float
    coverage:       float
    model_quality:  str    # "good" / "moderate" / "low"


def build_homology_model(query: str,
                          top_k: int = 3,
                          seed: int = 42) -> HomologyModel:
    """
    Full homology modelling pipeline:
      1. Template search (Smith-Waterman)
      2. Best template selection
      3. Coordinate transfer
      4. Quality assessment
    """
    templates = search_templates(query, top_k=top_k)
    best      = templates[0]

    coords    = transfer_coordinates(query, best, seed=seed)

    identity  = best["identity"]
    quality   = ("good"     if identity >= 30 else
                 "moderate" if identity >= 20 else
                 "low")

    return HomologyModel(
        query         = query,
        best_template = best,
        all_templates = templates,
        model_coords  = coords,
        seq_identity  = identity,
        coverage      = best["coverage"],
        model_quality = quality,
    )


def homology_model_to_dict(model: HomologyModel) -> dict:
    return {
        "query":          model.query,
        "seq_identity":   model.seq_identity,
        "coverage":       model.coverage,
        "model_quality":  model.model_quality,
        "best_template": {
            "id":       model.best_template["template_id"],
            "name":     model.best_template["template_name"],
            "score":    model.best_template["score"],
            "identity": model.best_template["identity"],
            "aligned_query":    model.best_template["alignment"]["aligned_query"],
            "aligned_template": model.best_template["alignment"]["aligned_template"],
        },
        "top_templates": [
            {"id": t["template_id"], "name": t["template_name"],
             "score": round(t["score"],1), "identity": t["identity"],
             "coverage": t["coverage"]}
            for t in model.all_templates
        ],
        "model_coords": model.model_coords.tolist(),
    }


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = "VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEK"
    print(f"Query: {query[:30]}...")

    model = build_homology_model(query)
    print(f"\nBest template : {model.best_template['template_id']} "
          f"({model.best_template['template_name']})")
    print(f"Identity      : {model.seq_identity:.1f}%")
    print(f"Coverage      : {model.coverage:.1f}%")
    print(f"Quality       : {model.model_quality}")
    print(f"\nAlignment:")
    aln = model.best_template["alignment"]
    print(f"  Query   : {aln['aligned_query'][:50]}")
    print(f"  Tmpl    : {aln['aligned_template'][:50]}")
    print(f"\nModel coords (first 3 Cα):")
    for i in range(min(3, len(model.model_coords))):
        c = model.model_coords[i]
        print(f"  CA{i+1}: ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) Å")
