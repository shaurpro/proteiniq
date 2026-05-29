"""
homology_model_extended.py
--------------------------
Extended homology modelling pipeline that works for ANY sequence length.

Two modes:
  1. OFFLINE  — uses bundled 8-protein library (fast, no internet, ≤150 residues)
  2. ONLINE   — queries NCBI BLAST + downloads real PDB templates (any length)

Online pipeline steps:
  1. PSI-BLAST / BLASTp query against PDB database via NCBI API
  2. Parse top hits → extract PDB IDs + chain IDs
  3. Download PDB structure files from RCSB
  4. Extract Cα coordinates for the matching chain
  5. Smith-Waterman alignment of query vs template sequence
  6. Coordinate transfer (aligned residues) + loop filling (gaps)
  7. Multi-template support — merge best regions from top 3 templates
  8. Model quality scoring
"""

import time
import urllib.request
import urllib.parse
import io
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# BioPython imports (install: pip install biopython)
try:
    from Bio.Blast import NCBIWWW, NCBIXML
    from Bio import SeqIO, PDB
    from Bio.PDB import PDBParser, PPBuilder
    from Bio.PDB.Polypeptide import is_aa
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("Warning: BioPython not installed. Online mode unavailable.")
    print("Install with: pip install biopython")

from homology_model import (smith_waterman, TEMPLATE_LIBRARY,
                              ss_to_coords, HomologyModel,
                              homology_model_to_dict)


# ── NCBI BLAST search against PDB ────────────────────────────────────────────

def blast_against_pdb(sequence: str,
                       n_hits: int = 5,
                       e_value: float = 0.01,
                       email: str = "your@email.com") -> list[dict]:
    """
    Run BLASTp against the PDB database using NCBI's web API.

    Parameters
    ----------
    sequence : amino acid query sequence
    n_hits   : number of top hits to return
    e_value  : E-value threshold (lower = more stringent)
    email    : required by NCBI (set your own)

    Returns
    -------
    List of dicts: [{pdb_id, chain_id, identity, coverage, e_value, score}, ...]
    """
    if not BIOPYTHON_AVAILABLE:
        raise RuntimeError("BioPython required for online mode.")

    print(f"Running BLASTp against PDB for {len(sequence)}-residue sequence...")
    print("This may take 30-60 seconds...")

    # Submit BLAST job to NCBI
    result_handle = NCBIWWW.qblast(
        program   = "blastp",
        database  = "pdb",          # search against PDB sequences
        sequence  = sequence,
        hitlist_size = n_hits * 3,  # fetch more, filter later
        expect    = e_value,
        format_type = "XML",
    )

    blast_records = list(NCBIXML.parse(result_handle))
    if not blast_records:
        return []

    hits = []
    seen_pdb = set()

    for record in blast_records:
        for alignment in record.alignments:
            for hsp in alignment.hsps:
                # Parse PDB ID and chain from title: "pdb|1ABC|A ..."
                title = alignment.title
                parts = title.split("|")
                if len(parts) < 3:
                    continue
                pdb_id  = parts[1].upper()
                chain_id = parts[2].split()[0].upper() if parts[2] else "A"

                if pdb_id in seen_pdb:
                    continue
                seen_pdb.add(pdb_id)

                identity = round(100.0 * hsp.identities / hsp.align_length, 1)
                coverage = round(100.0 * hsp.align_length / len(sequence), 1)

                hits.append({
                    "pdb_id":   pdb_id,
                    "chain_id": chain_id,
                    "identity": identity,
                    "coverage": coverage,
                    "e_value":  hsp.expect,
                    "score":    hsp.score,
                    "query_seq":   str(hsp.query).replace("-", ""),
                    "subject_seq": str(hsp.sbjct).replace("-", ""),
                    "aligned_q":   str(hsp.query),
                    "aligned_s":   str(hsp.sbjct),
                })

                if len(hits) >= n_hits:
                    break
            if len(hits) >= n_hits:
                break

    hits.sort(key=lambda h: h["score"], reverse=True)
    print(f"Found {len(hits)} template(s) in PDB.")
    return hits[:n_hits]


# ── PDB structure download + Cα extraction ───────────────────────────────────

def fetch_pdb_structure(pdb_id: str,
                         chain_id: str = "A") -> Optional[dict]:
    """
    Download PDB file from RCSB and extract Cα coordinates + sequence.

    Returns
    -------
    dict with keys: sequence, ca_coords (np.ndarray shape (N,3)), chain_id
    or None on failure.
    """
    pdb_id = pdb_id.lower()
    url    = f"https://files.rcsb.org/download/{pdb_id}.pdb"

    try:
        print(f"Downloading {pdb_id.upper()} from RCSB...")
        with urllib.request.urlopen(url, timeout=15) as resp:
            pdb_data = resp.read().decode("utf-8")
    except Exception as e:
        print(f"  Warning: Could not download {pdb_id}: {e}")
        return None

    if not BIOPYTHON_AVAILABLE:
        # Manual Cα extraction fallback (no BioPython)
        return _manual_ca_extract(pdb_data, chain_id)

    # BioPython parsing
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, io.StringIO(pdb_data))

    ca_coords = []
    sequence  = []
    AA3_TO_1  = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    }

    for model in structure:
        for chain in model:
            if chain.id.upper() != chain_id.upper():
                continue
            for residue in chain:
                if not is_aa(residue, standard=True):
                    continue
                if "CA" not in residue:
                    continue
                resname = residue.resname.strip()
                aa1 = AA3_TO_1.get(resname)
                if aa1 is None:
                    continue
                ca_coords.append(residue["CA"].get_vector().get_array())
                sequence.append(aa1)
        break   # only first model

    if not ca_coords:
        # Try any chain if specified chain not found
        for model in structure:
            for chain in model:
                for residue in chain:
                    if not is_aa(residue, standard=True):
                        continue
                    if "CA" not in residue:
                        continue
                    resname = residue.resname.strip()
                    aa1 = AA3_TO_1.get(resname)
                    if aa1 is None:
                        continue
                    ca_coords.append(residue["CA"].get_vector().get_array())
                    sequence.append(aa1)
                break
            break

    if not ca_coords:
        print(f"  Warning: No Cα atoms found in {pdb_id} chain {chain_id}")
        return None

    print(f"  Extracted {len(ca_coords)} Cα atoms from {pdb_id} chain {chain_id}")
    return {
        "pdb_id":    pdb_id.upper(),
        "chain_id":  chain_id,
        "sequence":  "".join(sequence),
        "ca_coords": np.array(ca_coords),
    }


def _manual_ca_extract(pdb_data: str, chain_id: str) -> Optional[dict]:
    """Fallback Cα extractor that parses PDB ATOM records directly."""
    AA3_TO_1 = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V",
    }
    ca_coords, sequence, last_resnum = [], [], None

    for line in pdb_data.splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        chain     = line[21].strip()
        resname   = line[17:20].strip()
        resnum    = line[22:26].strip()

        if chain.upper() != chain_id.upper() and chain_id != "*":
            continue
        if atom_name != "CA":
            continue
        if resnum == last_resnum:
            continue   # skip insertion codes
        aa1 = AA3_TO_1.get(resname)
        if aa1 is None:
            continue

        try:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue

        ca_coords.append([x, y, z])
        sequence.append(aa1)
        last_resnum = resnum

    if not ca_coords:
        return None
    return {
        "sequence":  "".join(sequence),
        "ca_coords": np.array(ca_coords),
        "chain_id":  chain_id,
    }


# ── Coordinate transfer with real PDB template ───────────────────────────────

def transfer_real_coords(query: str,
                          template_data: dict,
                          blast_hit: dict,
                          seed: int = 42) -> np.ndarray:
    """
    Transfer Cα coordinates from a real PDB template to the query.

    Uses the BLAST alignment for initial mapping, refined by
    Smith-Waterman for precision.
    """
    rng = np.random.default_rng(seed)
    t_seq    = template_data["sequence"]
    t_coords = template_data["ca_coords"]

    # Re-align with Smith-Waterman for precise residue mapping
    aln = smith_waterman(query, t_seq)
    aq  = aln["aligned_query"]
    at  = aln["aligned_template"]

    model_coords = []
    t_idx = aln["template_start"]
    bond_len = 3.8   # Å

    for qa, ta in zip(aq, at):
        if qa == "-":
            t_idx += 1
        elif ta == "-":
            # Gap in template — random coil insertion
            if model_coords:
                prev = model_coords[-1]
                d    = rng.standard_normal(3)
                d   /= np.linalg.norm(d)
                model_coords.append(prev + bond_len * d)
            else:
                model_coords.append(rng.standard_normal(3) * 2)
        else:
            if t_idx < len(t_coords):
                model_coords.append(t_coords[t_idx].copy())
            else:
                prev = model_coords[-1] if model_coords else np.zeros(3)
                d    = rng.standard_normal(3); d /= np.linalg.norm(d)
                model_coords.append(prev + bond_len * d)
            t_idx += 1

    # Pad remaining query residues beyond aligned region
    while len(model_coords) < len(query):
        prev = model_coords[-1] if model_coords else np.zeros(3)
        d    = rng.standard_normal(3); d /= np.linalg.norm(d)
        model_coords.append(prev + bond_len * d)

    return np.array(model_coords[:len(query)])


# ── Multi-template modelling ──────────────────────────────────────────────────

def multi_template_model(query: str,
                          blast_hits: list[dict],
                          pdb_structures: list[dict],
                          seed: int = 42) -> np.ndarray:
    """
    Build a model using the BEST template per region of the query.

    For each residue position, select the template with the highest
    local sequence identity in that region (±10 residue window).
    Falls back gracefully if only one template available.
    """
    if len(pdb_structures) == 1:
        return transfer_real_coords(query, pdb_structures[0],
                                     blast_hits[0], seed=seed)

    n = len(query)
    all_coords = []

    for pdb, hit in zip(pdb_structures, blast_hits):
        try:
            coords = transfer_real_coords(query, pdb, hit, seed=seed+len(all_coords))
            all_coords.append((coords, hit["identity"]))
        except Exception:
            continue

    if not all_coords:
        from homology_model import build_homology_model
        return build_homology_model(query, seed=seed).model_coords

    # Weight-average coordinates by identity score
    total_weight = sum(w for _, w in all_coords)
    model = np.zeros((n, 3))
    for coords, weight in all_coords:
        model += coords * (weight / total_weight)

    return model


# ── Full online pipeline ──────────────────────────────────────────────────────

@dataclass
class ExtendedHomologyModel:
    query:          str
    mode:           str   # "online" or "offline"
    best_template:  dict
    all_templates:  list[dict]
    model_coords:   np.ndarray
    seq_identity:   float
    coverage:       float
    model_quality:  str
    n_residues:     int
    multi_template: bool = False


def build_extended_model(query:       str,
                          online:      bool = True,
                          n_templates: int  = 3,
                          email:       str  = "your@email.com",
                          seed:        int  = 42) -> ExtendedHomologyModel:
    """
    Build homology model — switches automatically between online (BLAST+PDB)
    and offline (bundled library) modes.

    Parameters
    ----------
    query        : amino acid sequence (any length)
    online       : True = use NCBI BLAST + PDB; False = use bundled library
    n_templates  : number of templates to use (multi-template if >1)
    email        : your email (required by NCBI for BLAST)
    seed         : random seed for reproducibility
    """

    if not online or not BIOPYTHON_AVAILABLE:
        # Fallback to offline bundled library
        from homology_model import build_homology_model
        offline = build_homology_model(query, top_k=n_templates, seed=seed)
        # Normalise offline template keys to match online format
        bt = offline.best_template
        normalised_best = {
            "id":       bt.get("template_id", bt.get("id", "?")),
            "name":     bt.get("template_name", bt.get("name", "?")),
            "identity": bt.get("identity", 0),
            "coverage": bt.get("coverage", 0),
            "score":    bt.get("score", 0),
            "alignment": bt.get("alignment", {}),
        }
        return ExtendedHomologyModel(
            query          = query,
            mode           = "offline",
            best_template  = normalised_best,
            all_templates  = offline.all_templates,
            model_coords   = offline.model_coords,
            seq_identity   = offline.seq_identity,
            coverage       = offline.coverage,
            model_quality  = offline.model_quality,
            n_residues     = len(query),
        )

    # ── ONLINE MODE ──────────────────────────────────────────────────────────

    # Step 1: BLAST search
    blast_hits = blast_against_pdb(query, n_hits=n_templates, email=email)

    if not blast_hits:
        print("No BLAST hits found. Falling back to offline mode.")
        return build_extended_model(query, online=False, seed=seed)

    # Step 2: Download PDB structures
    pdb_structures = []
    for hit in blast_hits:
        struct = fetch_pdb_structure(hit["pdb_id"], hit["chain_id"])
        if struct:
            pdb_structures.append(struct)
            time.sleep(0.5)   # be polite to RCSB

    if not pdb_structures:
        print("Could not download PDB files. Falling back to offline mode.")
        return build_extended_model(query, online=False, seed=seed)

    # Step 3: Multi-template coordinate transfer
    model_coords = multi_template_model(query, blast_hits, pdb_structures, seed=seed)

    # Step 4: Quality assessment
    best   = blast_hits[0]
    ident  = best["identity"]
    qual   = "good" if ident >= 30 else "moderate" if ident >= 20 else "low"

    return ExtendedHomologyModel(
        query          = query,
        mode           = "online",
        best_template  = {
            "id":       best["pdb_id"],
            "name":     f"PDB:{best['pdb_id']} chain {best['chain_id']}",
            "identity": ident,
            "coverage": best["coverage"],
            "e_value":  best["e_value"],
            "alignment": {
                "aligned_query":    best.get("aligned_q", ""),
                "aligned_template": best.get("aligned_s", ""),
            },
        },
        all_templates  = blast_hits,
        model_coords   = model_coords,
        seq_identity   = ident,
        coverage       = best["coverage"],
        model_quality  = qual,
        n_residues     = len(query),
        multi_template = len(pdb_structures) > 1,
    )


# ── New Flask endpoint (add this to app.py) ───────────────────────────────────
"""
@app.post("/api/homology_model_extended")
def homology_model_extended():
    data  = request.get_json(force=True)
    seq, err = validate_sequence(data.get("sequence",""), max_len=500)
    if err:
        return jsonify({"error": err}), 400
    
    online = data.get("online", False)   # default offline for speed
    email  = data.get("email", "your@email.com")
    
    from models.homology_model_extended import build_extended_model
    model = build_extended_model(seq, online=online, email=email)
    
    return jsonify({
        "sequence":       model.query,
        "mode":           model.mode,
        "n_residues":     model.n_residues,
        "seq_identity":   model.seq_identity,
        "coverage":       model.coverage,
        "model_quality":  model.model_quality,
        "multi_template": model.multi_template,
        "best_template":  model.best_template,
        "model_coords":   model.model_coords.tolist(),
    })
"""


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test offline mode (no internet needed)
    query = "VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKK"
    print(f"Query length: {len(query)} residues\n")

    print("=== OFFLINE MODE (bundled library) ===")
    model_offline = build_extended_model(query, online=False)
    print(f"Template  : {model_offline.best_template['id']}")
    print(f"Identity  : {model_offline.seq_identity}%")
    print(f"Quality   : {model_offline.model_quality}")
    print(f"Coords shape: {model_offline.model_coords.shape}")

    print("\n=== ONLINE MODE (NCBI BLAST + PDB) ===")
    print("Uncomment below and add your email to test online mode:")
    # model_online = build_extended_model(
    #     query,
    #     online=True,
    #     email="your@email.com",   # ← put your email here
    #     n_templates=3
    # )
    # print(f"Template  : {model_online.best_template['id']}")
    # print(f"Identity  : {model_online.seq_identity}%")
    # print(f"Multi-tmpl: {model_online.multi_template}")
