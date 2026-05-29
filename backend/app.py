"""
app.py  —  ProteinIQ Flask REST API
------------------------------------
Endpoints
  POST /api/predict_ss      → KNN secondary structure prediction
  POST /api/hp_fold         → HP lattice folding (Monte Carlo SA)
  POST /api/minimize        → CHARMM energy minimisation
  POST /api/validate        → Ramachandran + Z-score validation
  GET  /api/health          → health check
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import traceback

from models.knn_predictor   import chou_fasman_predict, encode_sequence
from models.hp_model        import to_hp_string, monte_carlo_fold, conformation_to_dict
from models.energy_minimizer import (random_coil_coords, residue_charges,
                                      gradient_descent, energy_breakdown)
from models.ramachandran    import validate_structure

app = Flask(__name__)
CORS(app)   # allow React dev server on :3000


# ── Utilities ────────────────────────────────────────────────────────────────

def validate_sequence(seq: str, max_len: int = 200) -> tuple[str, str | None]:
    """Clean and validate amino acid sequence. Returns (clean_seq, error)."""
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    seq = seq.upper().strip().replace(" ", "").replace("\n", "")
    if not seq:
        return "", "Sequence is empty"
    if len(seq) > max_len:
        return "", f"Sequence too long (max {max_len} residues for demo)"
    invalid = set(seq) - valid_aa
    if invalid:
        return "", f"Invalid amino acid characters: {', '.join(sorted(invalid))}"
    return seq, None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "version": "1.0.0"})


@app.post("/api/predict_ss")
def predict_ss():
    """
    Predict secondary structure for a given amino acid sequence.

    Body: { "sequence": "ACDEFGHIKLMN" }
    Returns: {
        "sequence": "...",
        "prediction": "HHHCCCEEE...",
        "per_residue": [{"aa": "A", "ss": "H", "confidence": 0.8}, ...]
    }
    """
    try:
        data = request.get_json(force=True)
        seq, err = validate_sequence(data.get("sequence", ""))
        if err:
            return jsonify({"error": err}), 400

        # Chou-Fasman (no training required — instant)
        prediction = chou_fasman_predict(seq)

        # Build per-residue confidence (heuristic based on propensities)
        from models.knn_predictor import CF_HELIX, CF_STRAND
        per_residue = []
        for i, (aa, ss) in enumerate(zip(seq, prediction)):
            ph = CF_HELIX.get(aa, 1.0)
            pe = CF_STRAND.get(aa, 1.0)
            pc = 1.0 / (ph + pe + 0.1)
            total = ph + pe + pc
            conf  = max(ph, pe, pc) / total
            per_residue.append({
                "index":       i,
                "aa":          aa,
                "ss":          ss,
                "confidence":  round(float(conf), 3),
                "p_helix":     round(float(ph / total), 3),
                "p_strand":    round(float(pe / total), 3),
                "p_coil":      round(float(pc / total), 3),
            })

        # Summary counts
        counts = {s: prediction.count(s) for s in ("H", "E", "C")}

        return jsonify({
            "sequence":    seq,
            "prediction":  prediction,
            "per_residue": per_residue,
            "counts":      counts,
            "method":      "Chou-Fasman + context window",
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.post("/api/hp_fold")
def hp_fold():
    """
    Run HP lattice folding via Monte Carlo simulated annealing.

    Body: { "sequence": "ACDEFGHIKLMN", "steps": 5000 }
    Returns lattice positions, HP string, energy trace
    """
    try:
        data  = request.get_json(force=True)
        seq, err = validate_sequence(data.get("sequence", ""), max_len=60)
        if err:
            return jsonify({"error": err}), 400

        steps  = min(int(data.get("steps", 5000)), 20_000)
        hp_str = to_hp_string(seq)
        result = monte_carlo_fold(hp_str, steps=steps)

        # Subsample energy trace for response size
        trace = result["energy_trace"]
        if len(trace) > 100:
            step_size = len(trace) // 100
            trace = trace[::step_size]

        return jsonify({
            "sequence":      seq,
            "hp_string":     hp_str,
            "conformation":  conformation_to_dict(result["best_conformation"], hp_str),
            "final_energy":  result["final_energy"],
            "n_hh_contacts": result["n_contacts"],
            "energy_trace":  trace,
            "steps_run":     steps,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.post("/api/minimize")
def minimize():
    """
    Run CHARMM-like energy minimisation on a random-coil starting structure.

    Body: { "sequence": "ACDEFGHIKLMN", "max_steps": 300 }
    Returns energy trace and breakdown before/after minimisation.
    """
    try:
        data  = request.get_json(force=True)
        seq, err = validate_sequence(data.get("sequence", ""), max_len=40)
        if err:
            return jsonify({"error": err}), 400

        max_steps = min(int(data.get("max_steps", 300)), 500)

        coords  = random_coil_coords(len(seq))
        charges = residue_charges(seq)

        before = energy_breakdown(coords, charges)
        result = gradient_descent(coords, charges, max_steps=max_steps,
                                   log_every=max(1, max_steps // 50))
        after  = energy_breakdown(result.final_coords, charges)

        return jsonify({
            "sequence":      seq,
            "converged":     result.converged,
            "n_steps":       result.n_steps,
            "energy_before": before,
            "energy_after":  after,
            "energy_trace":  result.energy_trace,
            "reduction_pct": round(100 * (before["total"] - after["total"]) /
                                   max(abs(before["total"]), 1e-9), 2),
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.post("/api/validate")
def validate():
    """
    Ramachandran plot + Z-score validation.

    Body: { "sequence": "ACDEFGHIKLMN" }
    Internally generates a random-coil Cα trace (or accepts coords).
    Returns per-residue (φ, ψ), region classification, Z-score, grade.
    """
    try:
        data  = request.get_json(force=True)
        seq, err = validate_sequence(data.get("sequence", ""), max_len=100)
        if err:
            return jsonify({"error": err}), 400

        # Accept user-supplied coords or generate random coil
        if "coords" in data:
            coords = np.array(data["coords"], dtype=float)
        else:
            coords = random_coil_coords(len(seq))

        report = validate_structure(coords, sequence=seq)

        return jsonify({
            "sequence":        seq,
            "grade":           report.grade,
            "pct_core":        report.pct_core,
            "pct_allowed":     report.pct_allowed,
            "pct_outliers":    report.pct_outliers,
            "z_score":         report.z_score,
            "z_interpretation":report.z_interpretation,
            "counts": {
                "core":     report.n_core,
                "allowed":  report.n_allowed,
                "generous": report.n_generous,
                "outliers": report.n_outliers,
            },
            "per_residue": report.per_residue,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("🧬 ProteinIQ API — http://localhost:5000")
    app.run(debug=True, port=5000)


# ── MD Simulation endpoint ────────────────────────────────────────────────────

@app.post("/api/md_simulate")
def md_simulate():
    """
    Run NVT molecular dynamics on the given sequence.

    Body: { "sequence": "ACDEFGHIKLMN", "n_steps": 500, "temperature": 300 }
    Returns temperature, energy, and RMSD traces over time.
    """
    try:
        from models.md_simulation import run_md, trajectory_to_dict, random_coil_coords as md_coords
        from models.energy_minimizer import random_coil_coords, residue_charges

        data  = request.get_json(force=True)
        seq, err = validate_sequence(data.get("sequence", ""), max_len=30)
        if err:
            return jsonify({"error": err}), 400

        n_steps = min(int(data.get("n_steps", 500)), 1000)
        T       = float(data.get("temperature", 300.0))

        coords  = random_coil_coords(len(seq), seed=1)
        traj, final = run_md(coords, T_target=T, n_steps=n_steps,
                              save_every=max(1, n_steps//50), seed=42)

        return jsonify({
            "sequence":    seq,
            "n_steps":     n_steps,
            "temperature": T,
            "trajectory":  trajectory_to_dict(traj),
            "final_rmsd":  traj.rmsds[-1] if traj.rmsds else 0.0,
            "mean_temp":   round(np.mean(traj.temperatures), 1) if traj.temperatures else 0.0,
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── Homology modelling endpoint ───────────────────────────────────────────────

@app.post("/api/homology_model")
def homology_model():
    """
    Build a homology model via Smith-Waterman alignment + coordinate transfer.

    Body: { "sequence": "ACDEFGHIKLMN" }
    Returns template hits, alignment, sequence identity, model quality.
    """
    try:
        from models.homology_model import build_homology_model, homology_model_to_dict

        data  = request.get_json(force=True)
        seq, err = validate_sequence(data.get("sequence", ""), max_len=150)
        if err:
            return jsonify({"error": err}), 400

        model = build_homology_model(seq, top_k=3, seed=42)
        return jsonify(homology_model_to_dict(model))

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500
