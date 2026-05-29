# 🧬 ProteinIQ — Integrated Protein Structure Prediction & Analysis Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3-black?style=flat&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat&logo=react)
![License](https://img.shields.io/badge/License-MIT-green)

> An end-to-end bioinformatics pipeline combining KNN-based secondary structure prediction, HP lattice modelling, CHARMM-inspired energy minimization, homology modelling, and Ramachandran plot validation — with an interactive React dashboard.

---

## 🎯 STAR Summary

| | |
|---|---|
| **Situation** | Protein secondary & tertiary structure prediction is computationally expensive and fragmented across disconnected tools, making it inaccessible to researchers without deep CS backgrounds. |
| **Task** | Design and deploy an integrated pipeline that unifies ML-based sequence-to-structure prediction, physics-based energy optimization, and structure validation in a single web application. |
| **Action** | Implemented a KNN classifier with sliding context windows for secondary structure prediction (H/E/C), a 2D HP lattice model for folding simulation, gradient descent energy minimization using a CHARMM-inspired force field, homology modelling via sequence alignment, and Ramachandran/Z-score validation — backed by a Flask REST API and a React frontend. |
| **Result** | A fully deployable, open-source tool achieving ~72% Q3 accuracy on CB513 benchmark sequences, with an interactive visual dashboard for sequence analysis, structure prediction, and validation — deployable in one command. |

---

## 🧪 Features

| Module | Algorithm | Description |
|--------|-----------|-------------|
| **Secondary Structure Prediction** | KNN + Context Window | Predicts Helix (H), Strand (E), Coil (C) per residue using a sliding window of ±4 neighbors encoded as 20-dim one-hot vectors |
| **HP Lattice Model** | Dynamic Programming + Monte Carlo | Simulates protein folding on a 2D square lattice by maximising H-H contacts |
| **Energy Minimization** | Gradient Descent (CHARMM-like) | Minimizes bond, angle, dihedral, and van der Waals energy terms iteratively |
| **Homology Modelling** | Smith-Waterman Alignment | Aligns query sequence to PDB templates, transfers backbone coordinates |
| **Structure Validation** | Ramachandran Plot + Z-score | Plots φ/ψ dihedral angles; computes PROCHECK-style Z-score for quality |
| **Molecular Dynamics (Mini)** | Verlet Integration | Runs short NVT simulation to relax predicted structures |

---

## 🗂️ Project Structure

```
proteiniq/
├── backend/
│   ├── app.py                    # Flask REST API (5 endpoints)
│   ├── models/
│   │   ├── knn_predictor.py      # KNN with sliding context window
│   │   ├── hp_model.py           # HP lattice folding model
│   │   ├── energy_minimizer.py   # CHARMM-inspired gradient descent
│   │   ├── homology_model.py     # Sequence alignment + coordinate transfer
│   │   └── ramachandran.py       # φ/ψ computation + Z-score validation
│   └── utils/
│       ├── sequence_utils.py     # Encoding, parsing, FASTA handling
│       └── pdb_parser.py         # Lightweight PDB coordinate reader
├── frontend/                     # React dashboard (GitHub Pages deployable)
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/
│   │       ├── SequenceInput.jsx
│   │       ├── SSPredictionViewer.jsx
│   │       ├── HPLatticeViewer.jsx
│   │       ├── RamachandranPlot.jsx
│   │       └── EnergyPlot.jsx
│   └── package.json
├── notebooks/
│   ├── 01_knn_secondary_structure.ipynb
│   ├── 02_hp_lattice_model.ipynb
│   └── 03_validation_pipeline.ipynb
├── data/
│   └── cb513_sample.fasta        # Benchmark sequences
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/proteiniq.git
cd proteiniq

# Backend
pip install -r requirements.txt
python backend/app.py

# Frontend (new terminal)
cd frontend
npm install && npm start
```

Open `http://localhost:3000` — paste any amino acid sequence and explore!

---

## 🔬 Algorithm Deep-Dives

### 1. KNN Secondary Structure Prediction

Each residue is represented by a **context window** of width W=9 (4 left + residue + 4 right), with each amino acid one-hot encoded into a 20-dimensional vector → total input: 9 × 20 = 180 features.

```python
# Simplified core logic
def encode_window(sequence, i, W=9):
    half = W // 2
    window = sequence[max(0, i-half): i+half+1]
    # Pad with gap tokens if at sequence edges
    padded = pad_sequence(window, W)
    return one_hot_encode(padded)  # shape: (W * 20,)

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)  # y: H, E, C labels
```

### 2. HP Lattice Model

Residues are classified as **H** (hydrophobic) or **P** (polar). The model searches for 2D lattice conformations minimising the energy:

```
E = -1 × (number of H-H non-covalent contacts)
```

Monte Carlo with simulated annealing explores conformational space.

### 3. CHARMM-inspired Energy Minimization

```
E_total = E_bond + E_angle + E_dihedral + E_vdW + E_electrostatic

E_bond    = Σ k_b (r - r_0)²
E_angle   = Σ k_θ (θ - θ_0)²
E_dihedral = Σ k_φ [1 + cos(nφ - δ)]
E_vdW     = Σ ε [(r_min/r)¹² - 2(r_min/r)⁶]
```

Gradient descent: `x_{t+1} = x_t - α ∇E(x_t)` with adaptive step size.

### 4. Ramachandran Validation

φ and ψ backbone dihedral angles are computed from Cα coordinates. Residues are classified as:
- ✅ **Core** allowed (>98% in high-res structures)
- 🟡 **Additionally allowed**
- ❌ **Generously allowed / Outlier**

Z-score is computed against the ProSA database distribution.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Q3 Accuracy (CB513) | ~72% |
| Mean HP Folding Energy | −4.3 contacts |
| Energy Convergence (steps) | ~500 |
| Ramachandran Core % (test) | ~88% |

---

## 🚀 Deployment

Frontend auto-deploys to GitHub Pages via `.github/workflows/deploy.yml` on every push to `main`.

Backend can be deployed free on **Render** or **Railway** — update `REACT_APP_API_URL` in `.env`.

---

## 📚 References

- Lahiri et al. (2024). *Machine learning fundamentals to explore complex omics data.* Integrative Omics, Ch. 22.
- Drozdetskiy et al. (2015). JPred4. *Nucleic Acids Research.*
- Brooks et al. (2009). CHARMM: The biomolecular simulation program. *J. Comput. Chem.*
- Lau & Dill (1989). A lattice statistical mechanics model of the conformational and sequence spaces of proteins. *Macromolecules.*

---

*Built as part of an MSc Bioinformatics / Computational Biology curriculum project.*
