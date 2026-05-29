"""
energy_minimizer.py
-------------------
CHARMM-inspired molecular energy minimization via gradient descent.

Energy function components (Brooks et al., 2009):
    E_total = E_bond + E_angle + E_dihedral + E_vdW + E_electrostatic

Gradient descent update (Lahiri et al., 2024 — Eq. referenced in text):
    x_{t+1} = x_t - α * ∇E(x_t)

This module provides:
  1. A simplified CHARMM-like energy function over Cα coordinates
  2. Finite-difference gradient estimation
  3. Gradient descent with adaptive step size (Armijo line search)
  4. Energy trace for visualisation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

# ── CHARMM-like force-field parameters (simplified) ─────────────────────────

# Bond stretching: E = k_b * (r - r0)^2
BOND_PARAMS = {"k_b": 100.0, "r0": 3.8}   # Cα-Cα pseudo-bond in Angstroms

# Bond angle: E = k_θ * (θ - θ0)^2
ANGLE_PARAMS = {"k_theta": 40.0, "theta0": np.deg2rad(110)}  # ~110° for Cα

# Dihedral (torsion): E = k_φ * [1 + cos(n*φ - δ)]
DIHEDRAL_PARAMS = {"k_phi": 1.0, "n": 1, "delta": np.deg2rad(180)}

# van der Waals (Lennard-Jones): E = ε [(rmin/r)^12 - 2*(rmin/r)^6]
VDW_PARAMS = {"epsilon": 0.15, "rmin": 4.0}  # Angstroms

# Electrostatics: E = q_i*q_j / (ε_r * r)
ELEC_PARAMS = {"epsilon_r": 80.0}   # water dielectric


# ── Energy term functions ────────────────────────────────────────────────────

def e_bond(coords: np.ndarray) -> float:
    """Bond stretching energy for Cα pseudo-chain."""
    kb, r0 = BOND_PARAMS["k_b"], BOND_PARAMS["r0"]
    bonds   = coords[1:] - coords[:-1]
    dists   = np.linalg.norm(bonds, axis=1)
    return float(np.sum(kb * (dists - r0) ** 2))


def e_angle(coords: np.ndarray) -> float:
    """Bond-angle bending energy."""
    k_th, th0 = ANGLE_PARAMS["k_theta"], ANGLE_PARAMS["theta0"]
    total = 0.0
    for i in range(1, len(coords) - 1):
        v1 = coords[i-1] - coords[i]
        v2 = coords[i+1] - coords[i]
        cos_th = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        cos_th = np.clip(cos_th, -1.0, 1.0)
        theta  = np.arccos(cos_th)
        total += k_th * (theta - th0) ** 2
    return total


def e_dihedral(coords: np.ndarray) -> float:
    """Dihedral (torsion) energy — approximates backbone φ/ψ."""
    kp, n, delta = (DIHEDRAL_PARAMS["k_phi"],
                    DIHEDRAL_PARAMS["n"],
                    DIHEDRAL_PARAMS["delta"])
    total = 0.0
    for i in range(len(coords) - 3):
        b1 = coords[i+1] - coords[i]
        b2 = coords[i+2] - coords[i+1]
        b3 = coords[i+3] - coords[i+2]
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        norm1, norm2 = np.linalg.norm(n1), np.linalg.norm(n2)
        if norm1 < 1e-9 or norm2 < 1e-9:
            continue
        cos_phi = np.dot(n1, n2) / (norm1 * norm2)
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi = np.arccos(cos_phi)
        total += kp * (1 + np.cos(n * phi - delta))
    return total


def e_vdw(coords: np.ndarray) -> float:
    """Lennard-Jones van der Waals energy (non-bonded, skip 1-2 and 1-3)."""
    eps, rmin = VDW_PARAMS["epsilon"], VDW_PARAMS["rmin"]
    total = 0.0
    n = len(coords)
    for i in range(n):
        for j in range(i + 3, n):   # skip bonded neighbours
            r = np.linalg.norm(coords[i] - coords[j])
            r = max(r, 1.0)   # avoid singularity
            ratio = rmin / r
            total += eps * (ratio**12 - 2 * ratio**6)
    return total


def e_electrostatic(coords: np.ndarray, charges: np.ndarray) -> float:
    """Coulomb electrostatics with solvent dielectric."""
    er = ELEC_PARAMS["epsilon_r"]
    total = 0.0
    n = len(coords)
    for i in range(n):
        for j in range(i + 3, n):
            r = np.linalg.norm(coords[i] - coords[j])
            r = max(r, 1.0)
            total += charges[i] * charges[j] / (er * r)
    return total * 332.0    # unit conversion kcal/mol


def total_energy(coords: np.ndarray, charges: np.ndarray) -> float:
    """CHARMM-like total energy in kcal/mol."""
    return (e_bond(coords) + e_angle(coords) +
            e_dihedral(coords) + e_vdw(coords) +
            e_electrostatic(coords, charges))


# ── Finite-difference gradient ───────────────────────────────────────────────

def numerical_gradient(coords: np.ndarray, charges: np.ndarray,
                        h: float = 1e-4) -> np.ndarray:
    """Compute gradient ∇E via central finite differences."""
    grad = np.zeros_like(coords)
    for i in range(coords.shape[0]):
        for d in range(coords.shape[1]):
            coords[i, d] += h
            E_plus  = total_energy(coords, charges)
            coords[i, d] -= 2 * h
            E_minus = total_energy(coords, charges)
            coords[i, d] += h
            grad[i, d] = (E_plus - E_minus) / (2 * h)
    return grad


# ── Gradient Descent with Armijo line search ─────────────────────────────────

@dataclass
class MinimizationResult:
    final_coords:  np.ndarray
    final_energy:  float
    energy_trace:  list[float] = field(default_factory=list)
    grad_norm_trace: list[float] = field(default_factory=list)
    converged:     bool = False
    n_steps:       int  = 0


def gradient_descent(
    coords_init: np.ndarray,
    charges:     np.ndarray,
    max_steps:   int   = 500,
    alpha_init:  float = 0.01,
    tol:         float = 1e-4,
    log_every:   int   = 10,
) -> MinimizationResult:
    """
    Minimise total energy via gradient descent with simple backtracking.

    Parameters
    ----------
    coords_init : (N, 3) Cα coordinates in Angstroms
    charges     : (N,)   partial charges per residue
    max_steps   : maximum gradient descent iterations
    alpha_init  : initial step size
    tol         : gradient norm convergence threshold
    log_every   : record energy every this many steps
    """
    coords = coords_init.copy().astype(np.float64)
    result = MinimizationResult(
        final_coords=coords,
        final_energy=total_energy(coords, charges),
        energy_trace=[total_energy(coords, charges)],
    )

    alpha = alpha_init

    for step in range(1, max_steps + 1):
        grad       = numerical_gradient(coords, charges)
        grad_norm  = np.linalg.norm(grad)

        # Armijo backtracking line search
        E_curr  = total_energy(coords, charges)
        alpha_t = alpha
        for _ in range(20):
            coords_new = coords - alpha_t * grad
            E_new      = total_energy(coords_new, charges)
            if E_new < E_curr:
                break
            alpha_t *= 0.5

        coords = coords - alpha_t * grad

        if step % log_every == 0:
            E = total_energy(coords, charges)
            result.energy_trace.append(E)
            result.grad_norm_trace.append(float(grad_norm))

        if grad_norm < tol:
            result.converged = True
            result.n_steps   = step
            break

    result.final_coords = coords
    result.final_energy = total_energy(coords, charges)
    result.n_steps      = result.n_steps or max_steps
    return result


# ── Initialisation helpers ───────────────────────────────────────────────────

def random_coil_coords(n: int, bond_length: float = 3.8,
                        seed: int = 42) -> np.ndarray:
    """Generate random-coil Cα coordinates as starting point."""
    rng    = np.random.default_rng(seed)
    coords = np.zeros((n, 3))
    for i in range(1, n):
        direction = rng.standard_normal(3)
        direction /= np.linalg.norm(direction)
        coords[i]  = coords[i-1] + bond_length * direction
    return coords


def residue_charges(sequence: str) -> np.ndarray:
    """Assign simplified partial charges (+1 for K/R, -1 for D/E, else 0)."""
    charge_map = {"K": 1.0, "R": 1.0, "D": -1.0, "E": -1.0}
    return np.array([charge_map.get(aa, 0.0) for aa in sequence.upper()])


# ── Energy breakdown for reporting ──────────────────────────────────────────

def energy_breakdown(coords: np.ndarray, charges: np.ndarray) -> dict:
    return {
        "bond":          round(e_bond(coords), 4),
        "angle":         round(e_angle(coords), 4),
        "dihedral":      round(e_dihedral(coords), 4),
        "vdw":           round(e_vdw(coords), 4),
        "electrostatic": round(e_electrostatic(coords, charges), 4),
        "total":         round(total_energy(coords, charges), 4),
    }


# ── Quick demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seq     = "ACDEFGHIKLMN"
    n       = len(seq)
    coords  = random_coil_coords(n)
    charges = residue_charges(seq)

    print("Initial energy breakdown:")
    print(energy_breakdown(coords, charges))

    result = gradient_descent(coords, charges, max_steps=200, log_every=20)
    print(f"\nConverged: {result.converged} in {result.n_steps} steps")
    print(f"Final energy: {result.final_energy:.4f} kcal/mol")
    print(f"Energy trace (first 5): {result.energy_trace[:5]}")
