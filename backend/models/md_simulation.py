"""
md_simulation.py
----------------
Molecular Dynamics simulation of a Cα pseudo-atom protein chain.

Physics implemented:
  - Lennard-Jones pairwise potential (non-bonded interactions)
  - Harmonic bond potential (covalent Cα-Cα pseudo-bonds)
  - Harmonic angle potential (Cα-Cα-Cα bending)
  - Velocity Verlet integration (time-reversible, symplectic)
  - NVT ensemble via Berendsen velocity-rescaling thermostat
  - Periodic boundary conditions (cubic box)

Reference:
  Frenkel & Smit, "Understanding Molecular Simulation" (2002)
  Allen & Tildesley, "Computer Simulation of Liquids" (2017)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ── Physical constants (reduced units) ───────────────────────────────────────
KB      = 1.987e-3   # kcal/(mol·K)  Boltzmann constant
TIMESTEP = 0.002     # ps  (2 fs — standard MD timestep)

# ── Force-field parameters ────────────────────────────────────────────────────
LJ_EPSILON = 0.15    # kcal/mol  (Cα pseudo-atom)
LJ_SIGMA   = 5.0     # Å         (Cα van der Waals radius × 2)
LJ_CUTOFF  = 12.0    # Å

BOND_K     = 100.0   # kcal/(mol·Å²)
BOND_R0    = 3.8     # Å  (Cα-Cα)

ANGLE_K    = 40.0    # kcal/(mol·rad²)
ANGLE_TH0  = np.deg2rad(111.0)


# ── Energy + force functions ─────────────────────────────────────────────────

def lj_force_energy(coords: np.ndarray,
                    box: float) -> tuple[np.ndarray, float]:
    """
    Lennard-Jones pairwise forces and energy with minimum-image PBC.
    Returns (forces array shape (N,3), total_energy float).
    """
    n = len(coords)
    forces = np.zeros_like(coords)
    energy = 0.0
    cutoff2 = LJ_CUTOFF ** 2

    for i in range(n):
        for j in range(i + 3, n):   # skip 1-2 and 1-3 bonded
            rij = coords[j] - coords[i]
            # Minimum image convention
            rij -= box * np.round(rij / box)
            r2  = np.dot(rij, rij)
            if r2 > cutoff2 or r2 < 1e-6:
                continue
            r2inv = 1.0 / r2
            s2    = (LJ_SIGMA ** 2) * r2inv
            s6    = s2 ** 3
            s12   = s6 ** 2
            e     = 4.0 * LJ_EPSILON * (s12 - s6)
            energy += e
            f_mag  = 24.0 * LJ_EPSILON * r2inv * (2.0 * s12 - s6)
            f_vec  = f_mag * rij
            forces[i] -= f_vec
            forces[j] += f_vec

    return forces, energy


def bond_force_energy(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Harmonic bond stretching along the Cα chain."""
    forces = np.zeros_like(coords)
    energy = 0.0
    for i in range(len(coords) - 1):
        rij  = coords[i+1] - coords[i]
        r    = np.linalg.norm(rij) + 1e-9
        dr   = r - BOND_R0
        energy += BOND_K * dr ** 2
        f_mag   = -2.0 * BOND_K * dr / r
        f_vec   = f_mag * rij
        forces[i]   += f_vec
        forces[i+1] -= f_vec
    return forces, energy


def angle_force_energy(coords: np.ndarray) -> tuple[np.ndarray, float]:
    """Harmonic angle bending for Cα-Cα-Cα triplets."""
    forces = np.zeros_like(coords)
    energy = 0.0
    for i in range(len(coords) - 2):
        v1  = coords[i]   - coords[i+1]
        v2  = coords[i+2] - coords[i+1]
        n1, n2 = np.linalg.norm(v1)+1e-9, np.linalg.norm(v2)+1e-9
        cos_a  = np.clip(np.dot(v1,v2)/(n1*n2), -1, 1)
        theta  = np.arccos(cos_a)
        dth    = theta - ANGLE_TH0
        energy += ANGLE_K * dth ** 2
        # Gradient via chain rule
        sin_a  = np.sqrt(max(1 - cos_a**2, 1e-9))
        df     = -2.0 * ANGLE_K * dth / sin_a
        grad_i = df * (v2/n2 - cos_a * v1/n1) / n1
        grad_k = df * (v1/n1 - cos_a * v2/n2) / n2
        forces[i]   += grad_i
        forces[i+2] += grad_k
        forces[i+1] -= (grad_i + grad_k)
    return forces, energy


def total_forces(coords: np.ndarray,
                 box: float) -> tuple[np.ndarray, dict]:
    """Compute total forces and energy breakdown."""
    f_lj,  e_lj  = lj_force_energy(coords, box)
    f_b,   e_b   = bond_force_energy(coords)
    f_a,   e_a   = angle_force_energy(coords)
    return (f_lj + f_b + f_a,
            {"lj": round(e_lj,4), "bond": round(e_b,4),
             "angle": round(e_a,4), "total": round(e_lj+e_b+e_a,4)})


# ── Berendsen NVT thermostat ──────────────────────────────────────────────────

def berendsen_rescale(velocities: np.ndarray,
                      T_current: float,
                      T_target:  float,
                      tau_T:     float = 0.1,    # ps  coupling time
                      dt:        float = TIMESTEP) -> np.ndarray:
    """Rescale velocities to maintain target temperature."""
    if T_current < 1e-6:
        return velocities
    lam = np.sqrt(1.0 + (dt / tau_T) * (T_target / T_current - 1.0))
    return velocities * lam


def kinetic_temperature(velocities: np.ndarray,
                         mass: float = 110.0) -> float:
    """Instantaneous temperature from kinetic energy (equipartition)."""
    n   = len(velocities)
    dof = max(3 * n - 6, 1)      # degrees of freedom (subtract rot+trans)
    ke  = 0.5 * mass * np.sum(velocities ** 2)
    return float(2.0 * ke / (dof * KB))


# ── Velocity Verlet integrator ────────────────────────────────────────────────

@dataclass
class MDState:
    coords:    np.ndarray
    velocities: np.ndarray
    forces:    np.ndarray
    time:      float = 0.0
    step:      int   = 0


@dataclass
class MDTrajectory:
    steps:       list[int]   = field(default_factory=list)
    times:       list[float] = field(default_factory=list)
    temperatures: list[float] = field(default_factory=list)
    energies_total: list[float] = field(default_factory=list)
    energies_ke:    list[float] = field(default_factory=list)
    energies_pe:    list[float] = field(default_factory=list)
    rmsds:       list[float] = field(default_factory=list)
    snapshots:   list[np.ndarray] = field(default_factory=list)


def run_md(
    coords_init: np.ndarray,
    T_target:    float = 300.0,    # K
    n_steps:     int   = 1000,
    dt:          float = TIMESTEP,
    box:         float = 100.0,    # Å  (large box → no PBC effect for small peptides)
    mass:        float = 110.0,    # Da (average amino acid mass)
    save_every:  int   = 20,
    seed:        Optional[int] = 42,
) -> tuple[MDTrajectory, MDState]:
    """
    Run NVT molecular dynamics using velocity Verlet + Berendsen thermostat.

    Parameters
    ----------
    coords_init : (N, 3) starting Cα coordinates in Angstroms
    T_target    : target temperature in Kelvin
    n_steps     : number of MD steps
    dt          : timestep in ps
    box         : cubic box edge in Angstroms
    mass        : uniform pseudo-atom mass in Da
    save_every  : record trajectory every this many steps
    seed        : random seed for initial velocities

    Returns
    -------
    (MDTrajectory, final MDState)
    """
    rng = np.random.default_rng(seed)
    n   = len(coords_init)

    coords    = coords_init.copy().astype(np.float64)
    ref_coords = coords.copy()

    # Maxwell-Boltzmann initial velocities
    sigma_v   = np.sqrt(KB * T_target / mass)
    velocities = rng.normal(0, sigma_v, (n, 3))
    velocities -= velocities.mean(axis=0)   # remove COM drift

    forces, _ = total_forces(coords, box)

    state = MDState(coords=coords, velocities=velocities,
                    forces=forces, time=0.0, step=0)
    traj  = MDTrajectory()

    for step in range(1, n_steps + 1):
        # ── Velocity Verlet step 1: update positions ──────────────────────
        acc = state.forces / mass
        state.coords    += state.velocities * dt + 0.5 * acc * dt ** 2
        state.velocities += 0.5 * acc * dt

        # ── Compute new forces ────────────────────────────────────────────
        new_forces, e_dict = total_forces(state.coords, box)
        state.forces = new_forces

        # ── Velocity Verlet step 2: update velocities ─────────────────────
        acc_new = state.forces / mass
        state.velocities += 0.5 * acc_new * dt

        # ── Berendsen thermostat ──────────────────────────────────────────
        T_now = kinetic_temperature(state.velocities, mass)
        state.velocities = berendsen_rescale(state.velocities, T_now,
                                              T_target, dt=dt)

        state.time += dt
        state.step  = step

        # ── Record ────────────────────────────────────────────────────────
        if step % save_every == 0:
            T_rec = kinetic_temperature(state.velocities, mass)
            ke    = 0.5 * mass * np.sum(state.velocities ** 2)
            pe    = e_dict["total"]
            rmsd  = float(np.sqrt(np.mean(
                np.sum((state.coords - ref_coords) ** 2, axis=1))))

            traj.steps.append(step)
            traj.times.append(round(state.time, 4))
            traj.temperatures.append(round(T_rec, 2))
            traj.energies_ke.append(round(ke, 3))
            traj.energies_pe.append(round(pe, 3))
            traj.energies_total.append(round(ke + pe, 3))
            traj.rmsds.append(round(rmsd, 3))
            if len(traj.snapshots) < 20:    # keep max 20 snapshots
                traj.snapshots.append(state.coords.copy())

    return traj, state


# ── Serialisation helpers ─────────────────────────────────────────────────────

def trajectory_to_dict(traj: MDTrajectory) -> dict:
    return {
        "steps":        traj.steps,
        "times":        traj.times,
        "temperatures": traj.temperatures,
        "energies_total": traj.energies_total,
        "energies_ke":  traj.energies_ke,
        "energies_pe":  traj.energies_pe,
        "rmsds":        traj.rmsds,
        "n_snapshots":  len(traj.snapshots),
    }


# ── Initial velocity distribution helper ─────────────────────────────────────

def maxwell_boltzmann_speed(T: float, mass: float = 110.0,
                             n_samples: int = 500) -> np.ndarray:
    """Sample speeds from Maxwell-Boltzmann distribution for plotting."""
    rng = np.random.default_rng(0)
    sigma = np.sqrt(KB * T / mass)
    v = rng.normal(0, sigma, (n_samples, 3))
    return np.linalg.norm(v, axis=1)


# ── Quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from energy_minimizer import random_coil_coords

    seq    = "ACDEFGHIKLMN"
    coords = random_coil_coords(len(seq), seed=1)

    print(f"Running MD on {len(seq)}-residue chain...")
    print(f"  T_target=300K, 500 steps, dt={TIMESTEP}ps")

    traj, final = run_md(coords, T_target=300.0, n_steps=500,
                          save_every=10, seed=42)

    print(f"\nStep  Time(ps)  Temp(K)  E_total")
    for i in range(0, len(traj.steps), 2):
        print(f"{traj.steps[i]:5d}  {traj.times[i]:6.3f}    "
              f"{traj.temperatures[i]:6.1f}  {traj.energies_total[i]:10.2f}")
    print(f"\nFinal RMSD from start: {traj.rmsds[-1]:.2f} Å")
