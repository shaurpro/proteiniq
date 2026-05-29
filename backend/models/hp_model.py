"""
hp_model.py
-----------
2D HP (Hydrophobic-Polar) Lattice Model for protein folding simulation.

The HP model (Lau & Dill, 1989) simplifies the protein folding problem:
  - Each residue is either H (hydrophobic) or P (polar)
  - Protein is placed on a 2D square lattice
  - Energy = -1 per H-H non-covalent (topological) contact
  - Goal: find conformation minimising total energy

Solves using Monte Carlo with Simulated Annealing.
"""

import numpy as np
import random
import math
from dataclasses import dataclass, field
from typing import Optional

# ── HP Classification ────────────────────────────────────────────────────────

HYDROPHOBIC = set("ACFILMVWY")   # strongly hydrophobic amino acids
POLAR       = set("DEHKNQRST") | set("GP")

def to_hp_string(sequence: str) -> str:
    """Convert amino acid sequence to H/P string."""
    return "".join("H" if aa in HYDROPHOBIC else "P" for aa in sequence.upper())


# ── Conformation representation ──────────────────────────────────────────────

MOVES = {
    "R": (1,  0),
    "L": (-1, 0),
    "U": (0,  1),
    "D": (0, -1),
}
DIRECTIONS = list(MOVES.values())


@dataclass
class Conformation:
    positions: list[tuple[int,int]] = field(default_factory=list)
    energy: float = 0.0

    def is_valid(self) -> bool:
        """No two residues occupy the same lattice cell."""
        return len(self.positions) == len(set(self.positions))

    def compute_energy(self, hp_string: str) -> float:
        """Energy = -1 for each H-H non-covalent contact (topological neighbour)."""
        pos_set = {pos: i for i, pos in enumerate(self.positions)}
        energy  = 0.0
        for i, (x, y) in enumerate(self.positions):
            if hp_string[i] != "H":
                continue
            for dx, dy in DIRECTIONS:
                nb = (x+dx, y+dy)
                if nb in pos_set:
                    j = pos_set[nb]
                    # Non-covalent neighbour (not sequence-adjacent)
                    if abs(j - i) > 1:
                        energy -= 0.5   # count each pair once → -0.5 each side
        self.energy = energy
        return energy


# ── Move operators ───────────────────────────────────────────────────────────

def end_move(conf: Conformation, idx: int) -> Conformation:
    """Move an end residue to an adjacent free cell."""
    positions = list(conf.positions)
    if idx == 0:
        anchor = positions[1]
    else:
        anchor = positions[-2]
    ax, ay = anchor
    candidates = [(ax+dx, ay+dy) for dx,dy in DIRECTIONS
                  if (ax+dx, ay+dy) not in set(positions)]
    if not candidates:
        return conf
    new_pos = random.choice(candidates)
    if idx == 0:
        positions[0] = new_pos
    else:
        positions[-1] = new_pos
    return Conformation(positions=positions)


def corner_move(conf: Conformation, idx: int) -> Conformation:
    """Crankshaft / corner move for an internal residue."""
    positions = list(conf.positions)
    if idx <= 0 or idx >= len(positions) - 1:
        return conf
    prev, curr, nxt = positions[idx-1], positions[idx], positions[idx+1]
    # Valid corner move: prev and nxt differ by one step in each axis
    dx = nxt[0] - prev[0]
    dy = nxt[1] - prev[1]
    if abs(dx) + abs(dy) != 2:
        return conf
    # The corner position
    corner = (prev[0] + dx//2*2 if dx != 0 else prev[0],
              prev[1] + dy//2*2 if dy != 0 else prev[1])
    # Mirror of curr through the diagonal
    new_curr = (prev[0] + (nxt[0]-prev[0]) - (curr[0]-prev[0]),
                prev[1] + (nxt[1]-prev[1]) - (curr[1]-prev[1]))
    if new_curr in set(positions):
        return conf
    positions[idx] = new_curr
    return Conformation(positions=positions)


# ── Initial conformation (extended chain) ────────────────────────────────────

def extended_conformation(n: int) -> Conformation:
    """Place all residues in a straight line."""
    return Conformation(positions=[(i, 0) for i in range(n)])


# ── Monte Carlo Simulated Annealing ─────────────────────────────────────────

def monte_carlo_fold(hp_string: str,
                     T_start:   float = 5.0,
                     T_end:     float = 0.01,
                     steps:     int   = 10_000,
                     seed:      Optional[int] = 42) -> dict:
    """
    Fold the HP sequence using simulated annealing.

    Returns
    -------
    dict with keys:
        best_conformation : Conformation
        energy_trace      : list of (step, energy) for plotting
        temperature_trace : list of T values
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n       = len(hp_string)
    current = extended_conformation(n)
    current.compute_energy(hp_string)
    best    = Conformation(positions=list(current.positions),
                           energy=current.energy)

    energy_trace = [(0, current.energy)]
    temp_trace   = [T_start]
    alpha        = (T_end / T_start) ** (1 / steps)   # exponential cooling

    T = T_start
    for step in range(1, steps + 1):
        # Choose a random move
        idx   = random.randint(0, n-1)
        mover = random.choice([end_move, corner_move])
        new_conf = mover(current, idx)

        if not new_conf.is_valid():
            T *= alpha
            continue

        new_E = new_conf.compute_energy(hp_string)
        dE    = new_E - current.energy

        if dE < 0 or random.random() < math.exp(-dE / max(T, 1e-10)):
            current = new_conf
            if current.energy < best.energy:
                best = Conformation(positions=list(current.positions),
                                    energy=current.energy)

        T *= alpha

        if step % max(1, steps // 200) == 0:
            energy_trace.append((step, current.energy))
            temp_trace.append(T)

    energy_trace.append((steps, best.energy))

    return {
        "best_conformation": best,
        "energy_trace":      energy_trace,
        "temperature_trace": temp_trace,
        "final_energy":      best.energy,
        "hp_string":         hp_string,
        "n_contacts":        int(-best.energy * 2),
    }


# ── Serialisation helper for API ─────────────────────────────────────────────

def conformation_to_dict(conf: Conformation, hp_string: str) -> dict:
    return {
        "positions": conf.positions,
        "energy":    conf.energy,
        "types":     list(hp_string),   # H or P per residue
    }


# ── Quick demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seq = "HPPHPPHPPHHPPHHPPHPP"
    print(f"HP string : {seq}")
    result = monte_carlo_fold(seq, steps=5000)
    print(f"Best energy    : {result['final_energy']}")
    print(f"H-H contacts   : {result['n_contacts']}")
    print(f"Final positions: {result['best_conformation'].positions[:5]}...")
