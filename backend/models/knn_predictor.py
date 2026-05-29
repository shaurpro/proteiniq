"""
knn_predictor.py
----------------
KNN-based protein secondary structure predictor using sliding context windows.

Concept (from Lahiri et al., 2024, Ch. 22):
  Each residue is NOT classified in isolation — instead, a context window of
  W neighboring residues (centred on the target) is used as input to a KNN
  classifier. This mirrors human context-based reading (Fig 22.2 in the text).

Input:  amino acid sequence string (single-letter codes)
Output: per-residue secondary structure labels — H (helix), E (strand), C (coil)
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import joblib
import os

# ── Amino acid alphabet & encoding ──────────────────────────────────────────

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX   = {aa: i for i, aa in enumerate(AA_ALPHABET)}
GAP_TOKEN   = "-"          # used for padding at sequence edges
N_AA        = len(AA_ALPHABET) + 1   # 20 AAs + 1 gap


def one_hot_encode_aa(aa: str) -> np.ndarray:
    """Return a (21,) one-hot vector for a single amino acid or gap."""
    vec = np.zeros(N_AA, dtype=np.float32)
    if aa in AA_TO_IDX:
        vec[AA_TO_IDX[aa]] = 1.0
    else:
        vec[-1] = 1.0   # gap / unknown
    return vec


# ── Context window encoding ──────────────────────────────────────────────────

def encode_context_window(sequence: str, i: int, W: int = 9) -> np.ndarray:
    """
    Build the feature vector for residue at position i.

    The window spans [i - half, i + half] (inclusive), padded with gaps at
    sequence boundaries.  Each residue → 21-dim one-hot → concatenated.

    Returns np.ndarray of shape (W * N_AA,)
    """
    half = W // 2
    features = []
    for offset in range(-half, half + 1):
        pos = i + offset
        if 0 <= pos < len(sequence):
            aa = sequence[pos]
        else:
            aa = GAP_TOKEN
        features.append(one_hot_encode_aa(aa))
    return np.concatenate(features)   # shape: (W * 21,)


def encode_sequence(sequence: str, W: int = 9) -> np.ndarray:
    """
    Encode every residue in the sequence with its context window.

    Returns np.ndarray of shape (len(sequence), W * N_AA)
    """
    return np.array([encode_context_window(sequence, i, W) for i in range(len(sequence))])


# ── Model wrapper ────────────────────────────────────────────────────────────

class KNNSSPredictor:
    """
    Secondary structure predictor backed by sklearn KNeighborsClassifier.

    Usage
    -----
    predictor = KNNSSPredictor(n_neighbors=5, window_size=9)
    predictor.fit(sequences, labels)          # train
    pred = predictor.predict("ACDEFGHIKLMN") # predict
    """

    def __init__(self, n_neighbors: int = 5, window_size: int = 9,
                 metric: str = "euclidean"):
        self.n_neighbors  = n_neighbors
        self.window_size  = window_size
        self.metric       = metric
        self.knn          = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            n_jobs=-1
        )
        self.le           = LabelEncoder()
        self._fitted      = False

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, sequences: list[str], labels: list[str]):
        """
        Train on a list of sequences and their per-residue SS label strings.

        sequences : ["ACDEF...", "MNPQR..."]
        labels    : ["HHHCC...", "EEHCC..."]  (same length as each sequence)
        """
        X_list, y_list = [], []
        for seq, lab in zip(sequences, labels):
            assert len(seq) == len(lab), "Sequence and label length must match"
            X_list.append(encode_sequence(seq, self.window_size))
            y_list.extend(list(lab))

        X = np.vstack(X_list)
        y = np.array(y_list)

        y_enc = self.le.fit_transform(y)
        self.knn.fit(X, y_enc)
        self._fitted = True
        return self

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self, sequence: str) -> str:
        """Predict secondary structure for a single sequence string."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        X = encode_sequence(sequence, self.window_size)
        y_enc = self.knn.predict(X)
        return "".join(self.le.inverse_transform(y_enc))

    def predict_proba(self, sequence: str) -> np.ndarray:
        """Return class probabilities shape (len(seq), n_classes)."""
        X = encode_sequence(sequence, self.window_size)
        return self.knn.predict_proba(X)

    # ── Cross-validation Q3 accuracy ─────────────────────────────────────────

    def cross_validate(self, sequences: list[str], labels: list[str],
                       cv: int = 5) -> dict:
        X_list, y_list = [], []
        for seq, lab in zip(sequences, labels):
            X_list.append(encode_sequence(seq, self.window_size))
            y_list.extend(list(lab))
        X = np.vstack(X_list)
        y = self.le.fit_transform(np.array(y_list))
        scores = cross_val_score(self.knn, X, y, cv=cv, scoring="accuracy")
        return {"mean_q3": float(scores.mean()), "std": float(scores.std()),
                "per_fold": scores.tolist()}

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "models/knn_ss.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"knn": self.knn, "le": self.le,
                     "W": self.window_size}, path)

    @classmethod
    def load(cls, path: str = "models/knn_ss.pkl") -> "KNNSSPredictor":
        data = joblib.load(path)
        obj = cls(window_size=data["W"])
        obj.knn = data["knn"]
        obj.le  = data["le"]
        obj._fitted = True
        return obj


# ── Chou-Fasman fallback (no training data needed) ──────────────────────────

# Chou-Fasman propensities (simplified) for quick demo / unit-test
CF_HELIX = dict(A=1.42,L=1.21,M=1.45,F=1.13,W=1.08,K=1.16,Q=1.11,
                E=1.51,S=0.77,P=0.57,V=1.06,I=1.08,C=0.70,Y=0.69,
                H=1.00,D=1.01,N=0.67,T=0.83,G=0.57,R=0.98)
CF_STRAND = dict(A=0.83,L=1.30,M=1.05,F=1.38,W=1.37,K=0.74,Q=1.10,
                 E=0.37,S=0.75,P=0.55,V=1.70,I=1.60,C=1.19,Y=1.47,
                 H=0.87,D=0.54,N=0.89,T=1.19,G=0.75,R=0.93)

def chou_fasman_predict(sequence: str, window: int = 4) -> str:
    """Fallback predictor using Chou-Fasman propensities — no training needed."""
    result = []
    for i, aa in enumerate(sequence):
        win = sequence[max(0, i-window): i+window+1]
        ph = np.mean([CF_HELIX.get(a, 1.0) for a in win])
        pe = np.mean([CF_STRAND.get(a, 1.0) for a in win])
        if ph > 1.03 and ph > pe:
            result.append("H")
        elif pe > 1.05:
            result.append("E")
        else:
            result.append("C")
    return "".join(result)


# ── Quick demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    seq = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMN"
    pred_cf = chou_fasman_predict(seq)
    print(f"Sequence : {seq}")
    print(f"CF Pred  : {pred_cf}")

    # Toy training example
    train_seqs   = ["AAAAALLLLEEEE", "GGGGPPPPSSSSS", "VVVVIIIIFFFFF"]
    train_labels = ["HHHHHHHHHEEEE", "CCCCCCCCCEEEEE"[:13], "EEEEEEEEHHHHH"]
    predictor = KNNSSPredictor(n_neighbors=3, window_size=5)
    predictor.fit(train_seqs, train_labels)
    print(f"KNN Pred : {predictor.predict(seq[:13])}")
