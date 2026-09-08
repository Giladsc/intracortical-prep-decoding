"""Small shared pieces of the decoding machinery.

Deliberately minimal. The *loops* that use these - fitting one classifier per time
bin, and the train-time x test-time double loop - are written out inline in notebooks
03 and 04, because those loops are the thing being taught. What lives here is only
what would be noise if retyped in every notebook: the pipeline definition, the
cross-validation scheme with its fallback logic, and the circular error metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# The decoder
# ---------------------------------------------------------------------------
def make_pipeline(C: float = 1.0, random_state: int = 42) -> Pipeline:
    """StandardScaler -> L2-regularised multinomial logistic regression.

    Wrapping the scaler and the classifier in a single Pipeline is what makes the
    leakage discipline automatic: calling `pipe.fit(X_train, y_train)` fits the
    scaler on the training trials *only*, and `pipe.predict(X_test)` reuses those
    training means and standard deviations. Never fit a scaler on the full dataset
    before cross-validating.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            # L2 is scikit-learn's default penalty; naming it explicitly is
            # deprecated as of sklearn 1.8, so we rely on the default.
            C=C,
            solver="lbfgs",
            max_iter=2000,
            random_state=random_state,
        )),
    ])


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
@dataclass
class CVPlan:
    """A fixed set of train/test splits, reused at every time point."""
    splits: list[tuple[np.ndarray, np.ndarray]]
    scheme: str          # "StratifiedGroupKFold" or "StratifiedKFold"
    n_splits: int
    notes: list[str]

    def __str__(self) -> str:
        lines = [f"CV scheme: {self.scheme} with {self.n_splits} folds"]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _folds_are_usable(splits, y: np.ndarray, n_classes: int) -> str | None:
    """None if every fold has all classes on both sides; otherwise the reason why not."""
    for i, (train_idx, test_idx) in enumerate(splits):
        if test_idx.size == 0 or train_idx.size == 0:
            return f"fold {i} is empty"
        if np.unique(y[train_idx]).size != n_classes:
            return (f"fold {i} training set is missing "
                    f"{n_classes - np.unique(y[train_idx]).size} of {n_classes} directions")
        if np.unique(y[test_idx]).size != n_classes:
            return (f"fold {i} test set is missing "
                    f"{n_classes - np.unique(y[test_idx]).size} of {n_classes} directions")
    return None


def make_cv(y: np.ndarray, groups: np.ndarray, n_splits: int, random_state: int,
            n_classes: int = 8) -> CVPlan:
    """Build the splits once, so every time point is scored on identical folds.

    Preferred scheme is StratifiedGroupKFold with recording block as the group: no
    block ever appears in both training and test, so above-chance accuracy cannot be
    explained by slow block-specific drifts in the neural signal, and the result
    speaks to generalisation across recording sessions.

    Falls back to StratifiedKFold *loudly* if the block structure cannot support it -
    the scheme name is carried in `CVPlan.scheme` and written into the output table,
    so a within-block result can never be mistaken for a cross-block one.
    """
    notes: list[str] = []
    unique_groups = np.unique(groups)
    n_groups = unique_groups.size

    if n_groups >= 2:
        effective = min(n_splits, n_groups)
        if effective < n_splits:
            notes.append(
                f"n_splits reduced from {n_splits} to {effective}: only {n_groups} "
                f"recording blocks are available to group by."
            )
        cv = StratifiedGroupKFold(n_splits=effective, shuffle=True,
                                  random_state=random_state)
        splits = [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y, groups=groups)]
        problem = _folds_are_usable(splits, y, n_classes)
        if problem is None:
            notes.append(f"grouped by {n_groups} recording blocks: "
                         f"{unique_groups.astype(int).tolist()}")
            return CVPlan(splits=splits, scheme="StratifiedGroupKFold",
                          n_splits=effective, notes=notes)
        reason = (f"StratifiedGroupKFold could not produce usable folds ({problem}).")
    else:
        reason = (f"only {n_groups} recording block(s) present, so trials cannot be "
                  f"grouped by block.")

    print("WARNING: falling back to StratifiedKFold - " + reason)
    print("         Folds are NOT grouped by recording block, so decoding accuracy may")
    print("         partly reflect block-specific neural drift rather than a directional")
    print("         code that generalises across sessions. Interpret accordingly.")
    notes.append("FALLBACK: " + reason)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = [(tr, te) for tr, te in cv.split(np.zeros(len(y)), y)]
    return CVPlan(splits=splits, scheme="StratifiedKFold", n_splits=n_splits,
                  notes=notes)


def describe_folds(plan: CVPlan, y: np.ndarray, groups: np.ndarray,
                   n_classes: int = 8) -> "object":
    """A small per-fold table: sizes, blocks held out, and test class counts."""
    import pandas as pd
    rows = []
    for i, (train_idx, test_idx) in enumerate(plan.splits):
        counts = np.bincount(y[test_idx], minlength=n_classes)
        rows.append({
            "fold": i,
            "n_train": train_idx.size,
            "n_test": test_idx.size,
            "test_blocks": ",".join(str(int(b)) for b in np.unique(groups[test_idx])),
            "test_dir_counts": counts.tolist(),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Circular error metric
# ---------------------------------------------------------------------------
def angular_error_deg(y_true: np.ndarray, y_pred: np.ndarray,
                      n_directions: int = 8) -> np.ndarray:
    """Smallest angle between the true and predicted direction, per trial, in degrees.

    delta = min(|theta_true - theta_pred|, 2*pi - |theta_true - theta_pred|)

    Accuracy treats every mistake alike. This does not: confusing a direction with
    its neighbour (45 deg) is a very different failure from predicting the exact
    opposite direction (180 deg), and only the second is evidence that the decoder
    has learnt nothing about the geometry. Chance for uniform 8-class guessing is
    90 deg.
    """
    step = 360.0 / n_directions
    diff = np.abs(y_true.astype(float) - y_pred.astype(float)) * step
    return np.minimum(diff, 360.0 - diff)


def chance_angular_error_deg(n_directions: int = 8) -> float:
    """Mean angular error expected from uniform random guessing (90 deg for n=8)."""
    step = 360.0 / n_directions
    offsets = np.arange(n_directions) * step
    return float(np.minimum(offsets, 360.0 - offsets).mean())


# ---------------------------------------------------------------------------
# Permutation testing  (used only when config.RUN_PERMUTATIONS is True)
# ---------------------------------------------------------------------------
def permute_labels_within_blocks(y: np.ndarray, groups: np.ndarray,
                                 rng: np.random.Generator) -> np.ndarray:
    """Shuffle direction labels *within* each recording block.

    Permuting globally would break the association between direction and block as
    well as the one between direction and neural activity, making the null too easy
    to beat. Shuffling within block destroys only the direction/activity link, which
    is the one the hypothesis is about.
    """
    y_perm = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        y_perm[idx] = rng.permutation(y[idx])
    return y_perm
