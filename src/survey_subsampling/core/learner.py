#!/usr/bin/env python
"""Defines the learner dataclass."""

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class Learner:
    """Data class to record and present statistics from trained & evaluated models."""

    dx: str  # Diagnosis
    hc_n: int  # Number of healthy controls
    dx_n: int  # Number of patients
    x_ids: np.ndarray = field(
        default_factory=lambda: np.empty(())
    )  # List of features used in the learner
    fi: List = field(default_factory=lambda: list())  # Feature importance lists
    f1: np.ndarray = field(default_factory=lambda: np.empty(()))  # F1 score lists
    sen: np.ndarray = field(
        default_factory=lambda: np.empty(())
    )  # Sensitivity score lists
    spe: np.ndarray = field(
        default_factory=lambda: np.empty(())
    )  # Specificity score lists
    LRp: np.ndarray = field(
        default_factory=lambda: np.empty(())
    )  # Positive likelihood ratio lists
    LRn: np.ndarray = field(
        default_factory=lambda: np.empty(())
    )  # Negative likelihood ratio lists
    acc_train: np.ndarray = field(
        default_factory=lambda: np.empty(())
    )  # Performance on the training set
    acc_valid: np.ndarray = field(
        default_factory=lambda: np.empty(())
    )  # Performance on the validation set

    proba: np.ndarray = field(
        default_factory=lambda: np.empty(())
    )  # Prediction probability/confidence
    label: np.ndarray = field(default_factory=lambda: np.empty(()))  # Prediction labels

    def summary(self) -> pd.DataFrame:
        """Constructs a dataframe from the models and prints a summary report."""
        self._sanitize()

        tmpdict = [
            {
                "Dx": self.dx,
                "N_HC": self.hc_n,
                "N_Pt": self.dx_n,
                "N_xs": len(self.x_ids),
                "F1": np.mean(self.f1),
                "Sensitivity": np.mean(self.sen),
                "Specificity": np.mean(self.spe),
                "LR+": np.mean(self.LRp),
                "LR-": np.mean(self.LRn),
                "Accuracy (Train)": np.mean(self.acc_train),
                "Accuracy (Validation)": np.mean(self.acc_valid),
            }
        ]
        tmpdf = pd.DataFrame.from_dict(tmpdict)

        return tmpdf

    def _sanitize(self) -> None:
        """Make lists of lists a bit more palletable..."""
        self.proba = np.vstack(self.proba)  # type: ignore[call-overload]
        self.label = np.hstack(self.label)  # type: ignore[call-overload]
