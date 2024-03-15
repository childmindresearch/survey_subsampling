#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import List

import pandas as pd
import numpy as np


@dataclass
class Learner:
    """Data class to record and present statistics from trained & evaluated models"""
    dx: str  # Diagnosis
    hc_n: int  # Number of healthy controls
    dx_n: int  # Number of patients
    x_ids: List = field(default_factory=lambda: [])  # List of features used in the learner
    fi: List = field(default_factory=lambda: [])  # Feature importance lists
    f1: List = field(default_factory=lambda: [])  # F1 score lists
    sen: List = field(default_factory=lambda: [])  # Sensitivity score lists
    spe: List = field(default_factory=lambda: [])  # Specificity score lists
    LRp: List = field(default_factory=lambda: [])  # Positive likelihood ratio lists
    LRn: List = field(default_factory=lambda: [])  # Negative likelihood ratio lists
    acc_train: List = field(default_factory=lambda: [])  # Performance on the training set
    acc_valid: List = field(default_factory=lambda: [])  # Performance on the validation set

    proba: List = field(default_factory=lambda: [])  # Prediction probability/confidence
    label: List = field(default_factory=lambda: [])  # Prediction labels

    def summary(self, verbose=False):
        """Constructs a dataframe from the models and prints a summary report"""
        self._sanitize()
        
        tmpdict = [{
            'Dx': self.dx,
            'N_HC': self.hc_n,
            'N_Pt': self.dx_n,
            'N_xs': len(self.x_ids),
            'F1': np.mean(self.f1),
            'Sensitivity': np.mean(self.sen),
            'Specificity': np.mean(self.spe),
            'LR+': np.mean(self.LRp),
            'LR-': np.mean(self.LRn),
            'Accuracy (Train)': np.mean(self.acc_train),
            'Accuracy (Validation)': np.mean(self.acc_valid)
            
        }]
        tmpdf = pd.DataFrame.from_dict(tmpdict)

        if verbose:
            self._plot_probas()
            print(tmpdf)

        return tmpdf
    
    def _sanitize(self):
        """Make lists of lists a bit more palletable..."""
        self.proba = np.vstack(self.proba)
        self.label = np.hstack(self.label)
