"""Modèle MLP et fixation des graines aléatoires."""
import random

import numpy as np
import torch
from torch import nn


def fixer_seed(seed):
    """Fixe toutes les sources d'aléa (initialisation des poids, ordre des mini-lots, découpage)
    pour que deux exécutions avec la même seed donnent les mêmes résultats (sur CPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    """Perceptron multicouche : fenêtre aplatie (L*10 valeurs) -> 64 -> 32 -> 1 logit.

    - ReLU : non-linéarité standard, simple et efficace.
    - Pas de sigmoïde en sortie : on renvoie le LOGIT, car BCEWithLogitsLoss combine sigmoïde + entropie croisée
      de façon numériquement plus stable. Le score (probabilité) s'obtient ensuite par torch.sigmoid(logit).
    - Taille 64/32 : choix courant pour ~100 entrées et ~15 000 exemples ; non optimisée.
    """

    def __init__(self, n_entrees):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(n_entrees, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.reseau(x).squeeze(-1)   # (batch, 1) -> (batch,)
