"""Tests du fenêtrage, à lancer avec : python tests/test_fenetres.py

Pas de framework : des assertions lisibles, qu'on peut défendre à l'oral.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import data_processing as dp


def moteur_synthetique(unit, n_cycles):
    """Moteur fictif de n_cycles cycles. Les valeurs des capteurs sont volontairement reconnaissables :
    le capteur j au cycle c vaut c * 100 + j, ce qui permet de vérifier l'ordre d'aplatissement."""
    lignes = {"unit_nr": unit, "cycle": np.arange(1, n_cycles + 1)}
    for j, capteur in enumerate(dp.CAPTEURS):
        lignes[capteur] = np.arange(1, n_cycles + 1) * 100.0 + j
    lignes["RUL"] = np.arange(n_cycles - 1, -1, -1)      # RUL = dernier_cycle - cycle
    return pd.DataFrame(lignes)


def test_nombre_de_fenetres_et_positifs():
    """Cas demandé dans le cahier des charges : 200 cycles, L = 10, K = 22 -> 191 fenêtres dont 23 positives.
    191 = 200 - 10 + 1 (les 9 premiers cycles n'ont pas assez d'historique).
    23 = cycles de RUL 0 à 22 inclus, tous atteignables car 200 > 22 + 10."""
    df = dp.ajouter_etiquette(moteur_synthetique(1, 200), K=22)
    X, y, meta = dp.creer_fenetres(df, L=10, pas=1)
    assert X.shape == (191, 100), X.shape
    assert len(y) == 191 and int(y.sum()) == 23, int(y.sum())
    # L'étiquette vient du DERNIER cycle de la fenêtre.
    assert (meta["RUL"].to_numpy() <= 22).sum() == 23
    assert np.array_equal(y, (meta["RUL"].to_numpy() <= 22).astype(np.int64))
    print("OK : 191 fenêtres, 23 positives")


def test_ordre_chronologique_et_bornes():
    """La première fenêtre couvre les cycles 1..10 ; l'aplatissement est chronologique (cycle le plus ancien
    d'abord, suivi de ses 10 capteurs)."""
    df = dp.ajouter_etiquette(moteur_synthetique(1, 200), K=22)
    X, y, meta = dp.creer_fenetres(df, L=10, pas=1)
    assert meta["cycle"].iloc[0] == 10, "la première fenêtre doit se terminer au cycle L = 10"
    assert meta["cycle"].iloc[-1] == 200, "la dernière fenêtre doit se terminer au dernier cycle"
    attendu = np.array([c * 100.0 + j for c in range(1, 11) for j in range(10)], dtype=np.float32)
    assert np.array_equal(X[0], attendu), "ordre d'aplatissement inattendu"
    # Aucun zéro de padding : la valeur minimale vient du cycle 1, pas d'une valeur fabriquée.
    assert X.min() == 100.0
    print("OK : ordre chronologique, pas de padding")


def test_aucune_fenetre_a_cheval():
    """Deux moteurs collés : aucune fenêtre ne doit mélanger leurs cycles."""
    df = pd.concat([moteur_synthetique(1, 50), moteur_synthetique(2, 40)], ignore_index=True)
    df = dp.ajouter_etiquette(df, K=22)
    X, y, meta = dp.creer_fenetres(df, L=10, pas=1)
    assert len(y) == (50 - 9) + (40 - 9) == 72, len(y)
    # Le capteur 0 de chaque pas de temps (colonnes 0, 10, 20, ...) encode le cycle : il doit croître de 100 en 100.
    cycles_dans_fenetre = X[:, ::10] / 100.0
    assert np.allclose(np.diff(cycles_dans_fenetre, axis=1), 1.0), "une fenêtre enjambe deux moteurs"
    assert set(meta["unit_nr"]) == {1, 2}
    print("OK : aucune fenêtre à cheval sur deux moteurs")


def test_decoupage_par_moteur_disjoint():
    """Le découpage 80/20 sépare bien les moteurs (pas les fenêtres)."""
    df = pd.concat([moteur_synthetique(u, 60) for u in range(1, 101)], ignore_index=True)
    df = dp.ajouter_etiquette(df, K=22)
    tr, va = dp.decouper_par_moteur(df, frac_val=0.2, seed=42)
    assert tr["unit_nr"].nunique() == 80 and va["unit_nr"].nunique() == 20
    assert set(tr["unit_nr"]).isdisjoint(set(va["unit_nr"]))
    print("OK : 80 / 20 moteurs disjoints")


if __name__ == "__main__":
    test_nombre_de_fenetres_et_positifs()
    test_ordre_chronologique_et_bornes()
    test_aucune_fenetre_a_cheval()
    test_decoupage_par_moteur_disjoint()
    print("\nTous les tests passent.")
