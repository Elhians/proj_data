"""Métriques d'évaluation, indépendantes du modèle.

Toutes les fonctions prennent seulement :
  - y      : étiquettes vraies des fenêtres (0/1)
  - scores : score d'anomalie de chaque fenêtre, croissant avec l'anomalie (pour le MLP : la probabilité)
  - meta   : DataFrame (unit_nr, cycle, RUL) du dernier cycle de chaque fenêtre
Ce module est donc réutilisable tel quel pour EncDec-AD (dont le score sera une distance de Mahalanobis,
non bornée à 1 : aucune fonction ici ne suppose que le score est une probabilité).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)


def f_beta(precision, rappel, beta):
    """F-bêta = moyenne harmonique pondérée. beta > 1 privilégie le RAPPEL (rater une panne coûte cher),
    beta < 1 privilégie la PRÉCISION (les fausses alertes coûtent cher). beta = 1 : F1, équilibré."""
    num = (1 + beta ** 2) * precision * rappel
    den = (beta ** 2) * precision + rappel
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def choisir_tau(y_val, scores_val, beta):
    """Choisit le seuil de décision tau qui maximise le F-bêta SUR LA VALIDATION.

    tau n'a rien à voir avec K : K étiquette la réalité (RUL <= K), tau décide à partir du score (alerte si score >= tau).
    precision_recall_curve renvoie len(thresholds) + 1 points (le dernier point, rappel = 0, n'a pas de seuil associé),
    d'où le [:-1] pour aligner précision/rappel sur les seuils.
    Cette fonction ne doit JAMAIS être appelée sur le test : tau est figé après la validation.
    """
    precision, rappel, seuils = precision_recall_curve(y_val, scores_val)
    scores_fbeta = f_beta(precision[:-1], rappel[:-1], beta)
    i = int(np.argmax(scores_fbeta))
    return float(seuils[i]), float(scores_fbeta[i])


def metriques_classification(y, scores, tau, beta):
    """Niveau 1 (qualité du score, sans seuil) + niveau 2 (qualité de la décision à tau figé)."""
    y = np.asarray(y)
    pred = (np.asarray(scores) >= tau).astype(int)
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    taux_positifs = float(y.mean())
    precision = vp / (vp + fp) if (vp + fp) > 0 else 0.0
    rappel = vp / (vp + fn) if (vp + fn) > 0 else 0.0
    return {
        "tau": float(tau),
        "taux_positifs": taux_positifs,
        # Référence à battre : un modèle qui dit « toujours normal » obtient cette accuracy sans rien détecter.
        "accuracy_toujours_normal": 1.0 - taux_positifs,
        "VP": int(vp), "FP": int(fp), "FN": int(fn), "VN": int(vn),
        "accuracy": float((vp + vn) / len(y)),
        "precision": float(precision),
        "rappel": float(rappel),
        "F1": float(f_beta(np.array(precision), np.array(rappel), 1.0)),
        f"F{beta:g}": float(f_beta(np.array(precision), np.array(rappel), beta)),
        # PR-AUC : à comparer au taux de positifs, qui est la PR-AUC d'un modèle aléatoire.
        "PR_AUC": float(average_precision_score(y, scores)),
        "PR_AUC_hasard": taux_positifs,
        # ROC-AUC : métrique secondaire, optimiste quand la classe positive est rare.
        "ROC_AUC": float(roc_auc_score(y, scores)),
    }


def metriques_metier(meta, y, scores, tau, K):
    """Niveau 3 : une ligne par moteur, à partir du RUL de chaque cycle. Pensé pour le test.

    - delai_alerte : RUL au PREMIER cycle où score >= tau (plus il est grand, plus on prévient tôt).
                     NaN si le moteur n'a jamais déclenché d'alerte.
    - n_fausses_alertes : alertes émises alors que RUL > K (hors zone d'anomalie).
    - detecte : au moins une alerte DANS la zone RUL <= K (c'est la détection qui compte pour la maintenance).
    """
    d = meta.copy()
    d["y"] = np.asarray(y)
    d["alerte"] = (np.asarray(scores) >= tau).astype(int)

    lignes = []
    for unit, g in d.groupby("unit_nr"):
        g = g.sort_values("cycle")
        alertes = g[g["alerte"] == 1]
        lignes.append({
            "unit_nr": unit,
            "a_anomalie": int(g["y"].any()),
            "n_cycles_anomalie": int(g["y"].sum()),
            "delai_alerte": float(alertes["RUL"].iloc[0]) if len(alertes) else np.nan,
            "n_fausses_alertes": int(((g["alerte"] == 1) & (g["RUL"] > K)).sum()),
            "detecte": int(((g["alerte"] == 1) & (g["RUL"] <= K)).any()),
        })
    par_moteur = pd.DataFrame(lignes)

    a_anomalie = par_moteur["a_anomalie"] == 1
    resume = {
        "n_moteurs": int(len(par_moteur)),
        "n_moteurs_avec_anomalie": int(a_anomalie.sum()),
        "n_moteurs_anomalie_detectes": int(par_moteur.loc[a_anomalie, "detecte"].sum()),
        # Médiane calculée uniquement sur les moteurs à anomalie QUI ONT ALERTÉ : un moteur jamais alerté
        # n'a pas de délai. On publie donc aussi l'effectif, sinon la médiane paraîtrait meilleure qu'elle n'est.
        "n_moteurs_anomalie_ayant_alerte": int(par_moteur.loc[a_anomalie, "delai_alerte"].notna().sum()),
        "delai_alerte_median_si_alerte": float(
            par_moteur.loc[a_anomalie, "delai_alerte"].median()),
        "total_fausses_alertes": int(par_moteur["n_fausses_alertes"].sum()),
        "n_moteurs_avec_fausse_alerte": int((par_moteur["n_fausses_alertes"] > 0).sum()),
    }
    return par_moteur, resume


def erreurs_par_tranche(meta, y, scores, tau, K, largeur=5, portee=30):
    """Répartition des FP et FN par tranche de RUL, pour voir si les erreurs se concentrent autour de RUL ~ K.

    Les bornes sont alignées sur K (..., ]K-5, K], ]K, K+5], ...) : une fenêtre à RUL = K+1 et une à RUL = K-1
    sont presque identiques mais portent des étiquettes opposées, donc des erreurs y sont ATTENDUES.
    """
    d = meta.copy()
    d["y"] = np.asarray(y)
    d["pred"] = (np.asarray(scores) >= tau).astype(int)

    bornes = [-1] + list(range(K - portee, K + portee + 1, largeur)) + [int(d["RUL"].max()) + 1]
    bornes = sorted(set(b for b in bornes if b == -1 or b >= 0))
    d["tranche"] = pd.cut(d["RUL"], bins=bornes)

    d["FP"] = ((d["pred"] == 1) & (d["y"] == 0)).astype(int)
    d["FN"] = ((d["pred"] == 0) & (d["y"] == 1)).astype(int)
    g = d.groupby("tranche", observed=True)
    table = pd.DataFrame({
        "n_fenetres": g.size(),
        "n_positifs": g["y"].sum(),
        "FP": g["FP"].sum(),
        "FN": g["FN"].sum(),
    }).reset_index()
    table["tranche"] = table["tranche"].astype(str)
    return table
