"""Figures du pipeline. Chaque fonction enregistre un PNG et renvoie son chemin."""
import matplotlib

matplotlib.use("Agg")      # pas d'affichage interactif : on écrit des fichiers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def courbes_perte(historique, chemin):
    """Pertes train / validation par époque, avec un repère sur la meilleure époque (celle du modèle gardé)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    epoques = np.arange(1, len(historique["train"]) + 1)
    ax.plot(epoques, historique["train"], label="train")
    ax.plot(epoques, historique["val"], label="validation")
    best = int(np.argmin(historique["val"]))
    ax.axvline(best + 1, color="grey", ls="--", lw=1,
               label=f"meilleure époque ({best + 1})")
    ax.set_xlabel("époque")
    ax.set_ylabel("perte (BCEWithLogits)")
    ax.set_title("Courbes de perte")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chemin, dpi=130)
    plt.close(fig)
    return chemin


def courbe_pr(jeux, chemin):
    """Courbes précision-rappel. jeux = {nom: (y, scores, tau)}.

    La ligne horizontale « hasard » est le taux de positifs du jeu : c'est la précision d'un modèle aléatoire,
    donc la vraie référence à battre pour une classe rare (contrairement au 0,5 d'une courbe ROC).
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for (nom, (y, scores, tau)), couleur in zip(jeux.items(), ["tab:blue", "tab:orange"]):
        p, r, seuils = precision_recall_curve(y, scores)
        ax.plot(r, p, color=couleur,
                label=f"{nom} — PR-AUC = {average_precision_score(y, scores):.3f}")
        ax.axhline(np.mean(y), color=couleur, ls=":", lw=1,
                   label=f"hasard {nom} = taux de positifs ({np.mean(y):.3f})")
        # Point de fonctionnement effectivement utilisé (le seuil figé tau)
        i = int(np.searchsorted(seuils, tau))
        i = min(i, len(seuils) - 1)
        ax.plot(r[i], p[i], "o", color=couleur, ms=7)
        ax.annotate(f"tau={tau:.3f}", (r[i], p[i]), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("rappel (classe 1 = anomalie)")
    ax.set_ylabel("précision")
    ax.set_title("Courbe précision-rappel")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(chemin, dpi=130)
    plt.close(fig)
    return chemin


def matrice_confusion(metriques_par_jeu, chemin):
    """Matrices de confusion à tau, côte à côte (validation et test)."""
    fig, axes = plt.subplots(1, len(metriques_par_jeu), figsize=(4.2 * len(metriques_par_jeu), 3.8))
    axes = np.atleast_1d(axes)
    for ax, (nom, m) in zip(axes, metriques_par_jeu.items()):
        mat = np.array([[m["VN"], m["FP"]], [m["FN"], m["VP"]]])
        ax.imshow(mat, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]}", ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black")
        ax.set_xticks([0, 1], ["prédit 0", "prédit 1"])
        ax.set_yticks([0, 1], ["vrai 0", "vrai 1"])
        ax.set_title(f"{nom} (tau = {m['tau']:.3f})")
    fig.tight_layout()
    fig.savefig(chemin, dpi=130)
    plt.close(fig)
    return chemin


def choisir_4_moteurs(par_moteur_test, rul_final):
    """Choisit 4 moteurs de test à tracer, de façon DÉTERMINISTE et sans regarder les scores :
    les 3 moteurs au RUL final le plus faible (donc avec le plus de cycles en zone d'anomalie : le score
    devrait monter) et le moteur au RUL final le plus élevé (loin de la panne : le score devrait rester bas)."""
    ordre = np.argsort(rul_final, kind="stable")      # indices 0-based ; moteur = indice + 1
    return [int(ordre[0]) + 1, int(ordre[1]) + 1, int(ordre[2]) + 1, int(ordre[-1]) + 1]


def score_vs_rul(meta_test, scores_test, tau, K, moteurs, chemin):
    """Score du modèle en fonction du RUL, pour 4 moteurs de test.

    Lecture attendue : en allant vers la droite (RUL qui diminue, axe inversé), le score doit monter et
    franchir tau avant d'entrer dans la zone RUL <= K (zone grisée).
    """
    d = meta_test.copy()
    d["score"] = np.asarray(scores_test)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    for ax, unit in zip(axes.ravel(), moteurs):
        g = d[d["unit_nr"] == unit].sort_values("RUL", ascending=False)
        ax.plot(g["RUL"], g["score"], lw=1.2)
        ax.axhline(tau, color="red", ls="--", lw=1, label=f"tau = {tau:.3f}")
        ax.axvspan(-0.5, K, color="orange", alpha=0.18, label=f"zone RUL <= K ({K})")
        ax.invert_xaxis()                     # le temps avance vers la droite
        ax.set_title(f"moteur {unit} (RUL final = {int(g['RUL'].min())})", fontsize=9)
        ax.set_xlabel("RUL (cycles restants)")
        ax.set_ylabel("score d'anomalie")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(chemin, dpi=130)
    plt.close(fig)
    return chemin
