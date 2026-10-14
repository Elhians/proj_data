"""Génère notebooks/mlp_fd001.ipynb (puis on l'exécute avec nbconvert).

Passer par un script évite d'écrire du JSON de notebook à la main et garde les cellules lisibles ici.
Ce fichier n'est PAS nécessaire pour utiliser le notebook : il sert seulement à le (re)construire.
"""
import json
from pathlib import Path

cellules = []


def md(texte):
    cellules.append({"cell_type": "markdown", "metadata": {}, "source": texte.strip().splitlines(keepends=True)})


def code(texte):
    cellules.append({"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": texte.strip().splitlines(keepends=True)})


# ---------------------------------------------------------------- 0. Introduction
md("""
# Détection d'anomalies sur C-MAPSS FD001 — MLP sur fenêtres glissantes

**Tâche.** Classification binaire d'une *fenêtre* de L cycles consécutifs : `1 = anomalie`, `0 = normal`.
C-MAPSS ne contient **pas** d'étiquettes d'anomalie (le jeu a été conçu pour prédire le RUL) :
l'étiquette est **notre construction**, `anomalie = (RUL <= K)`.

**Deux seuils à ne jamais confondre.**

| | porte sur | rôle | choisi comment |
|---|---|---|---|
| **K** | le RUL (la réalité) | fabrique l'étiquette **avant** l'entraînement | choix métier (préavis de maintenance), arbitraire ici |
| **τ (tau)** | le score du modèle | transforme un score en décision (alerte si score ≥ τ) | maximise le F-β **sur la validation**, puis **figé** pour le test |

K définit l'épreuve ; τ est notre façon de la passer. Changer K change la tâche, donc deux F1 obtenus
avec des K différents ne sont pas comparables.

**Pourquoi des fenêtres et pas un cycle isolé ?** Le bruit cycle-à-cycle est de l'ordre d'un écart-type :
un seul relevé est peu informatif. Une anomalie est ici **collective** — c'est une *trajectoire* qui dérive,
pas un point aberrant. La fenêtre donne au modèle un bout de trajectoire.
""")

code("""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RACINE = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(RACINE / "src"))

import data_processing as dp
import evaluate as ev
import models
import visualization as viz

# Paramètres de la tâche. K et beta sont des CHOIX ARBITRAIRES, à justifier par nous,
# et à ne JAMAIS optimiser sur le jeu de test.
L, PAS, K, BETA, SEED = 10, 1, 22, 1.0, 42
DATA = RACINE / "data"
SORTIE = RACINE / "results" / "notebook"
SORTIE.mkdir(parents=True, exist_ok=True)

models.fixer_seed(SEED)
print(f"L={L}  pas={PAS}  K={K}  beta={BETA}  seed={SEED}")
""")

# ---------------------------------------------------------------- 1. Chargement
md("""
## 1. Chargement et vérifications

On charge le CSV déjà préparé (Kaggle) et on **vérifie par assertions** les hypothèses dont tout le reste dépend :
100 moteurs, cycles consécutifs sans trou (sinon une fenêtre de 10 lignes ne serait pas 10 vols consécutifs),
et `cycle + RUL` constant par moteur (c'est la définition `RUL = dernier_cycle − cycle`).
Si une assertion casse, le pipeline s'arrête : mieux vaut un arrêt qu'un résultat faux.
""")

code("""
prepare = dp.charger_train_prepare(DATA / "train_FD001_prepared.csv")
print(prepare.shape, "lignes x colonnes")
print("moteurs :", prepare["unit_nr"].nunique())
print("durée de vie : min %d, max %d, moyenne %.1f cycles" % (
    prepare.groupby("unit_nr")["cycle"].max().min(),
    prepare.groupby("unit_nr")["cycle"].max().max(),
    prepare.groupby("unit_nr")["cycle"].max().mean()))
prepare.head()
""")

# ---------------------------------------------------------------- 2. Étiquetage + découpage
md("""
## 2. Étiquetage et découpage train / validation

`anomalie = (RUL <= K)` dans une **nouvelle colonne** : on ne touche pas à `RUL`, dont on aura besoin
pour les métriques métier (délai d'alerte, erreurs par tranche de RUL).

**Le découpage se fait par MOTEUR, pas par fenêtre.** Avec un pas de 1, deux fenêtres voisines partagent
9 cycles sur 10 : si on mélangeait les fenêtres au hasard, la validation contiendrait des quasi-copies des
fenêtres d'entraînement et donnerait un score artificiellement bon. 80 moteurs / 20 moteurs, seed fixée.
""")

code("""
prepare = dp.ajouter_etiquette(prepare, K)
df_train, df_val = dp.decouper_par_moteur(prepare, frac_val=0.2, seed=SEED)

moteurs_tr, moteurs_va = set(df_train["unit_nr"]), set(df_val["unit_nr"])
assert moteurs_tr.isdisjoint(moteurs_va)
print(f"train : {len(moteurs_tr)} moteurs | validation : {len(moteurs_va)} moteurs | aucun moteur en commun")
print("taux de cycles anormaux (train) : %.4f" % df_train["anomalie"].mean())
""")

# ---------------------------------------------------------------- 3. Fenêtres
md("""
## 3. Fenêtres glissantes

Pour chaque moteur séparément, la fenêtre qui se termine au cycle *t* contient les cycles *[t−L+1 .. t]*,
avec *t* allant de *L−1* au dernier cycle, par pas de 1.

Trois décisions importantes :

1. **Étiquette = celle du dernier cycle.** C'est la décision réelle : à l'instant *t*, on ne voit que le passé.
   (Alternative écartée : « anormale si elle contient un cycle anormal », qui ferait alerter trop tôt.)
2. **Pas de padding** pour les L−1 premiers cycles : après standardisation, 0 = « moteur moyen »,
   donc remplir de zéros fabriquerait des données. On ne prédit simplement pas sur ces cycles.
   Un moteur de *n* cycles donne donc *n − L + 1* fenêtres. C'est une **limite du MLP** à citer :
   un LSTM n'aurait pas cette contrainte.
3. **Aucune fenêtre n'enjambe deux moteurs** (on travaille dans un `groupby`).

L'aplatissement met bout à bout les 10 capteurs du cycle le plus ancien, puis ceux du suivant, etc.
L'ordre chronologique est conservé, mais **le MLP ne sait pas que ce sont des pas de temps** : il a un poids
distinct pour chaque position, sans partage de poids (contrairement à une convolution). Autre limite à citer.
""")

code("""
X_train, y_train, meta_train = dp.creer_fenetres(df_train, L, PAS)
X_val,   y_val,   meta_val   = dp.creer_fenetres(df_val,   L, PAS)
print("X_train", X_train.shape, " (L x 10 capteurs =", L * 10, "entrées)")
print("X_val  ", X_val.shape)
""")

md("""
### Test unitaire du fenêtrage

Un moteur fictif de 200 cycles, avec L = 10 et K = 22, doit donner **191 fenêtres dont 23 positives**
(191 = 200 − 10 + 1 ; 23 = les cycles de RUL 0 à 22 inclus). Les tests complets sont dans
`tests/test_fenetres.py` (ordre chronologique, absence de padding, aucune fenêtre à cheval sur deux moteurs).
""")

code("""
import subprocess
print(subprocess.run([sys.executable, str(RACINE / "tests" / "test_fenetres.py")],
                     capture_output=True, text=True).stdout)
""")

# ---------------------------------------------------------------- 4. Scaler + test
md("""
## 4. Jeu de test officiel

Le CSV Kaggle est déjà standardisé, mais **le scaler n'a pas été sauvegardé**. Pour traiter le test avec
exactement la même transformation, on le **reproduit** (StandardScaler ajusté sur les 10 capteurs des
100 moteurs de `train_FD001.txt` brut) et on **vérifie par `np.allclose`** qu'il redonne le CSV préparé.
Si la vérification échoue, on s'arrête : appliquer au test un scaler différent de celui du train rendrait
les entrées incohérentes avec ce que le modèle a appris.

> **Limite assumée** : ce scaler a été ajusté sur les 100 moteurs, donc aussi sur nos 20 moteurs de validation.
> C'est une **fuite mineure** (seulement deux statistiques globales, moyenne et écart-type), héritée du dataset Kaggle.
> On la déclare plutôt que de la masquer.

Le test est **tronqué avant la panne** : `RUL_FD001.txt` donne le RUL du dernier cycle observé de chaque moteur,
d'où `RUL(t) = RUL_final + (cycle_max − cycle(t))`.
""")

code("""
brut_train = dp.charger_brut(DATA / "train_FD001.txt")
scaler = dp.ajuster_et_verifier_scaler(brut_train, prepare)
print("Scaler reproduit et VÉRIFIÉ (np.allclose) contre train_FD001_prepared.csv.")

import joblib
joblib.dump(scaler, SORTIE / "scaler.joblib")

brut_test = dp.charger_brut(DATA / "test_FD001.txt")
rul_final = np.loadtxt(DATA / "RUL_FD001.txt").astype(int)
df_test = dp.preparer_test(brut_test, rul_final, scaler, K, L)
X_test, y_test, meta_test = dp.creer_fenetres(df_test, L, PAS)

print("moteurs de test :", df_test["unit_nr"].nunique(),
      "| cycles par moteur : min %d, max %d" % (
          df_test.groupby("unit_nr")["cycle"].count().min(),
          df_test.groupby("unit_nr")["cycle"].count().max()))
print("RUL final : min %d, max %d" % (rul_final.min(), rul_final.max()))
""")

# ---------------------------------------------------------------- 5. Tableau de contrôle
md("""
## 5. Tableau de contrôle

À lire **avant** tout résultat de modèle : ces chiffres disent à quel point la tâche est déséquilibrée et
à quoi se comparer. Le taux de positifs du **test** est beaucoup plus faible que celui du train parce que
le test est tronqué : la plupart des moteurs s'arrêtent loin de la panne et n'ont **aucun** cycle anormal.
Conséquence : les métriques du test ne sont pas comparables telles quelles à celles de la validation.
""")

code("""
resumes = {"train": dp.resume_jeu(y_train, meta_train),
           "validation": dp.resume_jeu(y_val, meta_val),
           "test": dp.resume_jeu(y_test, meta_test)}
pd.DataFrame(resumes).T
""")

# ---------------------------------------------------------------- 6. Modèle
md("""
## 6. Modèle et entraînement

MLP `100 → 64 → 32 → 1`, ReLU, sortie = **logit** (pas de sigmoïde dans le modèle : `BCEWithLogitsLoss`
combine sigmoïde et entropie croisée de façon numériquement plus stable). Optimiseur Adam.
**Early stopping sur la perte de VALIDATION** : on garde les poids de la meilleure époque — jamais le test.

On entraîne **deux variantes** pour comparer :

- **sans `pos_weight`** : chaque fenêtre pèse pareil ;
- **avec `pos_weight` = n_négatifs / n_positifs** : une fenêtre anormale pèse ≈ 8 fois plus, ce qui pousse
  le modèle vers le rappel. Attention : cela change aussi l'échelle des scores, donc τ s'ajustera différemment.
""")

code("""
import types

import train as tr   # train.py ne lit les arguments que dans main(), on peut donc importer ses fonctions

def config(pos_weight):
    return types.SimpleNamespace(seed=SEED, pos_weight=pos_weight, max_epochs=200,
                                 patience=10, batch_size=256, lr=1e-3)

resultats = {}
for pw in (False, True):
    print(f"--- pos_weight = {pw}")
    modele, historique = tr.entrainer(X_train, y_train, X_val, y_val, config(pw))
    resultats[pw] = {"modele": modele, "historique": historique,
                     "s_val": tr.scores(modele, X_val), "s_test": tr.scores(modele, X_test)}
""")

code("""
from IPython.display import Image, display
for pw in (False, True):
    chemin = viz.courbes_perte(resultats[pw]["historique"], SORTIE / f"pertes_pw{int(pw)}.png")
    print(f"pos_weight = {pw}")
    display(Image(filename=str(chemin)))
""")

# ---------------------------------------------------------------- 7. tau
md("""
## 7. Choix du seuil τ

τ est choisi **uniquement sur la validation**, comme celui qui maximise le F-β (ici β = 1, donc F1),
via `precision_recall_curve`. Il est ensuite **figé** : on l'applique au test sans le retoucher.
Retoucher τ sur le test reviendrait à regarder la réponse avant de passer l'examen — c'est exactement
la critique faite à certains articles du domaine (seuil optimal choisi sur le test).

On rapporte aussi le **seuil naïf 0,5** pour voir ce que le choix de τ apporte (ou coûte).
""")

code("""
for pw in (False, True):
    tau, fbeta = ev.choisir_tau(y_val, resultats[pw]["s_val"], BETA)
    resultats[pw]["tau"] = tau
    print(f"pos_weight = {pw} : tau = {tau:.4f}  (F{BETA:g} validation = {fbeta:.4f})")
""")

# ---------------------------------------------------------------- 8. Métriques
md("""
## 8. Métriques

Trois niveaux, tous calculés par le **même code** (`src/evaluate.py`) sur la validation puis sur le test,
pour qu'un futur modèle (EncDec-AD) soit évalué exactement de la même façon.

- **Niveau 1 — qualité du score, sans seuil** : PR-AUC, à comparer au **taux de positifs** qui est la PR-AUC
  d'un modèle aléatoire. ROC-AUC en secondaire : il paraît très bon quand la classe positive est rare,
  car les vrais négatifs écrasent tout.
- **Niveau 2 — qualité de la décision, à τ figé** : matrice de confusion, précision, rappel, F1 / F-β,
  et l'accuracy comparée à celle du modèle trivial « toujours normal » (= 1 − taux de positifs).
  Sur le test, ce trivial dépasse 98 % : **l'accuracy seule est inacceptable ici**.
- **Niveau 3 — métier** : délai d'alerte, fausses alertes par moteur, moteurs détectés.
""")

code("""
lignes = {}
for pw in (False, True):
    tau = resultats[pw]["tau"]
    for jeu, y, s in (("validation", y_val, resultats[pw]["s_val"]),
                      ("test", y_test, resultats[pw]["s_test"])):
        lignes[f"{jeu} | tau (pw={pw})"] = ev.metriques_classification(y, s, tau, BETA)
        lignes[f"{jeu} | 0.5 (pw={pw})"] = ev.metriques_classification(y, s, 0.5, BETA)

tableau = pd.DataFrame(lignes)
tableau.round(4)
""")

md("""
### Comparaison des deux variantes et des deux seuils

Points à regarder : le **gain de PR-AUC** par rapport au hasard (ligne `PR_AUC` contre `PR_AUC_hasard`),
l'écart entre l'accuracy du modèle et celle du trivial « toujours normal », et l'effet du seuil naïf 0,5.
""")

code("""
tableau.loc[["taux_positifs", "accuracy_toujours_normal", "accuracy",
             "precision", "rappel", "F1", "PR_AUC", "PR_AUC_hasard", "ROC_AUC"]].round(4).T
""")

code("""
metier = {}
for pw in (False, True):
    par_moteur, resume = ev.metriques_metier(meta_test, y_test, resultats[pw]["s_test"],
                                             resultats[pw]["tau"], K)
    resultats[pw]["par_moteur"] = par_moteur
    metier[f"pos_weight={pw}"] = resume
pd.DataFrame(metier)
""")

md("""
**Lecture des métriques métier.** `delai_alerte_median` est le RUL au moment de la première alerte :
c'est le préavis réellement offert à la maintenance, en nombre de vols. `n_moteurs_anomalie_detectes`
compte les moteurs réellement en zone d'anomalie pour lesquels on a alerté dans cette zone — c'est la
question qui compte vraiment, bien plus que le nombre de fenêtres bien classées.
""")

code("""
for pw in (False, True):
    print(f"--- erreurs par tranche de RUL, pos_weight = {pw} (tau = {resultats[pw]['tau']:.3f})")
    print(ev.erreurs_par_tranche(meta_test, y_test, resultats[pw]["s_test"],
                                 resultats[pw]["tau"], K).to_string(index=False))
    print()
""")

md("""
Les tranches sont alignées sur K. On s'attend à ce que les erreurs se concentrent **juste autour de RUL ≈ K** :
une fenêtre à RUL = 23 et une à RUL = 21 sont physiquement presque identiques mais portent des étiquettes
opposées. Ces erreurs-là sont **inévitables** et ne traduisent pas une faiblesse du modèle, mais le caractère
arbitraire de la frontière K.
""")

# ---------------------------------------------------------------- 9. Graphiques
md("""
## 9. Graphiques

On garde la variante **sans `pos_weight`** pour les figures (voir le tableau ci-dessus pour la comparaison).
""")

code("""
MEILLEUR = False     # variante illustrée ; la comparaison chiffrée reste dans les tableaux ci-dessus
tau = resultats[MEILLEUR]["tau"]
s_val, s_test = resultats[MEILLEUR]["s_val"], resultats[MEILLEUR]["s_test"]

display(Image(filename=str(viz.courbe_pr(
    {"validation": (y_val, s_val, tau), "test": (y_test, s_test, tau)}, SORTIE / "courbe_pr.png"))))
""")

code("""
m = {"validation": ev.metriques_classification(y_val, s_val, tau, BETA),
     "test": ev.metriques_classification(y_test, s_test, tau, BETA)}
display(Image(filename=str(viz.matrice_confusion(m, SORTIE / "matrices_confusion.png"))))
""")

md("""
### Score en fonction du RUL

Les 4 moteurs sont choisis **sans regarder les scores** : les 3 au RUL final le plus faible (ils ont le plus
de cycles en zone d'anomalie) et celui au RUL final le plus élevé (il est loin de la panne, son score doit
rester bas). C'est la figure la plus parlante à l'oral : on doit voir le score **monter** à l'approche de la panne.
""")

code("""
moteurs = viz.choisir_4_moteurs(resultats[MEILLEUR]["par_moteur"], rul_final)
print("moteurs tracés :", moteurs)
display(Image(filename=str(viz.score_vs_rul(meta_test, s_test, tau, K, moteurs,
                                            SORTIE / "score_vs_rul.png"))))
""")

code("""
(SORTIE / "metriques_notebook.json").write_text(json.dumps({
    "parametres": {"L": L, "pas": PAS, "K": K, "beta": BETA, "seed": SEED},
    "resumes_jeux": resumes,
    "tau": {str(pw): resultats[pw]["tau"] for pw in (False, True)},
    "metriques": {k: v for k, v in lignes.items()},
    "metier_test": metier,
}, indent=2))
print("métriques écrites dans", SORTIE / "metriques_notebook.json")
""")

# ---------------------------------------------------------------- 10. Limites
md("""
## 10. Limites

**Sur les données**
- **Fuite mineure déclarée** : la standardisation et la sélection des 10 capteurs ont été calculées sur les
  100 moteurs de train, donc aussi sur nos 20 moteurs de validation. Faible (deux statistiques globales),
  mais réelle — nous l'héritons du dataset Kaggle et nous le disons plutôt que de le cacher.
- **K est arbitraire.** Il définit la tâche. Il devrait venir d'un besoin métier (préavis de maintenance en
  nombre de vols), pas d'une optimisation. Une analyse de sensibilité (K = 15, 22, 30, 50) doit se comparer
  sur le **gain de PR-AUC par rapport au taux de positifs**, jamais sur des F1 bruts : changer K change l'épreuve.
- **Le test est tronqué** : peu de fenêtres positives et une large majorité de moteurs sans aucun positif.
  Les chiffres du test portent donc sur peu d'exemples positifs et sont fragiles (un bootstrap par moteur
  donnerait un intervalle de confiance — non fait ici).
- **Retirer les capteurs corrélés** (choix du notebook Kaggle) peut jeter du signal et supprime la structure
  inter-capteurs qu'un autoencodeur exploiterait. Le seuil d'écart-type 0,01 portait sur des valeurs **brutes**,
  donc dépendantes des unités.
- Pipeline **FD001 seulement** : une seule condition opératoire, un seul mode de panne. Sur FD002
  (6 conditions), il faudrait normaliser par condition, sinon un régime sain jamais vu serait signalé à tort
  (« nouveau ≠ anormal »).

**Sur la méthode**
- **Chevauchement des fenêtres** : 90 % entre voisines. D'où le découpage par moteur ; mais les fenêtres d'un
  même jeu restent fortement corrélées, donc le nombre de fenêtres surestime la quantité d'information réelle.
- **Limites du MLP à fenêtre aplatie** : entrée de taille fixe, aucune notion d'ordre (un poids par position,
  pas de poids partagés comme une convolution), pas de mémoire au-delà de L cycles, et surtout **besoin
  d'étiquettes**. Or en usine on n'a pas de RUL : c'est tout l'intérêt d'apprendre le « normal » sans
  étiquettes (EncDec-AD, prochaine étape).
- **L = 10 n'est pas justifié** expérimentalement à ce stade (une variation de L reste à faire).
- L'hypothèse « le début de vie est normal » est une **hypothèse**, pas un fait vérifié.
""")

notebook = {
    "cells": cellules,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

chemin = Path(__file__).resolve().parent / "mlp_fd001.ipynb"
chemin.write_text(json.dumps(notebook, indent=1, ensure_ascii=False))
print("écrit :", chemin)
