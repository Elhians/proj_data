# Détection d'anomalies sur C-MAPSS FD001 — MLP sur fenêtres glissantes

Classification binaire d'une fenêtre de `L` cycles consécutifs : `1 = anomalie` si `RUL <= K`, sinon `0`.
C-MAPSS n'a pas d'étiquettes d'anomalie : l'étiquette est notre construction.

**K ≠ τ** : `K` porte sur le RUL et fabrique l'étiquette avant l'entraînement ; `τ` porte sur le score du
modèle et décide de l'alerte. `τ` est choisi sur la validation (F-β maximal) puis **figé** pour le test.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu   # version CPU, plus légère
.venv/bin/pip install -r requirements.txt
```

## Utilisation

```bash
.venv/bin/python tests/test_fenetres.py      # tests du fenêtrage
.venv/bin/python src/train.py                # K=22, L=10, pas=1, beta=1, seed=42
.venv/bin/python src/train.py --pos_weight   # avec pondération de la classe positive
.venv/bin/python src/train.py --K 15         # autre K (la tâche change : comparer le GAIN de PR-AUC)
```

Résultats dans `results/K{K}_L{L}_pw{0|1}/` : `metriques.json`, `moteurs_test.csv`,
`erreurs_par_tranche.csv`, `modele.pt`, `scaler.joblib`, `config.json`, `historique.json` et les figures.

Notebook commenté et exécuté : `notebooks/mlp_fd001.ipynb` (régénérable avec
`python notebooks/construire_notebook.py` puis `jupyter nbconvert --to notebook --execute --inplace`).

## Structure

| Fichier | Rôle |
|---|---|
| `src/data_processing.py` | chargement, assertions, étiquetage, découpage par moteur, fenêtres, scaler, jeu de test |
| `src/models.py` | MLP `L*10 → 64 → 32 → 1` et fixation des graines |
| `src/train.py` | entraînement avec early stopping et enchaînement complet (CLI) |
| `src/evaluate.py` | métriques, indépendantes du modèle (réutilisables pour EncDec-AD) |
| `src/visualization.py` | courbes de perte, PR, matrices de confusion, score vs RUL |
| `tests/test_fenetres.py` | tests du fenêtrage |
| `data/` | `train_FD001_prepared.csv` (Kaggle) et les fichiers bruts NASA |

`src/evaluate.py` ne reçoit que `y`, `scores` et `meta` : il ne suppose pas que le score est une probabilité,
donc il servira tel quel pour le score de Mahalanobis d'EncDec-AD.

## Garanties contre la fuite de données

- découpage **par moteur** (80/20), vérifié par assertion : aucune fenêtre d'un même moteur des deux côtés ;
- early stopping et choix de `τ` **sur la validation uniquement**, jamais sur le test ;
- scaler ajusté sur le train brut, puis `transform` seulement sur le test ;
- seeds fixées (`numpy`, `torch`, découpage, ordre des mini-lots) : deux exécutions donnent le même JSON.

**Fuite mineure assumée et déclarée** : le scaler et la sélection des 10 capteurs du dataset Kaggle ont été
calculés sur les 100 moteurs de train, donc aussi sur nos 20 moteurs de validation.
