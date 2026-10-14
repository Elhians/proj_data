"""Entraînement et évaluation du MLP de détection d'anomalie sur C-MAPSS FD001.

Usage :
    python src/train.py                 # K=22, L=10, pas=1, beta=1, seed=42, sans pondération
    python src/train.py --pos_weight    # avec pondération de la classe positive
    python src/train.py --K 15          # autre K (la tâche change : comparer via le GAIN de PR-AUC)

Enchaînement : données -> étiquetage (K) -> découpage par moteur -> fenêtres -> MLP
-> seuil tau choisi sur la VALIDATION -> métriques validation puis test (tau figé).
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

import data_processing as dp
import evaluate as ev
import models
import visualization as viz


def arguments():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Paramètres de la tâche : CHOIX ARBITRAIRES, à justifier par nous, jamais optimisés sur le test.
    p.add_argument("--L", type=int, default=10, help="longueur de fenêtre en cycles")
    p.add_argument("--pas", type=int, default=1, help="décalage entre deux fenêtres")
    p.add_argument("--K", type=int, default=22, help="anomalie si RUL <= K")
    p.add_argument("--beta", type=float, default=1.0, help="beta du F-bêta servant à choisir tau")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pos_weight", action="store_true",
                   help="pondère la classe positive par n_negatifs / n_positifs du train")
    # Réglages d'optimisation : valeurs standards, non optimisées (et surtout pas sur le test).
    p.add_argument("--max_epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=10, help="époques sans amélioration avant arrêt")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data_dir", type=Path, default=Path("data"))
    p.add_argument("--out_dir", type=Path, default=Path("results"))
    return p.parse_args()


def entrainer(X_train, y_train, X_val, y_val, args):
    """Entraîne le MLP avec early stopping sur la perte de VALIDATION.

    Early stopping sur la validation (jamais sur le test) : on garde les poids de l'époque où la perte de
    validation est la plus basse, ce qui évite de sur-apprendre les fenêtres d'entraînement.
    La perte de validation utilise le même critère (même pos_weight) que l'entraînement : on surveille ainsi
    exactement la quantité que l'on optimise.
    """
    models.fixer_seed(args.seed)
    modele = models.MLP(X_train.shape[1])

    pos_weight = None
    if args.pos_weight:
        n_pos = int(y_train.sum())
        pos_weight = torch.tensor([(len(y_train) - n_pos) / n_pos], dtype=torch.float32)
    critere = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=args.lr)

    Xt = torch.from_numpy(X_train)
    yt = torch.from_numpy(y_train.astype(np.float32))
    Xv = torch.from_numpy(X_val)
    yv = torch.from_numpy(y_val.astype(np.float32))

    generateur = torch.Generator().manual_seed(args.seed)   # ordre des mini-lots reproductible
    historique = {"train": [], "val": []}
    meilleure_perte, meilleurs_poids, depuis = float("inf"), None, 0

    for epoque in range(args.max_epochs):
        modele.train()
        ordre = torch.randperm(len(Xt), generator=generateur)
        perte_cumulee = 0.0
        for debut in range(0, len(ordre), args.batch_size):
            lot = ordre[debut:debut + args.batch_size]
            optimiseur.zero_grad()
            perte = critere(modele(Xt[lot]), yt[lot])
            perte.backward()
            optimiseur.step()
            perte_cumulee += perte.item() * len(lot)
        perte_train = perte_cumulee / len(ordre)

        modele.eval()
        with torch.no_grad():
            perte_val = critere(modele(Xv), yv).item()
        historique["train"].append(perte_train)
        historique["val"].append(perte_val)

        if perte_val < meilleure_perte:
            meilleure_perte = perte_val
            meilleurs_poids = {k: v.clone() for k, v in modele.state_dict().items()}
            depuis = 0
        else:
            depuis += 1
            if depuis >= args.patience:
                print(f"  early stopping à l'époque {epoque + 1} "
                      f"(meilleure perte val = {meilleure_perte:.4f})")
                break

    modele.load_state_dict(meilleurs_poids)      # on repart des meilleurs poids, pas des derniers
    return modele, historique


def scores(modele, X):
    """Score d'anomalie = sigmoïde(logit), c'est-à-dire la probabilité estimée d'être en anomalie."""
    modele.eval()
    with torch.no_grad():
        return torch.sigmoid(modele(torch.from_numpy(X))).numpy()


def main():
    args = arguments()
    models.fixer_seed(args.seed)
    dossier = args.out_dir / f"K{args.K}_L{args.L}_pw{int(args.pos_weight)}"
    dossier.mkdir(parents=True, exist_ok=True)

    # --- 1. Données d'entraînement / validation -------------------------------------------------
    prepare = dp.charger_train_prepare(args.data_dir / "train_FD001_prepared.csv")
    prepare = dp.ajouter_etiquette(prepare, args.K)
    df_train, df_val = dp.decouper_par_moteur(prepare, frac_val=0.2, seed=args.seed)

    # --- 2. Scaler reproduit et vérifié, puis jeu de test officiel -------------------------------
    brut_train = dp.charger_brut(args.data_dir / "train_FD001.txt")
    scaler = dp.ajuster_et_verifier_scaler(brut_train, prepare)
    print("Scaler reproduit et vérifié (np.allclose) contre train_FD001_prepared.csv.")
    joblib.dump(scaler, dossier / "scaler.joblib")

    brut_test = dp.charger_brut(args.data_dir / "test_FD001.txt")
    rul_final = np.loadtxt(args.data_dir / "RUL_FD001.txt").astype(int)
    df_test = dp.preparer_test(brut_test, rul_final, scaler, args.K, args.L)

    # --- 3. Fenêtres glissantes (mêmes règles pour les trois jeux) -------------------------------
    X_train, y_train, meta_train = dp.creer_fenetres(df_train, args.L, args.pas)
    X_val, y_val, meta_val = dp.creer_fenetres(df_val, args.L, args.pas)
    X_test, y_test, meta_test = dp.creer_fenetres(df_test, args.L, args.pas)
    assert set(meta_train["unit_nr"]).isdisjoint(set(meta_val["unit_nr"])), \
        "fuite : un moteur se retrouve en train et en validation"

    resumes = {"train": dp.resume_jeu(y_train, meta_train),
               "validation": dp.resume_jeu(y_val, meta_val),
               "test": dp.resume_jeu(y_test, meta_test)}
    print("\nTableau de contrôle (K = %d, L = %d, pas = %d)" % (args.K, args.L, args.pas))
    print(pd.DataFrame(resumes).T.to_string())

    # --- 4. Entraînement -------------------------------------------------------------------------
    print(f"\nEntraînement (pos_weight = {args.pos_weight})")
    modele, historique = entrainer(X_train, y_train, X_val, y_val, args)
    torch.save(modele.state_dict(), dossier / "modele.pt")
    (dossier / "historique.json").write_text(json.dumps(historique, indent=2))

    s_val = scores(modele, X_val)
    s_test = scores(modele, X_test)

    # --- 5. Seuil tau : choisi sur la VALIDATION, puis FIGÉ pour le test -------------------------
    tau, fbeta_val = ev.choisir_tau(y_val, s_val, args.beta)
    print(f"\ntau choisi sur la validation : {tau:.4f} (F{args.beta:g} = {fbeta_val:.4f}) — figé pour le test")

    # --- 6. Métriques : même code pour validation et test, à tau puis au seuil naïf 0,5 ----------
    metriques = {
        "validation_tau": ev.metriques_classification(y_val, s_val, tau, args.beta),
        "validation_0.5": ev.metriques_classification(y_val, s_val, 0.5, args.beta),
        "test_tau": ev.metriques_classification(y_test, s_test, tau, args.beta),
        "test_0.5": ev.metriques_classification(y_test, s_test, 0.5, args.beta),
    }
    print("\nMétriques (classe 1 = anomalie)")
    print(pd.DataFrame(metriques).to_string())

    par_moteur, resume_metier = ev.metriques_metier(meta_test, y_test, s_test, tau, args.K)
    par_moteur.to_csv(dossier / "moteurs_test.csv", index=False)
    print("\nMétriques métier (test, à tau)")
    for cle, valeur in resume_metier.items():
        print(f"  {cle} : {valeur}")

    tranches = ev.erreurs_par_tranche(meta_test, y_test, s_test, tau, args.K)
    tranches.to_csv(dossier / "erreurs_par_tranche.csv", index=False)
    print("\nErreurs par tranche de RUL (test, à tau)")
    print(tranches.to_string(index=False))

    # --- 7. Sauvegardes --------------------------------------------------------------------------
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (dossier / "config.json").write_text(json.dumps(config, indent=2))
    (dossier / "metriques.json").write_text(json.dumps(
        {"config": config, "resumes_jeux": resumes, "tau": tau,
         "f_beta_validation_au_choix_de_tau": fbeta_val,
         "metriques": metriques, "metier_test": resume_metier}, indent=2))

    viz.courbes_perte(historique, dossier / "pertes.png")
    viz.courbe_pr({"validation": (y_val, s_val, tau), "test": (y_test, s_test, tau)},
                  dossier / "courbe_pr.png")
    viz.matrice_confusion({"validation": metriques["validation_tau"],
                           "test": metriques["test_tau"]}, dossier / "matrices_confusion.png")
    moteurs = viz.choisir_4_moteurs(par_moteur, rul_final)
    viz.score_vs_rul(meta_test, s_test, tau, args.K, moteurs, dossier / "score_vs_rul.png")
    print(f"\nRésultats et figures écrits dans {dossier}")


if __name__ == "__main__":
    main()
