"""Chargement, étiquetage, découpage et fenêtrage des données C-MAPSS FD001.

Toutes les fonctions renvoient des objets numpy / pandas simples pour pouvoir être réutilisées
telles quelles par un autre modèle (EncDec-AD plus tard).
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Les 10 capteurs conservés par le notebook de préparation Kaggle (fd001-data-preparation.ipynb).
CAPTEURS = ["sensor2", "sensor3", "sensor4", "sensor7", "sensor8",
            "sensor9", "sensor15", "sensor17", "sensor20", "sensor21"]

# Colonnes des fichiers bruts NASA : 26 colonnes séparées par des espaces, sans en-tête.
COLONNES_BRUTES = (["unit_nr", "cycle"]
                   + [f"setting{i}" for i in range(1, 4)]
                   + [f"sensor{i}" for i in range(1, 22)])


def verifier_cycles_consecutifs(df):
    """Vérifie que chaque moteur a des cycles 1, 2, 3, ... sans trou (sinon une fenêtre de L lignes
    ne correspondrait pas à L vols consécutifs)."""
    for unit, g in df.groupby("unit_nr"):
        c = g["cycle"].to_numpy()
        assert c[0] == 1 and np.all(np.diff(c) == 1), f"moteur {unit} : cycles non consécutifs"


def charger_brut(path):
    """Lit un fichier brut NASA (train_FD001.txt ou test_FD001.txt)."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLONNES_BRUTES)
    verifier_cycles_consecutifs(df)
    return df


def charger_train_prepare(path):
    """Lit train_FD001_prepared.csv et vérifie les hypothèses dont dépend la suite."""
    df = pd.read_csv(path)
    assert list(df.columns) == ["unit_nr", "cycle"] + CAPTEURS + ["RUL"], "colonnes inattendues"
    assert df["unit_nr"].nunique() == 100, "FD001 doit contenir 100 moteurs de train"
    verifier_cycles_consecutifs(df)
    # RUL = dernier_cycle - cycle  <=>  cycle + RUL est constant pour un moteur, et RUL vaut 0 au dernier cycle.
    somme = df["cycle"] + df["RUL"]
    assert (somme.groupby(df["unit_nr"]).nunique() == 1).all(), "cycle + RUL non constant"
    assert (df.groupby("unit_nr")["RUL"].min() == 0).all(), "le train doit aller jusqu'à la panne (RUL = 0)"
    return df


def ajouter_etiquette(df, K):
    """Ajoute la colonne 'anomalie' = 1 si RUL <= K, sinon 0. La colonne RUL n'est pas modifiée.
    K est un CHOIX (préavis souhaité en nombre de vols), pas une propriété des données."""
    df = df.copy()
    df["anomalie"] = (df["RUL"] <= K).astype(int)
    return df


def decouper_par_moteur(df, frac_val=0.2, seed=42):
    """Découpe en train / validation PAR MOTEUR.

    Pourquoi par moteur : deux fenêtres voisines partagent L-1 cycles sur L. Si on mélangeait les fenêtres
    au hasard, la validation contiendrait des quasi-copies de fenêtres d'entraînement et serait trop optimiste.
    """
    moteurs = np.sort(df["unit_nr"].unique())
    rng = np.random.default_rng(seed)
    moteurs = rng.permutation(moteurs)
    n_val = int(round(frac_val * len(moteurs)))          # 20 moteurs sur 100
    moteurs_val = set(moteurs[:n_val].tolist())
    moteurs_train = set(moteurs[n_val:].tolist())
    assert moteurs_train.isdisjoint(moteurs_val), "un moteur est à la fois en train et en validation"
    train = df[df["unit_nr"].isin(moteurs_train)].copy()
    val = df[df["unit_nr"].isin(moteurs_val)].copy()
    return train, val


def creer_fenetres(df, L=10, pas=1):
    """Construit les fenêtres glissantes, moteur par moteur.

    Pour chaque moteur, la fenêtre se terminant au cycle d'indice t contient les lignes [t-L+1 .. t].
    t va de L-1 (première fenêtre complète) au dernier cycle, de 'pas' en 'pas'.
    - Aucune fenêtre n'enjambe deux moteurs (on travaille dans un groupby).
    - Pas de padding : les L-1 premiers cycles n'ont pas de prédiction (un 0 serait un « moteur moyen » inventé).
    - Aplatissement ligne par ligne : [cycle t-L+1 : 10 capteurs, cycle t-L+2 : 10 capteurs, ..., cycle t].
      L'ordre chronologique est donc conservé, mais le MLP ne « sait » pas que ce sont des pas de temps.
    - Étiquette de la fenêtre = étiquette de son DERNIER cycle (à l'instant t on ne voit que le passé).

    Renvoie X (n_fenetres, L*10), y (n_fenetres,) et meta (DataFrame : unit_nr, cycle, RUL du dernier cycle),
    meta servant aux métriques métier (délai d'alerte, erreurs par tranche de RUL).
    """
    X, y, meta = [], [], []
    for unit, g in df.groupby("unit_nr"):
        g = g.sort_values("cycle")
        valeurs = g[CAPTEURS].to_numpy()          # uniquement les 10 capteurs (jamais cycle, unit_nr, RUL)
        etiquettes = g["anomalie"].to_numpy()
        cycles = g["cycle"].to_numpy()
        rul = g["RUL"].to_numpy()
        n = len(g)
        n_avant = len(X)
        for t in range(L - 1, n, pas):
            X.append(valeurs[t - L + 1: t + 1].ravel())
            y.append(etiquettes[t])
            meta.append((unit, cycles[t], rul[t]))
        if pas == 1:
            assert len(X) - n_avant == n - L + 1, f"moteur {unit} : nombre de fenêtres incorrect"
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    meta = pd.DataFrame(meta, columns=["unit_nr", "cycle", "RUL"])
    return X, y, meta


def ajuster_et_verifier_scaler(train_brut, train_prepare):
    """Reproduit le StandardScaler du notebook Kaggle (ajusté sur les 10 capteurs des 100 moteurs du train brut)
    et vérifie qu'il redonne exactement les valeurs du CSV préparé. Sinon on s'arrête : appliquer au test un
    scaler différent de celui du train rendrait les entrées du test incohérentes avec celles vues à l'entraînement."""
    scaler = StandardScaler().fit(train_brut[CAPTEURS])
    if not (train_brut[["unit_nr", "cycle"]].to_numpy() == train_prepare[["unit_nr", "cycle"]].to_numpy()).all():
        raise SystemExit("ARRÊT : les lignes du train brut et du CSV préparé ne sont pas dans le même ordre.")
    recalcule = scaler.transform(train_brut[CAPTEURS])
    attendu = train_prepare[CAPTEURS].to_numpy()
    if not np.allclose(recalcule, attendu):
        ecart = np.abs(recalcule - attendu).max()
        raise SystemExit(f"ARRÊT : le scaler recalculé ne redonne pas le CSV préparé (écart max = {ecart:.3g}). "
                         "Causes possibles : autres capteurs, autres moteurs, ou écart-type non « population ».")
    return scaler


def preparer_test(test_brut, rul_final, scaler, K, L):
    """Prépare le jeu de test officiel comme le train : mêmes capteurs, même scaler, RUL et étiquette par cycle.

    rul_final[i] = RUL du DERNIER cycle observé du moteur i+1 (format de RUL_FD001.txt : une valeur par ligne).
    Le test est coupé avant la panne, donc RUL(t) = RUL_final + (cycle_max - cycle(t)).
    """
    moteurs = np.sort(test_brut["unit_nr"].unique())
    assert len(rul_final) == len(moteurs) and moteurs[0] == 1 and moteurs[-1] == len(moteurs), \
        "RUL_FD001.txt doit contenir une ligne par moteur de test (ligne i = moteur i)"
    n_cycles = test_brut.groupby("unit_nr")["cycle"].count()
    trop_courts = n_cycles[n_cycles < L]
    if len(trop_courts) > 0:
        raise SystemExit(f"ARRÊT : moteurs de test avec moins de L={L} cycles : {list(trop_courts.index)}. "
                         "Décider quoi en faire (les exclure ? réduire L ?) avant de continuer.")

    test = test_brut[["unit_nr", "cycle"]].copy()
    test[CAPTEURS] = scaler.transform(test_brut[CAPTEURS])   # transform seulement : JAMAIS de fit sur le test
    cycle_max = test.groupby("unit_nr")["cycle"].transform("max")
    rul_fin_par_ligne = test["unit_nr"].map(dict(zip(moteurs, rul_final)))
    test["RUL"] = (rul_fin_par_ligne + (cycle_max - test["cycle"])).astype(int)

    dernier = test[test["cycle"] == cycle_max]
    assert (dernier["RUL"].to_numpy() == np.asarray(rul_final)[dernier["unit_nr"].to_numpy() - 1]).all(), \
        "le RUL au dernier cycle doit valoir RUL_final"
    return ajouter_etiquette(test, K)


def resume_jeu(y, meta):
    """Tableau de contrôle d'un jeu de fenêtres."""
    return {
        "n_fenetres": int(len(y)),
        "n_positifs": int(y.sum()),
        "taux_positifs": float(y.mean()),
        "n_moteurs": int(meta["unit_nr"].nunique()),
        "n_moteurs_avec_positif": int(meta.loc[y == 1, "unit_nr"].nunique()),
    }
