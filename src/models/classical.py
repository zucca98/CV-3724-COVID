"""Modelli baseline classici: SVM e RandomForest con grid-search.

Scopo del file
--------------
Questo modulo implementa il "braccio classico" della pipeline di
classificazione COVID/non-COVID: due classificatori scikit-learn (SVM con
kernel RBF e RandomForest) allenati sulle feature handcrafted prodotte da
`src/features.py` (LBP + GLCM + HOG).

Perche' questi due modelli?
- **SVM (RBF)**: classico potente sui dataset di medie dimensioni con feature
  ad alta dimensionalita'. Il kernel RBF cattura relazioni non lineari.
- **RandomForest**: ensemble di alberi, naturalmente robusto a feature
  ridondanti e scale diverse, fornisce *feature importance* utile per
  l'interpretabilita'.

Entrambi i modelli vengono allenati con `GridSearchCV` su una griglia ridotta
di iperparametri (per restare in budget tempo/risorse), con stratified 5-fold
CV e selezione basata su ROC-AUC (metrica robusta allo sbilanciamento).

Pipeline SVM
------------
`StandardScaler -> PCA(200) -> SVC(rbf, probability=True)`

Il PCA e' un compromesso pragmatico: il vettore feature originale e' ~26.000
dimensioni (dominato dall'HOG); calcolare il kernel RBF su feature cosi'
estese sarebbe troppo lento. 200 componenti principali catturano > 95% della
varianza e rendono il calcolo gestibile su CPU.

Integrazione
------------
- Chiamato da `scripts/train_classical.py`.
- I modelli allenati vengono persistiti via `joblib` e caricati da
  `scripts/run_full_evaluation.py`.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.utils import get_logger

logger = get_logger(__name__)

# Costante a livello di modulo: lo stesso schema CV viene riutilizzato per
# SVM e RF, cosi' i due grid search sono confrontabili. `random_state=42`
# garantisce gli stessi split fra esecuzioni successive (riproducibilita').
_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Allena un'SVM RBF con GridSearchCV sul set combinato train+val.

    Strategia: concateniamo train+val in un unico set e lasciamo che la CV
    interna di `GridSearchCV` produca i suoi split. Il test set NON entra
    mai qui: viene tenuto da parte per la valutazione finale (no leakage).

    Pipeline sklearn applicata in ordine:
        1. `StandardScaler`  - normalizza le feature (media 0, std 1).
                              Necessario perche' SVC e' sensibile alle scale.
        2. `PCA(n=200)`      - riduce dimensionalita': 26K -> 200, mantenendo
                              ~95% della varianza. Risparmio computazionale
                              ENORME.
        3. `SVC(kernel='rbf')` - classificatore vero e proprio.
                              `probability=True` abilita `predict_proba`
                              (richiesto da `compute_metrics`).
                              `cache_size=500` aumenta il kernel cache a
                              500 MB, accelerando l'allenamento.

    Griglia di iperparametri (4 * 3 * 5 fold = 60 fit):
        - `C` in {0.1, 1, 10, 100}  - regolarizzazione (basso=margine ampio,
                                      alto=meno regolarizzazione).
        - `gamma` in {'scale', 1e-3, 1e-2} - ampiezza del kernel RBF.

    Args:
        X_train: Matrice feature training `(n_train, n_features)`.
        y_train: Label training `(n_train,)`.
        X_val: Matrice feature validation `(n_val, n_features)`.
        y_val: Label validation `(n_val,)`.

    Returns:
        Dizionario con:
            - `best_estimator`: pipeline allenata (scaler + pca + svm).
            - `best_params`: dict degli iperparametri vincenti.
            - `best_score`: ROC-AUC medio in CV della configurazione migliore.
    """
    # Concatenazione train + val: il refit finale di GridSearchCV avverra'
    # su entrambi gli split.
    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            # PCA: 200 componenti = compromesso fra perdita info e velocita'.
            # `random_state=42` per riproducibilita' (l'algoritmo PCA di
            # sklearn ha un componente randomico nel solver).
            ("pca", PCA(n_components=200, random_state=42)),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42, cache_size=500)),
        ]
    )

    # Griglia: parametri prefissati con doppio underscore "<step>__<param>"
    # come da convenzione sklearn Pipeline.
    param_grid = {
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.001, 0.01],
    }

    gs = GridSearchCV(
        pipeline,
        param_grid,
        cv=_CV,
        scoring="roc_auc",  # robusta allo sbilanciamento, premia il ranking corretto
        n_jobs=-1,          # tutti i core CPU disponibili
        verbose=1,
        refit=True,         # alla fine ri-allena su tutto il dataset con la migliore config
    )

    logger.info("SVM grid search: %d fits (4 C x 3 gamma x 5 folds)", 4 * 3 * 5)
    gs.fit(X, y)
    logger.info("SVM best params: %s  CV AUC=%.4f", gs.best_params_, gs.best_score_)

    return {
        "best_estimator": gs.best_estimator_,
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
    }


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    """Allena un RandomForest con GridSearchCV sul set combinato train+val.

    A differenza dell'SVM, **non serve** ne' lo scaling ne' la PCA: i
    RandomForest sono invarianti alle trasformazioni monotone delle feature
    e gestiscono bene feature ridondanti (gli alberi le ignorano per via
    della divisione di information gain).

    Griglia di iperparametri (3 * 3 * 5 fold = 45 fit):
        - `n_estimators` in {100, 300, 500}  - numero di alberi nella foresta.
                                              Piu' alti = piu' robusto ma
                                              piu' lento.
        - `max_depth` in {None, 10, 20}      - profondita' massima.
                                              `None` = nessun limite (rischio
                                              overfitting sui training small).

    Args:
        X_train: Matrice feature training `(n_train, n_features)`.
        y_train: Label training `(n_train,)`.
        X_val: Matrice feature validation `(n_val, n_features)`.
        y_val: Label validation `(n_val,)`.

    Returns:
        Dizionario con:
            - `best_estimator`: RandomForest allenato.
            - `best_params`: iperparametri vincenti.
            - `best_score`: ROC-AUC CV medio della configurazione migliore.
    """
    X = np.vstack([X_train, X_val])
    y = np.concatenate([y_train, y_val])

    # `n_jobs=-1` paralleliza l'allenamento dei singoli alberi (RF e'
    # facilmente parallelizzabile: ogni albero e' indipendente).
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
    }

    gs = GridSearchCV(
        rf,
        param_grid,
        cv=_CV,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    logger.info("RF grid search: %d fits (3 n_estimators x 3 max_depth x 5 folds)", 3 * 3 * 5)
    gs.fit(X, y)
    logger.info("RF best params: %s  CV AUC=%.4f", gs.best_params_, gs.best_score_)

    return {
        "best_estimator": gs.best_estimator_,
        "best_params": gs.best_params_,
        "best_score": float(gs.best_score_),
    }
