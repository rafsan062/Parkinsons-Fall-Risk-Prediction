"""Phase 3 evaluation helpers.

Grids, class weights, and two-stage routing are copied from
``Revised_Modelling/Revised_Model_Results.ipynb``. Selection is inner-CV
``f1_macro`` (``evaluation_protocol.md``). ``apply_dense_imputation`` is copied
from ``02_imputation.ipynb`` — do not ``%run`` that notebook.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None

TARGET = "falls_class"
ID_COL = "PATNO"

META_COLS = [
    "PATNO",
    "index_date",
    "outcome_date",
    "first_visit_date",
    "history_months",
    "n_falls_visits",
    "SEX",
    "RACE",
    "falls_raw",
    "falls_class",
]

CLINICAL_COLS = [
    "No_of_years",
    "Age",
    "Postural instability present at dx?",
    "Rigidity present at diagnosis?",
    "Dopaminergic therapy started for participant",
    "MoCA Total Score",
    "Freezing of gait (peak severity)",
    "lightheaded after standing",
    "fainted",
    "Total Depression Score",
    "able_weighted_score",
    "MDS-UPDRS Part I Score",
    "BMI",
    "Daytime_Sleepiness",
    "Urinary_Problems",
    "Gait",
    "Postural_Stability",
    "Hoehn_And_Yahr_Stage",
    "Postural_hypotension",
    "MDS-UPDRS Part III Score",
    "MDS-UPDRS PartIV score",
]

REVISION_FLAGS = [
    "dopamine_missing",
    "NP3GAIT_missing_due_to_101",
    "NP3PSTBL_missing_due_to_101",
    "NHY_missing_due_to_101",
]

DELTA_COLS = [
    "Delta MoCA",
    "Delta UPDRS I",
    "Delta UPDRS III",
    "Delta UPDRS IV",
    "Delta Depression",
    "Delta Gait",
    "Delta Postural Stability",
    "Delta H&Y",
    "Delta BMI",
    "Delta Neuro-QoL",
]

PRIMARY_NAMES = [
    "Logistic Regression",
    "LinearSVC",
    "Random Forest",
    "XGBoost",
    "LightGBM",
]
DENSE_NAMES = {"Logistic Regression", "LinearSVC", "Random Forest", "Dummy"}

DOP = "Dopaminergic therapy started for participant"
P4 = "MDS-UPDRS PartIV score"
FOG = "Freezing of gait (peak severity)"
ABLE = "able_weighted_score"
TINY = ["Postural_hypotension", "Postural instability present at dx?"]

SEEDS = list(range(5))  # headline: 0–4. Optional extension: 0–19 if the professor wants more.
SEEDS_OPTIONAL = list(range(20))


def _mode(series):
    m = series.mode(dropna=True)
    return m.iloc[0] if len(m) else np.nan


def apply_dense_imputation(train, test):
    """Train-only median/mode. Call after the seed split, never on the full 1040.

    Copied from ``02_imputation.ipynb``. ``train`` / ``test`` should already have
    clinical fills (Part IV = 0, dopamine_missing) as in the tree CSVs.
    """
    tr, te = train.copy(), test.copy()
    for c in TINY:
        if c not in tr.columns:
            continue
        fill = _mode(tr[c])
        tr[c] = tr[c].fillna(fill)
        te[c] = te[c].fillna(fill)
    if "dopamine_missing" not in tr.columns:
        tr["dopamine_missing"] = tr[DOP].isna().astype(int)
        te["dopamine_missing"] = te[DOP].isna().astype(int)
    dop_mode = _mode(tr[DOP])
    tr[DOP] = tr[DOP].fillna(dop_mode)
    te[DOP] = te[DOP].fillna(dop_mode)
    fog_mode = _mode(tr[FOG])
    tr[FOG] = tr[FOG].fillna(fog_mode)
    te[FOG] = te[FOG].fillna(fog_mode)
    able_med = tr[ABLE].median()
    tr[ABLE] = tr[ABLE].fillna(able_med)
    te[ABLE] = te[ABLE].fillna(able_med)
    for c in [c for c in tr.columns if c.startswith("Delta ")]:
        flag = c + "_missing"
        tr[flag] = tr[c].isna().astype(int)
        te[flag] = te[c].isna().astype(int)
        tr[c] = tr[c].fillna(0)
        te[c] = te[c].fillna(0)
    if P4 in tr.columns:
        tr[P4] = tr[P4].fillna(0)
        te[P4] = te[P4].fillna(0)
    return tr, te


def tree_feature_columns(use_deltas):
    cols = list(CLINICAL_COLS) + list(REVISION_FLAGS)
    if use_deltas:
        cols = cols + list(DELTA_COLS)
    return cols


def dense_feature_columns(df, use_deltas):
    cols = tree_feature_columns(use_deltas)
    extra = [
        c
        for c in df.columns
        if c.startswith("Delta ") and c.endswith("_missing") and c not in cols
    ]
    return cols + extra


def require_boosting():
    missing = []
    if XGBClassifier is None:
        missing.append("xgboost")
    if LGBMClassifier is None:
        missing.append("lightgbm")
    if missing:
        raise ImportError(
            "Primary families require " + ", ".join(missing) + ". Install before seed 0."
        )


def _cv(y, seed):
    counts = pd.Series(np.asarray(y)).value_counts()
    min_c = int(counts.min())
    n_splits = 5 if min_c >= 5 else 3 if min_c >= 3 else max(2, min_c)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _logreg(seed, class_weight):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=1000,
                    random_state=seed,
                ),
            ),
        ]
    )


def _linearsvc(seed, class_weight):
    return CalibratedClassifierCV(
        Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LinearSVC(
                        class_weight=class_weight,
                        random_state=seed,
                        dual=False,
                    ),
                ),
            ]
        ),
        method="sigmoid",
        cv=5,
    )


LOGREG_GRID = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__solver": ["lbfgs", "liblinear"]}
# sklearn 1.8+: liblinear refuses n_classes >= 3. Direct comparator only; binary stages keep the paper grid.
DIRECT_LOGREG_GRID = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100], "clf__solver": ["lbfgs"]}
SVC_GRID = {"estimator__clf__C": [0.01, 0.1, 1, 10, 100]}

S1_RF_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.3, 0.5],
}
S2_RF_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}
S1_XGB_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
S2_XGB_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
LGBM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, -1],
    "learning_rate": [0.05, 0.1, 0.2],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [10, 20, 30],
}


def candidates_stage1(seed):
    out = [
        ("Logistic Regression", _logreg(seed, "balanced"), LOGREG_GRID),
        ("LinearSVC", _linearsvc(seed, "balanced"), SVC_GRID),
        (
            "Random Forest",
            RandomForestClassifier(
                class_weight={0: 1, 1: 2},
                random_state=seed,
                n_jobs=1,
            ),
            S1_RF_GRID,
        ),
        (
            "XGBoost",
            XGBClassifier(
                scale_pos_weight=2,
                eval_metric="logloss",
                random_state=seed,
                verbosity=0,
                n_jobs=1,
            ),
            S1_XGB_GRID,
        ),
        (
            "LightGBM",
            LGBMClassifier(
                class_weight={0: 1, 1: 2},
                random_state=seed,
                verbosity=-1,
                n_jobs=1,
            ),
            LGBM_GRID,
        ),
    ]
    return out


def candidates_stage2(seed):
    return [
        ("Logistic Regression", _logreg(seed, "balanced"), LOGREG_GRID),
        ("LinearSVC", _linearsvc(seed, "balanced"), SVC_GRID),
        (
            "Random Forest",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
            S2_RF_GRID,
        ),
        (
            "XGBoost",
            XGBClassifier(
                eval_metric="logloss",
                random_state=seed,
                verbosity=0,
                n_jobs=1,
            ),
            S2_XGB_GRID,
        ),
        (
            "LightGBM",
            LGBMClassifier(
                class_weight="balanced",
                random_state=seed,
                verbosity=-1,
                n_jobs=1,
            ),
            LGBM_GRID,
        ),
    ]


def candidates_direct(seed):
    """Same families; Stage 1 grids; balanced where the estimator supports it."""
    return [
        ("Logistic Regression", _logreg(seed, "balanced"), DIRECT_LOGREG_GRID),
        ("LinearSVC", _linearsvc(seed, "balanced"), SVC_GRID),
        (
            "Random Forest",
            RandomForestClassifier(
                class_weight="balanced",
                random_state=seed,
                n_jobs=1,
            ),
            S1_RF_GRID,
        ),
        (
            "XGBoost",
            XGBClassifier(
                eval_metric="mlogloss",
                random_state=seed,
                verbosity=0,
                n_jobs=1,
            ),
            S1_XGB_GRID,
        ),
        (
            "LightGBM",
            LGBMClassifier(
                class_weight="balanced",
                random_state=seed,
                verbosity=-1,
                n_jobs=1,
            ),
            LGBM_GRID,
        ),
    ]


def _X_for(name, X_dense, X_tree):
    return X_tree if name not in DENSE_NAMES else X_dense


def tune(name, estimator, param_grid, X, y, seed):
    cv = _cv(y, seed)
    gs = GridSearchCV(
        estimator,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=0,
        refit=True,
        error_score="raise",
    )
    t0 = time.time()
    gs.fit(X, y)
    elapsed = time.time() - t0
    print(
        f"    {name}: CV macro F1={gs.best_score_:.4f}  "
        f"n_splits={cv.n_splits}  {elapsed:.0f}s  params={gs.best_params_}"
    )
    return {
        "name": name,
        "estimator": gs.best_estimator_,
        "cv_macro_f1": float(gs.best_score_),
        "best_params": gs.best_params_,
        "n_splits": int(cv.n_splits),
        "seconds": elapsed,
    }


def select_winner(tuned):
    """Winner = argmax inner-CV f1_macro. Dummy is never in ``tuned``."""
    return max(tuned, key=lambda r: r["cv_macro_f1"])


def two_stage_predict(s1, s2, X):
    pred_s1 = np.asarray(s1.predict(X))
    y_final = np.zeros(len(X), dtype=int)
    routed = pred_s1 == 1
    if routed.any():
        pred_s2 = np.asarray(s2.predict(X.iloc[np.where(routed)[0]]))
        y_final[routed] = np.where(pred_s2 == 0, 1, 2)
    return y_final


def two_stage_proba(s1, s2, X):
    """Constructed 3-class probabilities for AUC only (hard routing is ``two_stage_predict``)."""
    p_s1 = np.asarray(s1.predict_proba(X)[:, 1])
    p_s2 = np.zeros(len(X), dtype=float)
    routed = np.asarray(s1.predict(X)) == 1
    if routed.any():
        p_s2[routed] = np.asarray(s2.predict_proba(X.iloc[np.where(routed)[0]])[:, 1])
    p0 = 1.0 - p_s1
    p1 = p_s1 * (1.0 - p_s2)
    p2 = p_s1 * p_s2
    return np.column_stack([p0, p1, p2])


def metrics_binary(y_true, y_pred, y_score):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "recall_class0": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "recall_class1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, y_score))
    except ValueError:
        out["auc"] = float("nan")
    return out


def metrics_3class(y_true, y_pred, y_proba=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rec = recall_score(y_true, y_pred, labels=[0, 1, 2], average=None, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2])),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", labels=[0, 1, 2])),
        "recall_none": float(rec[0]),
        "recall_rare": float(rec[1]),
        "recall_recurrent": float(rec[2]),
    }
    if y_proba is not None:
        try:
            out["auc_ovr_macro"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=[0, 1, 2])
            )
        except ValueError:
            out["auc_ovr_macro"] = float("nan")
    else:
        out["auc_ovr_macro"] = float("nan")
    return out


def mean_t_ci(values, alpha=0.95):
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 2:
        return {"n": n, "mean": float(arr.mean()) if n else float("nan"), "sd": float("nan"),
                "median": float(np.median(arr)) if n else float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    if sd == 0:
        return {"n": n, "mean": mean, "sd": 0.0, "median": float(np.median(arr)),
                "ci_low": mean, "ci_high": mean}
    ci_low, ci_high = stats.t.interval(alpha, df=n - 1, loc=mean, scale=stats.sem(arr))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "median": float(np.median(arr)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def paired_wilcoxon(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a - b
    if np.allclose(diff, 0):
        return {"mean_diff": 0.0, "wilcoxon_stat": float("nan"), "p_value": 1.0, "n": int(len(diff))}
    stat, p = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
    return {
        "mean_diff": float(diff.mean()),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
        "n": int(len(diff)),
        **{f"diff_{k}": v for k, v in mean_t_ci(diff).items()},
    }


def split_and_matrices(df, seed, use_deltas):
    train, test = train_test_split(
        df,
        test_size=0.3,
        random_state=seed,
        stratify=df[TARGET],
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    tree_cols = tree_feature_columns(use_deltas)
    missing = [c for c in tree_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing predictor columns: {missing}")

    train_dense, test_dense = apply_dense_imputation(train, test)
    dense_cols = dense_feature_columns(train_dense, use_deltas)

    X_train_tree = train[tree_cols].copy()
    X_test_tree = test[tree_cols].copy()
    X_train_dense = train_dense[dense_cols].copy()
    X_test_dense = test_dense[dense_cols].copy()
    if X_train_dense.isna().any().any() or X_test_dense.isna().any().any():
        raise ValueError("Dense matrices still contain NaN after apply_dense_imputation.")

    y_train = train[TARGET].astype(int)
    y_test = test[TARGET].astype(int)
    ids_test = test[ID_COL]
    return {
        "train": train,
        "test": test,
        "X_train_tree": X_train_tree,
        "X_test_tree": X_test_tree,
        "X_train_dense": X_train_dense,
        "X_test_dense": X_test_dense,
        "y_train": y_train,
        "y_test": y_test,
        "ids_test": ids_test,
        "tree_cols": tree_cols,
        "dense_cols": dense_cols,
    }


def _tune_family_list(cands, X_dense, X_tree, y, seed, label):
    print(f"  -- {label} (n={len(y)}) --")
    tuned = []
    for name, est, grid in cands:
        X = _X_for(name, X_dense, X_tree)
        tuned.append(tune(name, clone(est), grid, X, y, seed))
    winner = select_winner(tuned)
    print(f"  WINNER {label}: {winner['name']}  CV macro F1={winner['cv_macro_f1']:.4f}")
    return tuned, winner


def run_one_seed(df, seed, use_deltas):
    require_boosting()
    pack = split_and_matrices(df, seed, use_deltas)
    y_train = pack["y_train"]
    y_test = pack["y_test"]
    y_train_s1 = (y_train != 0).astype(int)
    y_test_s1 = (y_test != 0).astype(int)
    fallers = y_train.isin([1, 2])
    y_train_s2 = (y_train.loc[fallers] == 2).astype(int)

    Xtr_d, Xte_d = pack["X_train_dense"], pack["X_test_dense"]
    Xtr_t, Xte_t = pack["X_train_tree"], pack["X_test_tree"]
    Xtr_d_s2, Xtr_t_s2 = Xtr_d.loc[fallers], Xtr_t.loc[fallers]

    print(f"\n===== seed {seed}  train={len(y_train)} test={len(y_test)}  "
          f"S2 fallers={int(fallers.sum())} =====")

    s1_tuned, s1_win = _tune_family_list(
        candidates_stage1(seed), Xtr_d, Xtr_t, y_train_s1, seed, "Stage 1"
    )
    s2_tuned, s2_win = _tune_family_list(
        candidates_stage2(seed), Xtr_d_s2, Xtr_t_s2, y_train_s2, seed, "Stage 2"
    )
    dir_tuned, dir_win = _tune_family_list(
        candidates_direct(seed), Xtr_d, Xtr_t, y_train, seed, "Direct 3-class"
    )

    Xte_s1 = _X_for(s1_win["name"], Xte_d, Xte_t)
    Xte_s2 = _X_for(s2_win["name"], Xte_d, Xte_t)
    Xte_dir = _X_for(dir_win["name"], Xte_d, Xte_t)
    # Two-stage: S1 and S2 may be different backends. Score each on its own matrix
    # with aligned rows. Routing uses S1's predictions on Xte_s1.
    pred_s1 = np.asarray(s1_win["estimator"].predict(Xte_s1))
    proba_s1 = np.asarray(s1_win["estimator"].predict_proba(Xte_s1)[:, 1])
    y_two = np.zeros(len(y_test), dtype=int)
    routed = pred_s1 == 1
    if routed.any():
        pred_s2 = np.asarray(
            s2_win["estimator"].predict(Xte_s2.iloc[np.where(routed)[0]])
        )
        y_two[routed] = np.where(pred_s2 == 0, 1, 2)
    p_s2 = np.zeros(len(y_test), dtype=float)
    if routed.any():
        p_s2[routed] = np.asarray(
            s2_win["estimator"].predict_proba(Xte_s2.iloc[np.where(routed)[0]])[:, 1]
        )
    proba_two = np.column_stack([1.0 - proba_s1, proba_s1 * (1.0 - p_s2), proba_s1 * p_s2])

    pred_dir = np.asarray(dir_win["estimator"].predict(Xte_dir))
    proba_dir = np.asarray(dir_win["estimator"].predict_proba(Xte_dir))

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(Xtr_d, y_train)
    pred_dummy = dummy.predict(Xte_d)
    # most_frequent has no meaningful ranking; still emit a 3-col proba for shape
    dummy_proba = dummy.predict_proba(Xte_d)

    s1_test = metrics_binary(y_test_s1, pred_s1, proba_s1)
    two_test = metrics_3class(y_test, y_two, proba_two)
    dir_test = metrics_3class(y_test, pred_dir, proba_dir)
    dummy_test = metrics_3class(y_test, pred_dummy, dummy_proba)

    def _cv_table(tuned):
        return {r["name"]: {"cv_macro_f1": r["cv_macro_f1"], "best_params": r["best_params"],
                            "seconds": r["seconds"], "n_splits": r["n_splits"]}
                for r in tuned}

    row = {
        "seed": seed,
        "use_deltas": use_deltas,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_s2_train": int(fallers.sum()),
        "s1_winner": s1_win["name"],
        "s1_cv_macro_f1": s1_win["cv_macro_f1"],
        "s2_winner": s2_win["name"],
        "s2_cv_macro_f1": s2_win["cv_macro_f1"],
        "direct_winner": dir_win["name"],
        "direct_cv_macro_f1": dir_win["cv_macro_f1"],
        "two_stage_test_f1_macro": two_test["f1_macro"],
        "direct_test_f1_macro": dir_test["f1_macro"],
        "dummy_test_f1_macro": dummy_test["f1_macro"],
        "two_stage": two_test,
        "direct": dir_test,
        "dummy": dummy_test,
        "stage1_test": s1_test,
        "cv": {
            "stage1": _cv_table(s1_tuned),
            "stage2": _cv_table(s2_tuned),
            "direct": _cv_table(dir_tuned),
        },
        "s1_best_params": s1_win["best_params"],
        "s2_best_params": s2_win["best_params"],
        "direct_best_params": dir_win["best_params"],
    }
    print(
        f"  TEST 3-class macro F1  two-stage={two_test['f1_macro']:.4f}  "
        f"direct={dir_test['f1_macro']:.4f}  dummy={dummy_test['f1_macro']:.4f}"
    )
    return row


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def run_evaluation(csv_path, out_dir, use_deltas, seeds=None, resume=True):
    require_boosting()
    seeds = list(SEEDS if seeds is None else seeds)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    if not df[ID_COL].is_unique:
        raise ValueError("PATNO is not unique.")
    if df[TARGET].isna().any():
        raise ValueError("falls_class has NaN.")

    summary_path = out_dir / "seeds.csv"
    rows = []
    for seed in seeds:
        seed_path = out_dir / f"seed_{seed}.json"
        if resume and seed_path.exists():
            print(f"skip seed {seed} (already written)")
            with open(seed_path) as f:
                rows.append(json.load(f))
            continue
        row = run_one_seed(df, seed, use_deltas)
        with open(seed_path, "w") as f:
            json.dump(_json_ready(row), f, indent=2)
        rows.append(row)
        _write_seeds_csv(rows, summary_path)
    _write_seeds_csv(rows, summary_path)
    summary = summarize_runs(rows)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(_json_ready(summary), f, indent=2)
    return rows, summary


def _write_seeds_csv(rows, path):
    flat = []
    for r in rows:
        flat.append(
            {
                "seed": r["seed"],
                "s1_winner": r["s1_winner"],
                "s2_winner": r["s2_winner"],
                "direct_winner": r["direct_winner"],
                "two_stage_f1_macro": r["two_stage"]["f1_macro"],
                "direct_f1_macro": r["direct"]["f1_macro"],
                "dummy_f1_macro": r["dummy"]["f1_macro"],
                "two_stage_accuracy": r["two_stage"]["accuracy"],
                "direct_accuracy": r["direct"]["accuracy"],
                "two_stage_f1_weighted": r["two_stage"]["f1_weighted"],
                "direct_f1_weighted": r["direct"]["f1_weighted"],
                "two_stage_auc": r["two_stage"]["auc_ovr_macro"],
                "direct_auc": r["direct"]["auc_ovr_macro"],
                "two_stage_recall_none": r["two_stage"]["recall_none"],
                "two_stage_recall_rare": r["two_stage"]["recall_rare"],
                "two_stage_recall_recurrent": r["two_stage"]["recall_recurrent"],
                "direct_recall_none": r["direct"]["recall_none"],
                "direct_recall_rare": r["direct"]["recall_rare"],
                "direct_recall_recurrent": r["direct"]["recall_recurrent"],
                "s1_cv_macro_f1": r["s1_cv_macro_f1"],
                "s2_cv_macro_f1": r["s2_cv_macro_f1"],
                "direct_cv_macro_f1": r["direct_cv_macro_f1"],
            }
        )
    pd.DataFrame(flat).to_csv(path, index=False)


def summarize_runs(rows):
    two = [r["two_stage"]["f1_macro"] for r in rows]
    direct = [r["direct"]["f1_macro"] for r in rows]
    dummy = [r["dummy"]["f1_macro"] for r in rows]
    return {
        "n_seeds": len(rows),
        "two_stage_f1_macro": mean_t_ci(two),
        "direct_f1_macro": mean_t_ci(direct),
        "dummy_f1_macro": mean_t_ci(dummy),
        "two_stage_minus_direct": paired_wilcoxon(two, direct),
        "two_stage_accuracy": mean_t_ci([r["two_stage"]["accuracy"] for r in rows]),
        "direct_accuracy": mean_t_ci([r["direct"]["accuracy"] for r in rows]),
        "two_stage_recall_recurrent": mean_t_ci(
            [r["two_stage"]["recall_recurrent"] for r in rows]
        ),
        "direct_recall_recurrent": mean_t_ci(
            [r["direct"]["recall_recurrent"] for r in rows]
        ),
        "s1_winners": pd.Series([r["s1_winner"] for r in rows]).value_counts().to_dict(),
        "s2_winners": pd.Series([r["s2_winner"] for r in rows]).value_counts().to_dict(),
        "direct_winners": pd.Series([r["direct_winner"] for r in rows]).value_counts().to_dict(),
    }


def compare_delta_promotion(no_delta_rows, with_delta_rows, min_gain=0.02, alpha=0.05):
    """Pre-specified rule: deltas lead iff mean test macro F1 is +0.02 and Wilcoxon p<0.05."""
    a = np.array([r["two_stage"]["f1_macro"] for r in with_delta_rows], dtype=float)
    b = np.array([r["two_stage"]["f1_macro"] for r in no_delta_rows], dtype=float)
    if len(a) != len(b):
        raise ValueError("Delta comparison requires the same number of seeds.")
    w = paired_wilcoxon(a, b)
    mean_gain = float(a.mean() - b.mean())
    promote = bool(mean_gain >= min_gain and w["p_value"] < alpha)
    return {
        "mean_with_deltas": float(a.mean()),
        "mean_no_deltas": float(b.mean()),
        "mean_gain": mean_gain,
        "min_gain": min_gain,
        "wilcoxon": w,
        "promote_deltas_to_paper_model": promote,
    }


def project_paths(start=None):
    here = Path(start or Path.cwd()).resolve()
    if here.name == "revision_work":
        root = here.parent
        rev = here
    elif (here / "revision_work").is_dir():
        root = here
        rev = here / "revision_work"
    else:
        rev = here
        root = here.parent
    return root, rev


warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)


def _cli():
    import argparse

    root, rev = project_paths()
    p = argparse.ArgumentParser(description="Phase 3 multi-seed evaluation")
    p.add_argument(
        "--which",
        choices=["no_deltas", "with_deltas"],
        required=True,
    )
    p.add_argument("--seeds", default="0-4", help="e.g. 0-4 (headline) or 0-19 (optional extension)")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()
    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",")]
    if args.which == "no_deltas":
        csv = rev / "data" / "modelling_landmark_trees.csv"
        out = rev / "results" / "no_deltas"
        use_deltas = False
    else:
        csv = rev / "data" / "modelling_landmark_with_deltas_trees.csv"
        out = rev / "results" / "with_deltas"
        use_deltas = True
    print("csv", csv)
    print("out", out)
    print("seeds", seeds)
    run_evaluation(csv, out, use_deltas=use_deltas, seeds=seeds, resume=not args.no_resume)


if __name__ == "__main__":
    _cli()
