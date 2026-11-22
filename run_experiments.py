import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from ensemble_models import run_ensemble_experiment
from grader_contracts.ensembles import EnsembleInput

warnings.filterwarnings("ignore")
SEED = 42
NUMERIC = [1, 4, 12]
CATEGORICAL = [i for i in range(20) if i not in NUMERIC]


def linear_features():
    return ColumnTransformer([
        ("num", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def tree_features():
    return ColumnTransformer([("num", "passthrough", NUMERIC), ("cat", "passthrough", CATEGORICAL)])


def make_member(kind, params):
    if kind == "lr":
        return make_pipeline(linear_features(), LogisticRegression(max_iter=3000, **params))
    if kind == "mlp":
        return make_pipeline(linear_features(), MLPClassifier(max_iter=2000, random_state=SEED, **params))
    if kind == "rf":
        return make_pipeline(tree_features(), RandomForestClassifier(n_estimators=300, random_state=SEED, **params))
    if kind == "et":
        return make_pipeline(tree_features(), ExtraTreesClassifier(n_estimators=300, random_state=SEED, **params))
    return make_pipeline(tree_features(), GradientBoostingClassifier(subsample=0.7, random_state=SEED, **params))




if __name__ == "__main__":
    ROOT = Path(__file__).parent

    german = pd.read_csv(ROOT / "german.csv", sep=";")
    X = german.drop(columns="Creditability").to_numpy(dtype=float)
    y = german["Creditability"].to_numpy(dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

    GRIDS = {
        "lr": [{"C": 0.03}, {"C": 0.1}, {"C": 1.0}],
        "rf": [{"min_samples_leaf": 1}, {"min_samples_leaf": 2}, {"min_samples_leaf": 5}],
        "et": [{"min_samples_leaf": 1}, {"min_samples_leaf": 3}, {"min_samples_leaf": 5}],
        "gb": [
            {"n_estimators": 200, "max_depth": 2, "learning_rate": 0.05},
            {"n_estimators": 250, "max_depth": 3, "learning_rate": 0.03},
        ],
        "mlp": [
            {"hidden_layer_sizes": (16,), "alpha": 1.0},
            {"hidden_layer_sizes": (16,), "alpha": 3.0},
            {"hidden_layer_sizes": (32,), "alpha": 3.0},
        ],
    }
    NAMES = {
        "lr": "LogisticRegression (one-hot)",
        "rf": "RandomForest",
        "et": "ExtraTrees",
        "gb": "GradientBoosting",
        "mlp": "MLP",
    }


    def evaluate(name, model):
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {
            "name": name,
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "proba": proba,
        }


    rows = [
        evaluate("LogisticRegression, стандартизация без one-hot (база из ЛР3)",
                 make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))),
        evaluate("HistGradientBoosting (learning_rate=0.05, max_depth=3)",
                 HistGradientBoostingClassifier(learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                                l2_regularization=1.0, max_iter=150, random_state=SEED)),
    ]

    winners = {}
    for kind, grid in GRIDS.items():
        family = []
        for params in grid:
            description = ", ".join(f"{k}={v}" for k, v in params.items())
            family.append(evaluate(f"{NAMES[kind]}: {description}", make_member(kind, params)))
            family[-1]["params"] = params
        rows.extend(family)
        winners[kind] = max(family, key=lambda r: r["cv_mean"])["params"]

    ENSEMBLES = {
        "Soft Voting: LR + RF + GB": ["lr", "rf", "gb"],
        "Soft Voting: LR + RF + ET + GB": ["lr", "rf", "et", "gb"],
        "Soft Voting: LR + RF + ET + GB + MLP": ["lr", "rf", "et", "gb", "mlp"],
    }
    for name, kinds in ENSEMBLES.items():
        members = [(k, make_member(k, winners[k])) for k in kinds]
        rows.append(evaluate(name, VotingClassifier(members, voting="soft")))

    best = max(rows, key=lambda r: r["cv_mean"])
    assert best["name"] == "Soft Voting: LR + RF + ET + GB + MLP"

    final = run_ensemble_experiment(EnsembleInput(X_train, y_train, X_test, y_test, SEED))
    assert np.allclose(final.probabilities, best["proba"])

    results = {
        "model_name": final.model_name,
        "roc_auc": final.roc_auc,
        "accuracy": final.accuracy,
        "precision": final.precision,
        "recall": final.recall,
        "f1": final.f1,
    }
    (ROOT / "experiment_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Модель: {final.model_name}")
    print(f"ROC-AUC:   {final.roc_auc:.4f}")
    print(f"Accuracy:  {final.accuracy:.4f}")
    print(f"Precision: {final.precision:.4f}")
    print(f"Recall:    {final.recall:.4f}")
    print(f"F1:        {final.f1:.4f}")
