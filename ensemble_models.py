from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    seed = data.random_seed
    X_train = np.asarray(data.train_features, dtype=float)
    X_test = np.asarray(data.test_features, dtype=float)
    y_train = np.asarray(data.train_target, dtype=int).reshape(-1)
    y_test = np.asarray(data.test_target, dtype=int).reshape(-1)

    numeric = [1, 4, 12]
    categorical = [i for i in range(20) if i not in numeric]

    def linear_features():
        return ColumnTransformer([
            ("num", make_pipeline(FunctionTransformer(np.log1p), StandardScaler()), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])

    def tree_features():
        return ColumnTransformer([("num", "passthrough", numeric), ("cat", "passthrough", categorical)])

    model = VotingClassifier(
        [
            ("lr", make_pipeline(linear_features(), LogisticRegression(C=0.1, max_iter=3000))),
            ("rf", make_pipeline(tree_features(), RandomForestClassifier(
                n_estimators=300, min_samples_leaf=1, random_state=seed))),
            ("et", make_pipeline(tree_features(), ExtraTreesClassifier(
                n_estimators=300, min_samples_leaf=3, random_state=seed))),
            ("gb", make_pipeline(tree_features(), GradientBoostingClassifier(
                n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.7, random_state=seed))),
            ("mlp", make_pipeline(linear_features(), MLPClassifier(
                hidden_layer_sizes=(16,), alpha=3.0, max_iter=2000, random_state=seed))),
        ],
        voting="soft",
    )
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return EnsembleResult(
        model_name="Soft Voting: LogisticRegression + RandomForest + ExtraTrees + GradientBoosting + MLP",
        probabilities=probabilities,
        predictions=predictions,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        f1=float(f1_score(y_test, predictions, zero_division=0)),
    )
