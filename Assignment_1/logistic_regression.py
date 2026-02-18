import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from sklearn.preprocessing import MaxAbsScaler

import numpy as np
from scipy import sparse


def perform_logistic_regression(
    x_train: sparse.csr_matrix,
    x_val: sparse.csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    param_grid: dict,
    scoring: str = "accuracy",
    max_iter: int = 5000,
    verbose: int = 1,
    seed: int = 67,
) -> GridSearchCV:

    scaler = MaxAbsScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    x_full = sparse.vstack([x_train_scaled, x_val_scaled])
    y_full = np.concatenate([y_train, y_val])

    train_indices = np.full(x_train.shape[0], -1)
    val_indices = np.full(x_val.shape[0], 0)

    split_indices = np.concatenate([train_indices, val_indices])

    predefined_split = PredefinedSplit(split_indices)

    grid_search = GridSearchCV(
        estimator=LogisticRegression(max_iter=max_iter, random_state=seed),
        param_grid=param_grid,
        cv=predefined_split,
        scoring=scoring,
        verbose=verbose,
        n_jobs=-1,
    )

    grid_search.fit(x_full, y_full)

    return grid_search


def evaluate_logistic_regression(
    model: LogisticRegression, x_test: sparse.csr_matrix, y_test: pd.Series
) -> None:
    y_pred = model.predict(x_test)

    print(f"Model - Logistic Regression (C = {model.get_params()['C']})")
    print("Accuracy:", accuracy_score(y_true=y_test, y_pred=y_pred))
    print("F1 Score:", f1_score(y_true=y_test, y_pred=y_pred, average="macro"))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true=y_test, y_pred=y_pred))
