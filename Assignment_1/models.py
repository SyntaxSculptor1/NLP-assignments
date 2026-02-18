from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from scipy import sparse
from rich.console import Console
from pathlib import Path

console = Console()

def scale_data_and_define_split(
    x_train: sparse.csr_matrix,
    x_val: sparse.csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    verbose: bool = True
) -> Tuple[sparse.csr_matrix, np.ndarray, PredefinedSplit]:
    if verbose:
        console.print(f"\n [bold white] Scaling data and defining split: [/bold white]")

    scaler = MaxAbsScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    x_full = sparse.vstack([x_train_scaled, x_val_scaled])
    y_full = np.concatenate([y_train, y_val])

    assert x_train.shape and x_val.shape

    train_indices = np.full(x_train.shape[0], -1)
    val_indices = np.full(x_val.shape[0], 0)

    split_indices = np.concatenate([train_indices, val_indices])

    predefined_split = PredefinedSplit(split_indices)

    if verbose:
        console.print("Finished data scaling.")

    assert isinstance(x_full, sparse.csr_matrix)
    
    return (
        x_full,
        y_full,
        predefined_split
    )

def perform_logistic_regression(
    x_full: sparse.csr_matrix,
    y_full: np.ndarray,
    predefined_split: PredefinedSplit,
    param_grid: dict,
    scoring: str = "accuracy",
    max_iter: int = 5000,
    seed: int = 67,
    verbose: bool = True
) -> LogisticRegression:
    
    if verbose:
        console.print("\n [bold white] Performing Logistic Regression: [/bold white]")
        console.print("Performing Grid Search for Logistic Regression...")

    grid_search = GridSearchCV(
        estimator=LogisticRegression(max_iter=max_iter, random_state=seed),
        param_grid=param_grid,
        cv=predefined_split,
        scoring=scoring,
        verbose=int(verbose),
        n_jobs=-1,
    )

    grid_search.fit(x_full, y_full)

    if verbose:
        console.print("Finished Grid Search for Logistic Regression.")

    return grid_search.best_estimator_

def perform_support_vector_machine(
    x_full: sparse.csr_matrix,
    y_full: np.ndarray,
    predefined_split: PredefinedSplit,
    param_grid: dict,
    scoring: str = "accuracy",
    verbose: bool = True,
) -> LinearSVC:
    
    if verbose:
        console.print("\n [bold white] Performing Support Vector Machine: [/bold white]")
        console.print("Performing Grid Search for Support Vector Machine...")

    grid_search = GridSearchCV(
        estimator=LinearSVC(),
        param_grid=param_grid,
        cv=predefined_split, 
        scoring=scoring,
        verbose=int(verbose),
        n_jobs=-1,
    )

    grid_search.fit(x_full, y_full)

    if verbose:
        console.print("Performed Grid Search for Support Vector Machine.")

    return grid_search.best_estimator_

