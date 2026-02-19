from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.preprocessing import MaxAbsScaler
from scipy import sparse
from rich.console import Console

console = Console()

def scale_data_and_define_split(
    x_train: sparse.csr_matrix,
    x_val: sparse.csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    verbose: bool = True
) -> Tuple[sparse.csr_matrix, np.ndarray, PredefinedSplit]:
    """
    Scale the data and define the split.

    Args:
        x_train (sparse.csr_matrix): The training features.
        x_val (sparse.csr_matrix): The validation features.
        y_train (pd.Series): The training labels.
        y_val (pd.Series): The validation labels.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        Tuple[sparse.csr_matrix, np.ndarray, PredefinedSplit]: The scaled data, labels, and split.
    """
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
    """
    Perform Logistic Regression.

    Args:
        x_full (sparse.csr_matrix): The full features.
        y_full (np.ndarray): The full labels.
        predefined_split (PredefinedSplit): The predefined split.
        param_grid (dict): The parameter grid.
        scoring (str, optional): The scoring metric. Defaults to "accuracy".
        max_iter (int, optional): The maximum number of iterations. Defaults to 5000.
        seed (int, optional): The random seed. Defaults to 67.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        LogisticRegression: The best estimator.
    """
    
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
    """
    Perform Support Vector Machine.

    Args:
        x_full (sparse.csr_matrix): The full features.
        y_full (np.ndarray): The full labels.
        predefined_split (PredefinedSplit): The predefined split.
        param_grid (dict): The parameter grid.
        scoring (str, optional): The scoring metric. Defaults to "accuracy".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        LinearSVC: The best estimator.
    """
    
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