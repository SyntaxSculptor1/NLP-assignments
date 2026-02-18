import pandas as pd
from sklearn.model_selection import  GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MaxAbsScaler

import numpy as np
from scipy import sparse

def perform_support_vector_machine(
    x_train: sparse.csr_matrix,
    x_val: sparse.csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    param_grid: dict,
    scoring: str = "accuracy",
    verbose: int = 1
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
        estimator=LinearSVC(),
        param_grid=param_grid,
        cv=predefined_split, 
        scoring=scoring,
        verbose=verbose,
        n_jobs=-1,
    )

    grid_search.fit(x_full, y_full)

    return grid_search

def evaluate_support_vector_machine(
    model: LinearSVC,
    x_test: sparse.csr_matrix,
    y_test: pd.Series
) -> None:
    y_pred = model.predict(x_test)

    print(f"Model - Support Vector Machine (C = {model.get_params()['C']})")
    print("Accuracy:", accuracy_score(y_true= y_test, y_pred= y_pred))
    print("F1 Score:", f1_score(y_true=y_test, y_pred=y_pred, average="macro"))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true= y_test, y_pred= y_pred))