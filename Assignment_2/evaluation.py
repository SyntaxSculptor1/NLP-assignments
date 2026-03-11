from pathlib import Path

import pandas as pd
from rich.console import Console
from scipy import sparse
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
import numpy as np
from keras import Sequential

from utils import CATEGORIES

console = Console()


def evaluate_model(
    model: Sequential,
    model_name: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True,
    plot: bool = True,
) -> None:
    """
    Evaluate the performance of a given model on the test set.

    Args:
        model (Sequential): The model to evaluate.
        model_name (str): The name of the model.
        x_test (np.ndarray): The test features.
        y_test (np.ndarray): The test labels.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        plot (bool, optional): Whether to plot the confusion matrix. Defaults to True.

    Returns:
        None
    """
    if verbose:
        console.print(f"\n [bold white] Evaluating model: {model_name} [/bold white]")

    y_pred = model.predict(x_test)

    y_pred, y_test = np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average="macro")
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

    console.print("\nAccuracy:", accuracy)
    console.print("F1 Score:", f1)

    console.print("Confusion Matrix:")
    console.print("Rows: Actual Class, Columns: Predicted Class")
    console.print(cm)

    if plot:
        save_path = (
            Path(__file__).parent / "plots" / f"{model_name}_confusion_matrix.png"
        )

        display = ConfusionMatrixDisplay(cm, display_labels=CATEGORIES)
        display.plot()
        display.ax_.set_title(f"Confusion Matrix of {model_name}")
        display.figure_.savefig(str(save_path))

    if verbose:
        console.print(f"\n [white] Finished evaluating {model_name}. [/white]")


def find_misclassified(
    model: Sequential,
    model_name: str,
    x_test_raw: pd.DataFrame,
    x_test_cleaned: np.ndarray,
    y_test: np.ndarray,
    text_column: str = "text",
    label_column: str = "label",
    verbose: bool = True,
    save: bool = True,
) -> pd.DataFrame:
    """
    Find misclassified examples in the test set.

    Args:
        model (Sequential): The model to evaluate.
        model_name (str): The name of the model.
        x_test_raw (pd.DataFrame): The raw test features.
        x_test_cleaned (np.ndarray): The cleaned test features.
        y_test (np.ndarray): The test labels.
        text_column (str, optional): The column containing the text. Defaults to "text".
        label_column (str, optional): The column containing the label. Defaults to "label".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        save (bool, optional): Whether to save the misclassified examples. Defaults to True.

    Returns:
        pd.DataFrame: The misclassified examples.
    """
    if verbose:
        console.print(f"\n [bold white] Finding misclassified examples: {model_name} [/bold white]")

    y_pred = model.predict(x_test_cleaned)

    y_pred, y_test = np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1)

    y_pred, y_test = np.array([CATEGORIES[x] for x in y_pred]), np.array([CATEGORIES[x] for x in y_test])

    mask = y_pred != y_test
    misclassified = x_test_raw[mask]

    misclassified.insert(loc=0, column="Predicted Class", value=y_pred[mask])
    misclassified.insert(loc=0, column="Actual Class", value=y_test[mask])

    misclassified.insert(
        loc=3, column="Cleaned_Text", value=x_test_cleaned[mask]
    )

    misclassified.rename(columns={text_column: "Raw_Text"}, inplace=True)
    misclassified = misclassified.reset_index(drop=True).drop(label_column, axis=1)

    if save:
        save_path = (
            Path(__file__).parent / "misclassified" / f"{model_name}_misclassified.csv"
        )
        misclassified.to_csv(save_path, index=False)

    if verbose:
        console.print(f"\n Found {len(misclassified)} misclassified examples.")
        console.print(misclassified.head())

    return misclassified
