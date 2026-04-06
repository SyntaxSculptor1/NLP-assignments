from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from rich.console import Console
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from transformers import Trainer

from utils import CATEGORIES

console = Console()

def evaluate_model(
    model: Trainer,
    model_name: str,
    set_name: str,
    test_dataset: Dataset,
    verbose: bool = True,
    plot: bool = True,
) -> None:
    """
    Evaluate the performance of a given model on the test set.

    Args:
        model (Trainer): The model to evaluate.
        model_name (str): The name of the model.
        set_name (str): The name of the set.
        test_dataset (Dataset): The test dataset.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        plot (bool, optional): Whether to plot the confusion matrix. Defaults to True.

    Returns:
        None
    """
    if verbose:
        console.print(f"\n [bold white] Evaluating model: {model_name} on {set_name} [/bold white]")

    predictions = model.predict(test_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_test = predictions.label_ids

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
            Path(__file__).parent / "plots" / f"{model_name}_confusion_matrix_{set_name}.png"
        )

        display = ConfusionMatrixDisplay(cm, display_labels=CATEGORIES)
        display.plot()
        display.ax_.set_title(f"Confusion Matrix of {model_name} on {set_name}")

        display.figure_.tight_layout()
        display.figure_.savefig(str(save_path), bbox_inches="tight")

    if verbose:
        console.print(f"\n [white] Finished evaluating {model_name} on {set_name}. [/white]")

def find_misclassified(
    model: Trainer,
    model_name: str,
    x_test_raw: pd.DataFrame,
    x_test_cleaned: Dataset,
    text_column: str = "text",
    label_column: str = "labels",
    verbose: bool = True,
    save: bool = True,
) -> pd.DataFrame:

    predictions = model.predict(x_test_cleaned)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_test = predictions.label_ids

    y_pred_named = np.array([CATEGORIES[x] for x in y_pred])
    y_test_named = np.array([CATEGORIES[x] for x in y_test])

    mask = y_pred_named != y_test_named
    misclassified = x_test_raw[mask].copy()

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
