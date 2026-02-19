from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
from rich.console import Console
from scipy import sparse
import pandas as pd

console = Console()

def evaluate_model(
    model: LogisticRegression | LinearSVC,
    model_name: str,
    x_test: sparse.csr_matrix,
    y_test: pd.Series,
    verbose: bool = True,
    plot: bool = True
) -> None:
    """
    Evaluate the performance of a given model on the test set.

    Args:
        model (LogisticRegression | LinearSVC): The model to evaluate.
        model_name (str): The name of the model.
        x_test (sparse.csr_matrix): The test features.
        y_test (pd.Series): The test labels.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        plot (bool, optional): Whether to plot the confusion matrix. Defaults to True.
    
    Returns:
        None
    """
    if verbose:
        console.print(f"\n [bold white] Evaluating {model_name}: [/bold white]")

    y_pred = model.predict(x_test)
    c = model.get_params()['C']
    accuracy = accuracy_score(y_true= y_test, y_pred= y_pred)
    f1 = f1_score(y_true=y_test, y_pred=y_pred, average="macro")
    cm = confusion_matrix(y_true= y_test, y_pred= y_pred)

    console.print(f"Model - Support Vector Machine (C = {c})")
    console.print("Accuracy:", accuracy)
    console.print("F1 Score:", f1)

    console.print("Confusion Matrix:")
    console.print("Rows: Actual Class, Columns: Predicted Class")
    console.print(cm)

    if plot:
        save_path = Path(__file__).parent / "plots" / f"{model_name}_confusion_matrix.png"

        display = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
        display.plot()
        display.ax_.set_title(f"Confusion Matrix of {model_name} (C = {c})")
        display.figure_.savefig(str(save_path))

    if verbose:
        console.print(f"Finished evaluating {model_name}.")


def find_misclassified(
    model: LogisticRegression | LinearSVC,
    model_name: str,
    x_test_raw: pd.DataFrame,
    x_test_cleaned: pd.DataFrame,
    x_test: sparse.csr_matrix,
    y_test: pd.Series,
    text_column: str = "text",
    label_column: str = "label",
    verbose: bool = True,
    save: bool = True
) -> pd.DataFrame:
    """
    Find misclassified examples in the test set.

    Args:
        model (LogisticRegression | LinearSVC): The model to evaluate.
        model_name (str): The name of the model.
        x_test_raw (pd.DataFrame): The raw test features.
        x_test_cleaned (pd.DataFrame): The cleaned test features.
        x_test (sparse.csr_matrix): The test features.
        y_test (pd.Series): The test labels.
        text_column (str, optional): The column containing the text. Defaults to "text".
        label_column (str, optional): The column containing the label. Defaults to "label".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        save (bool, optional): Whether to save the misclassified examples. Defaults to True.
    
    Returns:
        pd.DataFrame: The misclassified examples.
    """
    if verbose:
        console.print(f"\n [bold white] Finding misclassified examples: [/bold white]")

    y_pred = model.predict(x_test)

    mask = y_pred != y_test
    misclassified = x_test_raw[mask]
    misclassified.insert(loc=0, column="Predicted Class", value=y_pred[mask])
    misclassified.insert(loc=0, column="Actual Class", value=y_test[mask])
    misclassified.insert(loc=3, column="Cleaned_Text", value=x_test_cleaned[text_column][mask])
    misclassified.rename(columns={text_column: "Raw_Text"}, inplace=True)
    misclassified = misclassified.reset_index(drop=True).drop(label_column, axis=1)

    if save:
        save_path = Path(__file__).parent / "misclassified" / f"{model_name}_misclassified.csv"
        misclassified.to_csv(save_path, index=False)

    if verbose:
        console.print(f"Found {len(misclassified)} misclassified examples.")
        console.print(misclassified.head())
        

    return misclassified
    
