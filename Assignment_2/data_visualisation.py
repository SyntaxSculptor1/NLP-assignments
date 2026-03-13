import argparse

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from load_data import load_data, load_nltk_models
from nltk.stem import WordNetLemmatizer
from preprocess_data import merge_title_description, text_cleaning
from rich.console import Console
from rich.panel import Panel

console = Console()


def argument_parsing() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="sh0416/ag_news",
        type=str,
        help="Dataset to use. Default: sh0416/ag_news",
    )

    parser.add_argument("--verbose", default=True, type=bool, help="Verbose mode. Default: True")

    parser.add_argument("--save", default=True, type=bool, help="Save mode. Default: True")

    parser.add_argument("--split", default=0.1, type=float, help="Split ratio. Default: 0.1")

    parser.add_argument("--seed", default=67, type=int, help="Random seed. Default: 67")

    args = parser.parse_args()

    return args


def transform_text_length(data: pd.DataFrame, text_column: str = "text", verbose: bool = True) -> pd.Series:
    """
    Transform the text length into a pandas Series.

    Args:
        data (pd.DataFrame): DataFrame containing the text column.
        text_column (str, optional): The name of the text column. Defaults to "text".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        pd.Series: Series containing the text lengths.
    """
    return data[text_column].apply(lambda x: len(x.split(" ")))

def plot_text_length(data: pd.DataFrame, text_column: str = "text", save: bool = True, verbose: bool = True) -> None:
    """
    Plot the text length distribution of the given DataFrame.

    Parameters:
        data (pd.DataFrame): DataFrame containing the text column.
        text_column (str, optional): The name of the text column. Defaults to "text".
        save (bool, optional): Whether to save the plot. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        None
    """
    if verbose:
        console.print(f"\n[bold white] Plotting text length distribution: [/bold white]")

    text_length = transform_text_length(data=data, text_column=text_column)
    plt.hist(text_length, bins=20)
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.title("Text Length Distribution")

    if save is not None:
        save_path = Path(__file__).parent / "plots" / "text_length_distribution.png"
        plt.savefig(save_path)

    plt.show()

    if verbose:
        console.print(f"\nText length distribution plotted.")

def main() -> None:
    """
    Main function

    Returns:
        None
    """

    program_title = Panel("[bold white] Assignment 2 - Neural Model Comparison + Ablation - Data visualisation [/bold white]")
    console.print(program_title)

    args = argument_parsing()

    DATASET = args.dataset
    VERBOSE = args.verbose
    SPLIT = args.split
    SEED = args.seed
    SAVE = args.save

    load_nltk_models(verbose=VERBOSE)

    train_data, validation_data, test_data_raw = load_data(dataset=DATASET, split=SPLIT, seed=SEED, verbose=VERBOSE)

    train_data, validation_data, test_data_raw = merge_title_description(
        datasets=[train_data, validation_data, test_data_raw],
        title_column="title",
        description_column="description",
        new_column="text",
        verbose=VERBOSE,
    )

    lemmatizer = WordNetLemmatizer()
    train_data, validation_data, test_data = text_cleaning(
        datasets=[train_data, validation_data, test_data_raw], column="text", lemmatizer=lemmatizer, verbose=VERBOSE
    )

    full_data = pd.concat([train_data, validation_data, test_data], ignore_index=True)

    plot_text_length(data=full_data, text_column="text", save=SAVE, verbose=VERBOSE)

if __name__ == "__main__":
    main()
