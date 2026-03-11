import argparse

import pandas as pd

from models import perform_cnn, perform_lstm
from evaluation import evaluate_model, find_misclassified
from load_data import load_data, load_nltk_models
from keras.layers import TextVectorization
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

    parser.add_argument("--split", default=0.1, type=float, help="Split ratio. Default: 0.1")

    parser.add_argument("--seed", default=67, type=int, help="Random seed. Default: 67")

    parser.add_argument("--padding", default=100, type=int, help="Padding length. Default: 100")

    parser.add_argument("--max_tokens", default=10000, type=int, help="Maximum number of tokens. Default: 10000")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size. Default: 32")

    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs. Default: 100")

    args = parser.parse_args()

    return args


def main() -> None:
    """
    Main function

    Returns:
        None
    """

    program_title = Panel("[bold white] Assignment 2 - Neural Model Comparison + Ablation [/bold white]")
    console.print(program_title)

    args = argument_parsing()

    DATASET = args.dataset
    VERBOSE = args.verbose
    SPLIT = args.split
    SEED = args.seed
    PADDING = args.padding
    MAX_TOKENS = args.max_tokens
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs


    load_nltk_models(verbose=VERBOSE)

    train_data, validation_data, test_data = load_data(dataset=DATASET, split=SPLIT, seed=SEED, verbose=VERBOSE)

    train_data, validation_data, test_data = merge_title_description(
        datasets=[train_data, validation_data, test_data],
        title_column="title",
        description_column="description",
        new_column="text",
        verbose=VERBOSE,
    )

    lemmatizer = WordNetLemmatizer()
    train_data, validation_data, test_data = text_cleaning(
        datasets=[train_data, validation_data, test_data], column="text", lemmatizer=lemmatizer, verbose=VERBOSE
    )

    x_train, x_val, x_test = (
        train_data["text"].to_numpy(),
        validation_data["text"].to_numpy(),
        test_data["text"].to_numpy(),
    )

    y_full = pd.concat(
        [train_data["label"], validation_data["label"], test_data["label"]],
        ignore_index=True
    )
    y_full = pd.get_dummies(y_full)

    y_train, y_val, y_test = (
        y_full[: len(train_data)].to_numpy(),
        y_full[len(train_data) : len(train_data) + len(validation_data)].to_numpy(),
        y_full[len(train_data) + len(validation_data):].to_numpy(),
    )

    vectorizer = TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=PADDING,
    )

    vectorizer.adapt(x_train)

    model = perform_lstm(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        vectorizer=vectorizer,
        padding_length=PADDING,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=VERBOSE,
    )

if __name__ == "__main__":
    main()
