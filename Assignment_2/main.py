import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse

from models import training_cnn, training_lstm
from evaluation import evaluate_model, find_misclassified
from load_data import load_data, load_nltk_models
from keras.layers import TextVectorization
from nltk.stem import WordNetLemmatizer
from preprocess_data import merge_title_description, text_cleaning, onehotencode_labels
from rich.console import Console
from rich.panel import Panel

import warnings
import tensorflow as tf
import absl.logging
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
absl.logging.set_verbosity(absl.logging.ERROR)

console = Console()

models_mapping = {
        "CNN-Dropout-0": {
            "dropout": 0.0,
            "model": training_cnn,
        },
        "CNN-Dropout-0.3": {
            "dropout": 0.3,
            "model": training_cnn,
        },
        "LSTM-Dropout-0": {
            "dropout": 0.0,
            "model": training_lstm,
        },
        "LSTM-Dropout-0.3": {
            "dropout": 0.3,
            "model": training_lstm,
        },
    }


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

    parser.add_argument(
        "--verbose", default=True, type=bool, help="Verbose mode. Default: True"
    )

    parser.add_argument(
        "--split", default=0.1, type=float, help="Split ratio. Default: 0.1"
    )

    parser.add_argument("--seed", default=67, type=int, help="Random seed. Default: 67")

    parser.add_argument(
        "--padding", default=100, type=int, help="Padding length. Default: 100"
    )

    parser.add_argument(
        "--max_tokens",
        default=10000,
        type=int,
        help="Maximum number of tokens. Default: 10000",
    )

    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size. Default: 32"
    )

    parser.add_argument(
        "--epochs", default=100, type=int, help="Number of epochs. Default: 100"
    )

    args = parser.parse_args()

    return args


def main() -> None:
    """
    Main function

    Returns:
        None
    """

    program_title = Panel(
        "[bold white] Assignment 2 - Neural Model Comparison + Ablation [/bold white]"
    )

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

    train_data, validation_data, test_data_raw = load_data(
        dataset=DATASET, split=SPLIT, seed=SEED, verbose=VERBOSE
    )

    train_data, validation_data, test_data_raw = merge_title_description(
        datasets=[train_data, validation_data, test_data_raw],
        title_column="title",
        description_column="description",
        new_column="text",
        verbose=VERBOSE,
    )

    lemmatizer = WordNetLemmatizer()
    train_data, validation_data, test_data = text_cleaning(
        datasets=[train_data, validation_data, test_data_raw],
        column="text",
        lemmatizer=lemmatizer,
        verbose=VERBOSE,
    )

    x_train, x_val, x_test = (
        train_data["text"].to_numpy(),
        validation_data["text"].to_numpy(),
        test_data["text"].to_numpy(),
    )

    y_train, y_val, y_test = onehotencode_labels(
        y_train=train_data["label"],
        y_val=validation_data["label"],
        y_test=test_data["label"],
    )

    y_train, y_val, y_test = (
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
    )

    vectorizer = TextVectorization(
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=PADDING,
    )

    vectorizer.adapt(x_train)

    for model_name, model_info in models_mapping.items():
        model = model_info["model"](
            model_name=model_name,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            vectorizer=vectorizer,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            dropout=model_info["dropout"],
            verbose=VERBOSE,
        )

        evaluate_model(
            model=model,
            x_test=x_test,
            y_test=y_test,
            model_name=model_name,
            verbose=VERBOSE,
        )

        find_misclassified(
            model=model,
            model_name=model_name,
            x_test_raw=test_data_raw,
            x_test_cleaned=x_test,
            y_test=y_test,
            verbose=VERBOSE,
        )

if __name__ == "__main__":
    main()
