import argparse

from rich.console import Console
from rich.panel import Panel

import torch
from datasets import Dataset
from transformers import AutoTokenizer

from load_data import load_data
from preprocess_data import merge_title_description, headline_only_selection, convert_dataframes_to_dataset
from tokenization import load_tokenizer, tokenize_datasets
from models import load_automodel, train_model
from evaluation import evaluate_model
from utils import get_model_settings, CATEGORIES


console = Console()

def argument_parsing() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="distilbert-base-uncased",
        type=str,
        help="Model to use. Default: distilbert-base-uncased",
    )

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
        "--padding", default="max_length", type=str, help="Padding type. Default: max_length"
    )

    parser.add_argument(
        "--truncation", default=True, type=bool, help="Truncation. Default: True"
    )

    parser.add_argument(
        "--batch_size", default=32, type=int, help="Batch size. Default: 32"
    )

    parser.add_argument(
        "--epochs", default=15, type=int, help="Number of epochs. Default: 15"
    )

    parser.add_argument(
        "--learning_rate", default=2e-5, type=float, help="Learning Rate. Default: 2e-5"
    )

    parser.add_argument(
        "--weight_decay", default=0.01, type=float, help="Weight decay. Default: 0.01"
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
        "[bold white] Assignment 3 - Transformer Fine-tuning + Robustness + Limitations [/bold white]"
    )

    console.print(program_title)

    args = argument_parsing()

    MODEL = args.model
    DATASET = args.dataset
    VERBOSE = args.verbose
    SPLIT = args.split
    SEED = args.seed
    PADDING = args.padding
    TRUNCATION = args.truncation
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data, validation_data, test_data_raw = load_data(
        dataset=DATASET,
        label_column="label",
        split=SPLIT,
        seed=SEED,
        verbose=VERBOSE,
    )

    headline_description_datasets = merge_title_description(
        datasets=[train_data, validation_data, test_data_raw],
        title_column="title",
        description_column="description",
        new_text_column="text",
        label_column="label",
        new_label_column="labels",
        verbose=VERBOSE,
    )

    headline_datasets = headline_only_selection(
        datasets=[train_data, validation_data, test_data_raw],
        headline_column="title",
        new_text_column="text",
        label_column="label",
        new_label_column="labels",
        verbose=VERBOSE,
    )

    headline_description_datasets = convert_dataframes_to_dataset(
        name="Headline & Description",
        datasets=headline_description_datasets,
        verbose=VERBOSE,
    )
    headline_datasets = convert_dataframes_to_dataset(
        name="Headline Only",
        datasets=headline_datasets,
        verbose=VERBOSE,
    )

    tokenizer = load_tokenizer(
        model_name=MODEL,
        verbose=VERBOSE,
    )

    headline_description_datasets = tokenize_datasets(
        name="Headline & Description",
        datasets=headline_description_datasets,
        tokenizer=tokenizer,
        text_column="text",
        padding_type=PADDING,
        truncation=TRUNCATION,
        verbose=VERBOSE,
    )
    headline_datasets = tokenize_datasets(
        name="Headline Only",
        datasets=headline_datasets,
        tokenizer=tokenizer,
        text_column="text",
        padding_type=PADDING,
        truncation=TRUNCATION,
        verbose=VERBOSE,
    )

    model = load_automodel(
        model_name=MODEL,
        num_labels=len(CATEGORIES),
        verbose=VERBOSE,
    )

    for model_name, model_settings in get_model_settings().items():        
        if model_settings["description"]:
            train_data, validation_data, test_data = headline_description_datasets
        else:
            train_data, validation_data, test_data = headline_datasets
            
        noise = model_settings["noise"]
        
        train_data_length = len(train_data)
        train_data_indices = range(int(train_data_length * noise))
        train_data = train_data.shuffle(seed=SEED).select(train_data_indices)
        
        trainer = train_model(
            name=model_name,
            automodel=model,
            train_data=train_data,
            validation_data=validation_data,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            weight_decay=WEIGHT_DECAY,
            learning_rate=LEARNING_RATE,
            verbose=VERBOSE,
        )

        evaluate_model(
            model=trainer,
            model_name=model_name,
            set_name="Validation",
            test_dataset=validation_data,
            verbose=VERBOSE,
            plot=True,
        )

        evaluate_model(
            model=trainer,
            model_name=model_name,
            set_name="Test",
            test_dataset=test_data,
            verbose=VERBOSE,
            plot=True
        )

if __name__ == "__main__":
    main()
