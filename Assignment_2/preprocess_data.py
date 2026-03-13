import html
import re
import string
from typing import List, Tuple

import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from rich.console import Console

console = Console()


def merge_title_description(
    datasets: List[pd.DataFrame], title_column: str = "title", description_column: str = "description", new_column: str = "text", verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Merge the title and description columns into a new column.

    Args:
        datasets (List[pd.DataFrame]): The datasets to merge.
        title_column (str, optional): The column containing the title. Defaults to "title".
        description_column (str, optional): The column containing the description. Defaults to "description".
        new_column (str, optional): The name of the new column. Defaults to "text".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The merged datasets.
    """
    if verbose:
        console.print(f"\n [bold white] Merging {title_column} and {description_column} columns into {new_column}: [/bold white]")

    cleaned_datasets = tuple(dataset.copy(deep=True) for dataset in datasets)

    for dataset in cleaned_datasets:
        dataset[new_column] = dataset[title_column].astype(str) + " " + dataset[description_column].astype(str)
        dataset.drop([title_column, description_column], axis=1, inplace=True)

    if verbose:
        console.print(f"Merged {title_column} and {description_column} into {new_column}.")

    assert len(cleaned_datasets) == 3

    return cleaned_datasets


def pre_tokenization_normalization_helper(text: str) -> str:
    """
    Pre-tokenization normalization helper. By default, this function normalizes the text by removing HTML tags,
    removing extra spaces, and converting the text to lowercase.

    Args:
        text (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    text = html.unescape(text)
    clean_text = re.sub(r"<[^>]+>", "", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    clean_text = clean_text.lower()
    return clean_text


def post_tokenization_normalization_helper(text: List[str], lemmatizer: WordNetLemmatizer) -> str:
    """
    Post tokenization normalization helper. By default, this function normalizes the text by removing punctuation,
    removing empty strings, and lemmatizing the words.

    Args:
        text (List[str]): The text to normalize.
        lemmatizer (WordNetLemmatizer): The lemmatizer to use.

    Returns:
        str: The normalized text.
    """
    cleaned_text = []

    for word in text:
        word = word.strip(string.punctuation)

        if not word:
            continue

        word_lemma = lemmatizer.lemmatize(word)

        cleaned_text.append(word_lemma)

    return " ".join(cleaned_text)


def text_cleaning(
    datasets: List[pd.DataFrame], lemmatizer: WordNetLemmatizer, column: str = "text", verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Text cleaning. By default, this function normalizes the text by removing HTML tags, removing extra spaces,
    converting the text to lowercase, removing punctuation, removing empty strings, and lemmatizing the words.

    Args:
        datasets (List[pd.DataFrame]): The datasets to clean.
        lemmatizer (WordNetLemmatizer): The lemmatizer to use.
        column (str, optional): The column to clean. Defaults to "text".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The cleaned datasets.
    """

    if verbose:
        console.print(f"\n [bold white] Cleaning {column} by normalization: [/bold white]")

    cleaned_datasets = tuple(dataset.copy(deep=True) for dataset in datasets)

    for dataset in cleaned_datasets:
        dataset[column] = dataset[column].apply(pre_tokenization_normalization_helper)
        dataset[column] = dataset[column].apply(word_tokenize)
        dataset[column] = dataset[column].apply(post_tokenization_normalization_helper, lemmatizer=lemmatizer)

    if verbose:
        console.print(f"Cleaned {column} by normalization.")

    assert len(cleaned_datasets) == 3

    return cleaned_datasets


def onehotencode_labels(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    One-hot encode the labels.

    Args:
        y_train (pd.Series): The training labels.
        y_val (pd.Series): The validation labels.
        y_test (pd.Series): The test labels.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The one-hot encoded labels.
    """
    if verbose:
        console.print(f"\n [bold white] One-hot encoding labels: [/bold white]")
    
    y_full = pd.concat(
        [y_train, y_val, y_test],
        ignore_index=True,
    )

    y_full = pd.get_dummies(y_full).astype(int)

    y_train_encoded, y_val_encoded, y_test_encoded = (
        y_full[: len(y_train)],
        y_full[len(y_train) : len(y_train) + len(y_val)],
        y_full[len(y_train) + len(y_val) :],
    )

    if verbose:
        console.print("One-hot encoded labels.")

    return y_train_encoded, y_val_encoded, y_test_encoded

