import pandas as pd
from typing import Tuple, List
import re
import html
import string
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
from rich.console import Console
import numpy as np
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import MaxAbsScaler

console = Console()

def merge_title_description(
    datasets: List[pd.DataFrame],
    title_column: str = "title",
    description_column: str = "description",
    new_column: str = "text",
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if verbose:
        console.print(f"\n [bold white] Merging {title_column} and {description_column} columns into {new_column}: [/bold white]")

    cleaned_datasets = tuple(dataset.copy(deep=True) for dataset in datasets)

    for dataset in cleaned_datasets:
        dataset[new_column] = (
            dataset[title_column].astype(str) + " " + dataset[description_column].astype(str)
        )
        dataset.drop([title_column, description_column], axis=1, inplace=True)
    
    if verbose:
        console.print(f"Merged {title_column} and {description_column} into {new_column}.")

    assert len(cleaned_datasets) == 3

    return cleaned_datasets

def pre_tokenization_normalization_helper(
    text: str
) -> str:
    text = html.unescape(text)
    clean_text = re.sub(r"<[^>]+>", "", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    clean_text = clean_text.lower()
    return clean_text
    
def post_tokenization_normalization_helper(
    text: List[str],
    lemmatizer: WordNetLemmatizer
) -> str:
    cleaned_text = []

    for word in text:
        word = word.strip(string.punctuation)

        if not word:
            continue

        word_lemma = lemmatizer.lemmatize(word)

        cleaned_text.append(word_lemma)
    
    return " ".join(cleaned_text)

def text_cleaning(
    datasets: List[pd.DataFrame],
    lemmatizer: WordNetLemmatizer,
    column: str = "text",
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
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

def tfidf_transform(
    train_texts: pd.Series,
    validation_texts: pd.Series,
    test_texts: pd.Series,
    vectorizer: TfidfVectorizer,
    verbose: bool = True
) -> List[sparse.csr_matrix]:
    
    if verbose:
        console.print(f"\n [bold white] TF-IDF transformation: [/bold white]")

    features = []
    for idx, dataset in enumerate([train_texts, validation_texts, test_texts]):
        if idx == 0:
            text_features = vectorizer.fit_transform(dataset)
        else:
            text_features = vectorizer.transform(dataset)

        features.append(text_features)

    if verbose:
        console.print("TF-IDF transformation on data completed.")

    return features
