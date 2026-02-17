import string
import re
import html

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import MaxAbsScaler
from datasets import load_dataset

import numpy as np
from scipy import sparse

from typing import List, Iterator, Tuple
import argparse


def load_nltk_models() -> None:
    nltk.download("wordnet")
    nltk.download("punkt_tab")

def load_data(
    dataset: str = "sh0416/ag_news",
    split: float = 0.1,
    seed: int = 67
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    full_train_data = load_dataset(dataset, split="train").to_pandas()

    train_data, validation_data = train_test_split(
        full_train_data, random_state=seed, test_size=split, shuffle=True
    )

    test_data = load_dataset(dataset, split="test").to_pandas()

    assert (
        not isinstance(train_data, Iterator) and
        not isinstance(validation_data, Iterator) and
        not isinstance(test_data, Iterator)
    ) 

    return (
        pd.DataFrame(train_data),
        pd.DataFrame(validation_data),
        pd.DataFrame(test_data),
    )


def merge_title_description(
    datasets: List[pd.DataFrame],
    title_column: str = "title",
    description_column: str = "description",
    new_column: str = "text"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    datasets = tuple(dataset.copy(deep=True) for dataset in datasets)

    for dataset in datasets:
        dataset[new_column] = (
            dataset[title_column].astype(str) + " " + dataset[description_column].astype(str)
        )
        dataset.drop([title_column, description_column], axis=1, inplace=True)
    
    return datasets

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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    datasets = tuple(dataset.copy(deep=True) for dataset in datasets)

    for dataset in datasets:
        dataset[column] = dataset[column].apply(pre_tokenization_normalization_helper)
        dataset[column] = dataset[column].apply(word_tokenize)
        dataset[column] = dataset[column].apply(post_tokenization_normalization_helper, lemmatizer=lemmatizer)

    return datasets

def tfidf_transform(
    train_texts: pd.Series,
    validation_texts: pd.Series,
    test_texts: pd.Series,
    vectorizer: TfidfVectorizer
) -> List[sparse.csr_matrix]:
    features = []
    for idx, dataset in enumerate([train_texts, validation_texts, test_texts]):
        if idx == 0:
            text_features = vectorizer.fit_transform(dataset)
        else:
            text_features = vectorizer.transform(dataset)

        features.append(text_features)

    return features

def perform_logistic_regression(
    x_train: sparse.csr_matrix,
    x_val: sparse.csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    param_grid: dict,
    scoring: str = "accuracy",
    max_iter: int = 5000,
    verbose: int = 1,
    seed: int = 67
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
        estimator=LogisticRegression(max_iter=max_iter, random_state=seed),
        param_grid=param_grid,
        cv=predefined_split, 
        scoring=scoring,
        verbose=verbose,
        n_jobs=-1,
    )

    grid_search.fit(x_full, y_full)

    return grid_search

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
        estimator=SVC(),
        param_grid=param_grid,
        cv=predefined_split, 
        scoring=scoring,
        verbose=verbose,
        n_jobs=-1,
    )

    grid_search.fit(x_full, y_full)

    return grid_search

def evaluate_logistic_regression(
    model: LogisticRegression,
    x_test: sparse.csr_matrix,
    y_test: pd.Series
) -> None:
    y_pred = model.predict(x_test)

    print(f"Model - Logistic Regression (C = {model.get_params()['C']})")
    print("Accuracy:", accuracy_score(y_true= y_test, y_pred= y_pred))
    print("F1 Score:", f1_score(y_true=y_test, y_pred=y_pred, average="macro"))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true= y_test, y_pred= y_pred))

def evaluate_support_vector_machine(
    model: SVC,
    x_test: sparse.csr_matrix,
    y_test: pd.Series
) -> None:
    y_pred = model.predict(x_test)

    print(f"Model - Support Vector Machine (C = {model.get_params()['C']})")
    print("Accuracy:", accuracy_score(y_true= y_test, y_pred= y_pred))
    print("F1 Score:", f1_score(y_true=y_test, y_pred=y_pred, average="macro"))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true= y_test, y_pred= y_pred))

def argument_parsing() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        default="sh0416/ag_news",
        type=str,
        help="Dataset to use. Default: sh0416/ag_news"
    )

    parser.add_argument(
        "--verbose",
        default=True,
        type=bool,
        help="Verbose mode. Default: True"
    )

    parser.add_argument(
        "--split",
        default=0.1,
        type=float,
        help="Split ratio. Default: 0.1"
    )

    parser.add_argument(
        "--seed",
        default=67,
        type=int,
        help="Random seed. Default: 67"
    )

    args = parser.parse_args()

    return args

def main() -> None:
    args = argument_parsing()

    DATASET = args.dataset
    VERBOSE = args.verbose
    SPLIT = args.split
    SEED = args.seed

    load_nltk_models()

    train_data, validation_data, test_data = load_data(
        dataset=DATASET,
        split=SPLIT,
        seed=SEED
    )

    train_data, validation_data, test_data = merge_title_description(
        datasets=[train_data, validation_data, test_data],
        title_column="title",
        description_column="description",
        new_column="text"
    )

    lemmatizer = WordNetLemmatizer()
    train_data, validation_data, test_data = text_cleaning(
        datasets=[train_data, validation_data, test_data],
        column="text",
        lemmatizer = lemmatizer
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english"
    )

    x_train, x_val, x_test = tfidf_transform(
        train_texts= train_data["text"],
        validation_texts= validation_data["text"],
        test_texts= test_data["text"],
        vectorizer= vectorizer
    )

    y_train, y_val, y_test = train_data["label"], validation_data["label"], test_data["label"]

    logreg_param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100]
    }
    logreg_grid_search = perform_logistic_regression(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        param_grid=logreg_param_grid,
        scoring="accuracy",
        max_iter=5000,
        verbose=int(VERBOSE),
        seed=SEED
    )

    evaluate_logistic_regression(
        model=logreg_grid_search.best_estimator_,
        x_test=x_test,
        y_test=y_test
    )

    svm_param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100]
    }
    svm_grid_search = perform_support_vector_machine(
        x_train=x_train,
        x_val=x_val,
        y_train=y_train,
        y_val=y_val,
        param_grid=svm_param_grid,
        scoring="accuracy",
        verbose=int(VERBOSE)
    )

    evaluate_support_vector_machine(
        model=svm_grid_search.best_estimator_,
        x_test=x_test,
        y_test=y_test
    )


if __name__ == "__main__":
    main()
