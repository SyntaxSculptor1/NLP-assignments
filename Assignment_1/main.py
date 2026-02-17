import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from datasets import load_dataset
from typing import Iterator, List
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import html
import string
from scipy.sparse import csr_matrix, vstack
import numpy as np


nltk.download("wordnet")
nltk.download("punkt_tab")

SEED = 67
SPLIT = 0.1

#! THIS DOES NOT WORK AT ALL BUT THE IDEA IS SIMILAR TO THIS
LEMMATIZER = WordNetLemmatizer()
VECTORIZER = TfidfVectorizer(
    stop_words="english",
    token_pattern=r"(?u)\b\w\w+\b",
)


def load_data():
    full_train_data = load_dataset("sh0416/ag_news", split="train").to_pandas()
    train_data, validation_data = train_test_split(
        full_train_data, random_state=SEED, test_size=SPLIT, shuffle=True
    )

    test_data = load_dataset("sh0416/ag_news", split="test").to_pandas()

    if (
        isinstance(train_data, Iterator)
        or isinstance(validation_data, Iterator)
        or isinstance(test_data, Iterator)
    ):
        raise Exception("One of the datasets is an Iterator object")

    return (
        pd.DataFrame(train_data),
        pd.DataFrame(validation_data),
        pd.DataFrame(test_data),
    )


def merge_inplace_columns(datasets: List[pd.DataFrame]):
    """
    Inplace merge of title and description
    """
    for dataset in datasets:
        dataset["features"] = (
            dataset["title"].astype(str) + " " + dataset["description"].astype(str)
        )
        dataset.drop(["title", "description"], axis=1, inplace=True)


def pre_tokenize_clean_helper(text: str):
    """
    Helper function that tells us what to clean up when tokenizing
    """
    # convert HTML entities into < and >
    text = html.unescape(text)
    # remove entire HTML tags between < >
    clean_text = re.sub(r"<[^>]+>", "", text)
    # clean up white spaces
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    return clean_text


def pre_tokenization_normalisation_inplace(datasets: List[pd.DataFrame]):
    """
    Normalisation done before tokenisation in place
    """

    for dataset in datasets:
        dataset["features"] = dataset["features"].apply(pre_tokenize_clean_helper)


def tokenize_inplace(datasets: List[pd.DataFrame]):
    for dataset in datasets:
        dataset["features"] = dataset["features"].apply(word_tokenize)


##! NORMALISATION NEEDS THE FOLLOWING:
# 1. lower case
# 2. lemmatisation
# 3. Stop word removal
# 4. maybe number removal?
def post_tokenize_normalize_helper(text: List[str]) -> List[str]:
    cleaned_text = []

    for word in text:
        word = word.strip(string.punctuation)

        if not word:
            continue

        word = word.lower()
        word_lemma = LEMMATIZER.lemmatize(word)

        cleaned_text.append(word_lemma)
    
    return " ".join(cleaned_text)

def post_tokenize_normalize_inplace(datasets: List[pd.DataFrame]):
    for dataset in datasets:
        dataset["features"] = dataset["features"].apply(post_tokenize_normalize_helper)

def tfidf_transform(datasets: List[pd.DataFrame]) -> List[pd.DataFrame]:
    x_values = []
    for idx, dataset in enumerate(datasets):
        if idx == 0:
            features = VECTORIZER.fit_transform(dataset["features"])
        else:
            features = VECTORIZER.transform(dataset["features"])

        x_values.append(features)

    return x_values

def logistic_regression(
    x_train: csr_matrix,
    x_val: csr_matrix,
    x_test: csr_matrix,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series
) -> None:

    scaler = MaxAbsScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    x_full = vstack([x_train_scaled, x_val_scaled])
    y_full = np.concatenate([y_train, y_val])

    train_indices = np.full(x_train.shape[0], -1)
    val_indices = np.full(x_val.shape[0], 0)

    test_fold = np.concatenate([train_indices, val_indices])

    ps = PredefinedSplit(test_fold)

    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2']
    }

    grid_search = GridSearchCV(
        estimator=LogisticRegression(max_iter=5000),
        param_grid=param_grid,
        cv=ps, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(x_full, y_full)

    print(f"Best C found: {grid_search.best_params_['C']}")
    print(f"Validation Accuracy: {grid_search.best_score_}")

    return grid_search.best_estimator_


def main():
    train_data, validation_data, test_data = load_data()

    train_data.to_csv("ag_news_raw.csv", index=False)

    merge_inplace_columns([train_data, validation_data, test_data])

    pre_tokenization_normalisation_inplace([train_data, validation_data, test_data])

    tokenize_inplace([train_data, validation_data, test_data])

    post_tokenize_normalize_inplace([train_data, validation_data, test_data])

    x_train, x_val, x_test = tfidf_transform([train_data, validation_data, test_data])
    y_train, y_val, y_test = train_data["label"], validation_data["label"], test_data["label"]

    logistic_regression(
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test
    )

    train_data.to_csv("ag_news_processed.csv", index=False)


if __name__ == "__main__":
    main()
