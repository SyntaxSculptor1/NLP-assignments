import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
from typing import Iterator, List
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import html

nltk.download("punkt_tab")

SEED = 67
SPLIT = 0.1

#! THIS DOES NOT WORK AT ALL BUT THE IDEA IS SIMILAR TO THIS
LEMMATIZER = WordNetLemmatizer()
VECTORIZER = TfidfVectorizer(
    tokenizer=LEMMATIZER,
    lowercase=True,
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
def normalization_helper(text: str):

    return text


def normalize_inplace(datasets: List[pd.DataFrame]):
    # DOES NOT WORK
    for dataset in datasets:
        dataset["features"] = VECTORIZER.fit_transform(dataset["features"])


def main():
    train_data, validation_data, test_data = load_data()

    train_data.to_csv("ag_news_raw.csv", index=False)

    merge_inplace_columns([train_data, validation_data, test_data])

    pre_tokenization_normalisation_inplace([train_data, validation_data, test_data])

    tokenize_inplace([train_data, validation_data, test_data])

    # CURRENTLY WORKS UP TO HERE, NORMALISATION DOES NOT WORK YET
    normalize_inplace([train_data, validation_data, test_data])

    train_data.to_csv("ag_news_processed.csv", index=False)

    print("We are done")


if __name__ == "__main__":
    main()
