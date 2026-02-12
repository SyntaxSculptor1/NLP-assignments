import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from typing import Iterator, List
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")

SEED = 67
SPLIT = 0.1

def load_data():
    full_train_data = load_dataset("sh0416/ag_news", split="train").to_pandas()
    train_data, validation_data = train_test_split(full_train_data, random_state=SEED, test_size=SPLIT, shuffle=True)

    test_data = load_dataset("sh0416/ag_news", split="test").to_pandas()

    if isinstance(train_data, Iterator) or isinstance(validation_data, Iterator) or isinstance(test_data, Iterator):
        raise Exception("One of the datasets is an Iterator object")

    return pd.DataFrame(train_data), pd.DataFrame(validation_data), pd.DataFrame(test_data)

def merge_inplace_columns(datasets:List[pd.DataFrame]):
    '''
    Inplace merge of title and description
    '''
    for dataset in datasets:
        dataset["features"] = dataset["title"].astype(str) +" "+ dataset["description"].astype(str)
        dataset.drop(["title", "description"], axis=1, inplace=True)

def tokenize_inplace(datasets: List[pd.DataFrame]):
    for dataset in datasets:
        dataset["features"] = dataset["features"].apply(word_tokenize)

def normalize_inplace(datasets: List[pd.DataFrame]):

    for dataset in datasets:
        pass



def main():
    train_data, validation_data, test_data = load_data()

    merge_inplace_columns([train_data, validation_data, test_data])

    #normalize_inplace

    train_data.to_csv("ag_news_processed.csv", index=False)

    tokenize_inplace([train_data, validation_data, test_data])



if __name__ == "__main__":
    main()
