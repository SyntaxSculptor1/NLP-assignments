import nltk
import pandas as pd
from typing import Iterator, Tuple
from datasets import load_dataset
from sklearn.model_selection import train_test_split

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

