from typing import Iterator, Tuple

import pandas as pd
from datasets import load_dataset
from rich.console import Console
from sklearn.model_selection import train_test_split

console = Console()

def load_data(
    dataset: str = "sh0416/ag_news", 
    label_column: str = "label",
    split: float = 0.1, 
    seed: int = 67, 
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the data from the dataset.

    Args:
        dataset (str, optional): The dataset to load. Defaults to "sh0416/ag_news".
        split (float, optional): The split ratio. Defaults to 0.1.
        seed (int, optional): The random seed. Defaults to 67.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The train, validation, and test data.
    """
    if verbose:
        console.print(f"\n[bold white] Loading data from {dataset} with split ratio {split}: [/bold white]")

    full_train_data = load_dataset(dataset, split="train").to_pandas()

    train_data, validation_data = train_test_split(full_train_data, random_state=seed, test_size=split, shuffle=True)

    test_data = load_dataset(dataset, split="test").to_pandas()

    assert not isinstance(train_data, Iterator) and not isinstance(validation_data, Iterator) and not isinstance(test_data, Iterator)

    if verbose:
        console.print(f"Loaded data from {dataset}.")
        
    dfs = (pd.DataFrame(train_data), pd.DataFrame(validation_data), pd.DataFrame(test_data))
    
    for df in dfs:
        df[label_column] = df[label_column].apply(lambda x: x - 1)
        
    return dfs
