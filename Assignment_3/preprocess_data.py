import html
import re
import string
from typing import List, Tuple


import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from rich.console import Console

console = Console()

def headline_only_selection(
    datasets: List[pd.DataFrame],
    headline_column: str = "title",
    new_text_column: str = "text",
    label_column: str = "label",
    new_label_column: str = "labels",
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if verbose:
        console.print(f"\n [bold white] Selecting {headline_column} column: [/bold white]")

    cleaned_datasets = list(dataset.copy(deep=True) for dataset in datasets)

    for dataset in cleaned_datasets:
        dataset[new_text_column] = dataset[headline_column]
        dataset[new_label_column] = dataset[label_column]
        cleaned_datasets[0] = dataset[[new_text_column, new_label_column]]

    cleaned_datasets = tuple(cleaned_datasets)

    if verbose:
        console.print(f"Selected {headline_column} column.")

    assert len(cleaned_datasets) == 3

    return cleaned_datasets

def merge_title_description(
    datasets: List[pd.DataFrame],
    title_column: str = "title",
    description_column: str = "description",
    new_text_column: str = "text",
    label_column: str = "label",
    new_label_column: str = "labels",
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if verbose:
        console.print(f"\n [bold white] Merging {title_column} and {description_column} columns into {new_text_column}: [/bold white]")

    cleaned_datasets = tuple(dataset.copy(deep=True) for dataset in datasets)

    for dataset in cleaned_datasets:
        dataset[new_text_column] = dataset[title_column].astype(str) + " " + dataset[description_column].astype(str)
        dataset.drop([title_column, description_column], axis=1, inplace=True)
        dataset.rename(columns={label_column: new_label_column}, inplace=True)

    if verbose:
        console.print(f"Merged {title_column} and {description_column} into {new_text_column}.")

    assert len(cleaned_datasets) == 3

    return cleaned_datasets

def convert_dataframes_to_dataset(
    name: str,
    datasets: List[pd.DataFrame],
    verbose: bool = True
) -> List[Dataset]:
    if verbose:
        console.print(f"\n [bold white] Converting DataFrames to Datasets - {name}: [/bold white]")
        
    converted_datasets = [Dataset.from_pandas(dataset) for dataset in datasets]
    
    if verbose:
        console.print(f"Converted {name} dataframes to Datasets.")
        
    return converted_datasets