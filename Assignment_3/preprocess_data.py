import html
import re
import string
from typing import List, Tuple

import pandas as pd
from datasets import Dataset
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
    """
    Select the headline column from the dataset.

    Args:
        datasets (List[pd.DataFrame]): The datasets to select the headline column from.
        headline_column (str, optional): The column containing the headline. Defaults to "title".
        new_text_column (str, optional): The name of the new column. Defaults to "text".
        label_column (str, optional): The column containing the label. Defaults to "label".
        new_label_column (str, optional): The name of the new column. Defaults to "labels".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The datasets with the headline column selected.
    """

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
    """
    Merge the title and description columns into a new column.

    Args:
        datasets (List[pd.DataFrame]): The datasets to merge.
        title_column (str, optional): The column containing the title. Defaults to "title".
        description_column (str, optional): The column containing the description. Defaults to "description".
        new_text_column (str, optional): The name of the new column. Defaults to "text".
        label_column (str, optional): The column containing the label. Defaults to "label".
        new_label_column (str, optional): The name of the new column. Defaults to "labels".
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The datasets with the title and description columns merged.
    """

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
    """
    Convert the dataframes to datasets.

    Args:
        name (str): The name of the datasets.
        datasets (List[pd.DataFrame]): The datasets to convert.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        List[Dataset]: The converted datasets.
    """
    if verbose:
        console.print(f"\n [bold white] Converting DataFrames to Datasets - {name}: [/bold white]")
        
    converted_datasets = [Dataset.from_pandas(dataset) for dataset in datasets]
    
    if verbose:
        console.print(f"Converted {name} dataframes to Datasets.")
        
    return converted_datasets