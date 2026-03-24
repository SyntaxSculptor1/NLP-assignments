import html
import re
from typing import List

from datasets import Dataset
from rich.console import Console
from transformers import AutoTokenizer

console = Console()

def load_tokenizer(
    model_name: str,
    verbose: bool = True,
) -> AutoTokenizer:
    """
    Load a tokenizer from the Hugging Face model hub.

    Args:
        model_name (str): The name of the model to load.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        AutoTokenizer: The loaded tokenizer.
    """
    if verbose:
        console.print(f"\n [bold white] Loading tokenizer for model - {model_name}: [/bold white]")
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if verbose:
        console.print(f"Tokenizer loaded for model - {model_name}")
    
    return tokenizer

def pre_tokenization_normalization_helper(text: str) -> str:
    """
    Pre-tokenization normalization helper. By default, this function normalizes the text by removing HTML tags,
    and removing extra spaces.

    Args:
        text (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    text = html.unescape(text)
    clean_text = re.sub(r"<[^>]+>", "", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    return clean_text

    
def tokenize_datasets(
    name: str,
    datasets: List[Dataset],
    tokenizer: AutoTokenizer,
    text_column: str = "text",
    padding_type: str = "max_length",
    truncation: bool = True,
    verbose: bool = True
) -> List[Dataset]:
    """
    Tokenize the datasets.

    Args:
        name (str): The name of the datasets.
        datasets (List[Dataset]): The datasets to tokenize.
        tokenizer (AutoTokenizer): The tokenizer to use.
        text_column (str, optional): The column containing the text. Defaults to "text".
        padding_type (str, optional): The type of padding to use. Defaults to "max_length".
        truncation (bool, optional): Whether to truncate the text. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        List[Dataset]: The tokenized datasets.
    """
    
    if verbose:
        console.print(f"\n [bold white] Tokenizing datasets for {name}: [/bold white]")
    
    def pre_tokenizer_normalization_function(batch) -> dict:
        return {text_column: [pre_tokenization_normalization_helper(text) for text in batch[text_column]]}
        
    tokenized_datasets = [dataset.map(pre_tokenizer_normalization_function, batched=True) for dataset in datasets]
    
    def tokenize_function(batch) -> dict:
        return tokenizer(batch[text_column], padding=padding_type, truncation=truncation)
    
    tokenized_datasets = [dataset.map(tokenize_function, batched=True) for dataset in datasets]
    
    if verbose:
        console.print(f"Finished tokenizing datasets for {name}.")
            
    return tokenized_datasets