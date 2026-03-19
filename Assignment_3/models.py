from pathlib import Path

from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from rich.console import Console

console = Console()

def load_automodel(
    model_name: str,
    num_labels: int,
    verbose: bool = True,
) -> AutoModelForSequenceClassification:
    if verbose:
        console.print(f"\n [bold white] Loading model {model_name}... [/bold white]")
        
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    
    if verbose:
        print(f"Model {model_name} loaded successfully.")
        
    return model

def train_model(
    name: str,
    automodel: AutoModelForSequenceClassification,
    train_data: Dataset,
    validation_data: Dataset,
    epochs: int = 5,
    batch_size: int = 8,
    weight_decay: float = 0.01,
    learning_rate: float = 2e-5,
    verbose: bool = True,
) -> Trainer:
    if verbose:
        console.print(f"\n [bold white]Training model {name}...[/bold white]")
    
    if verbose:
        console.print(f"Setting up training arguments for {name}...")
    
    training_args = TrainingArguments(
        output_dir=Path(__file__).parent / "results" / name,
        logging_dir=Path(__file__).parent / "logs" / name,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
    )
    
    trainer = Trainer(
        model=automodel,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
    )
    
    if verbose:
        console.print(f"\n Training model {name}...")
        
    trainer.train()
    
    if verbose:
        console.print(f"Finished training model {name}.")
    
    return trainer