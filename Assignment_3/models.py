from pathlib import Path

from datasets import Dataset
from rich.console import Console
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

console = Console()

def load_automodel(
    model_name: str,
    num_labels: int,
    verbose: bool = True,
) -> AutoModelForSequenceClassification:
    """
    Load a pre-trained model from the Hugging Face model hub.

    Args:
        model_name (str): The name of the model to load.
        num_labels (int): The number of labels for the model.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        AutoModelForSequenceClassification: The loaded model.
    """
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
    patience: int = 3,
    save: bool = True,
    verbose: bool = True,
) -> Trainer:
    """
    Train a model using the Trainer API.

    Args:
        name (str): The name of the model.
        automodel (AutoModelForSequenceClassification): The model to train.
        train_data (Dataset): The training data.
        validation_data (Dataset): The validation data.
        epochs (int, optional): The number of epochs to train for. Defaults to 5.
        batch_size (int, optional): The batch size to use for training. Defaults to 8.
        weight_decay (float, optional): The weight decay to use for training. Defaults to 0.01.
        learning_rate (float, optional): The learning rate to use for training. Defaults to 2e-5.
        patience (int, optional): The patience for early stopping. Defaults to 3.
        save (bool, optional): Whether to save the model. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.

    Returns:
        Trainer: The trained model.
    """
    if verbose:
        console.print(f"\n [bold white]Training model {name}...[/bold white]")

    if verbose:
        console.print(f"Setting up training arguments for {name}...")

    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=patience)

    training_args = TrainingArguments(
        output_dir=Path(__file__).parent / "results" / name,
        logging_dir=Path(__file__).parent / "logs" / name,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=automodel,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        callbacks=[early_stopping_callback]
    )

    if verbose:
        console.print(f"\n Training model {name}...")

    trainer.train()

    if save:
        trainer.save_model(Path(__file__).parent / "trained_models" / name)

    if verbose:
        console.print(f"Finished training model {name}.")

    return trainer
