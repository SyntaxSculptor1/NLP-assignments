import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, GlobalMaxPool1D, Conv1D, Input, Embedding, TextVectorization, Bidirectional, LSTM
from keras.callbacks import EarlyStopping
from rich.console import Console

console = Console()

def build_cnn(
    vectorizer: TextVectorization,
    num_classes: int = 4,
    filters: int = 128,
    kernel_size: int = 5,
    activation: str = "relu",
    embedding_size: int = 100,
    dense_size: int = 64,
    vocab_size: int = 10000,
    padding_length: int = 100,
    dropout: float = 0.0,
):
    """
    Builds a Convolutional Neural Network (CNN) model for text classification.

    Parameters:
        vectorizer (TextVectorization): The vectorizer to use.
        num_classes (int): The number of classes to predict.
        filters (int): The number of filters for the Conv1D layer.
        kernel_size (int): The kernel size for the Conv1D layer.
        activation (str): The activation function for the Conv1D and Dense layers.
        embedding_size (int): The size of the embedding layer.
        dense_size (int): The size of the Dense layer after the Conv1D layer.
        vocab_size (int): The size of the vocabulary for the Embedding layer.
        padding_length (int): The length to which text sequences are padded.
        dropout (float): The dropout rate for the Dropout layer.

    Returns:
        Sequential: The CNN model.
    """
    return Sequential([
        Input(shape=(), dtype=tf.string),
        vectorizer,
        Embedding(vocab_size, embedding_size),
        Conv1D(filters=filters, kernel_size=kernel_size, activation=activation),
        GlobalMaxPool1D(),
        Dense(dense_size, activation=activation),
        Dropout(dropout),
        Dense(num_classes, activation="softmax")
    ])
    

def training_cnn(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    vectorizer: TextVectorization,
    num_classes: int = 4,
    filters: int = 128,
    kernel_size: int = 5,
    activation: str = "relu",
    embedding_size: int = 100,
    dense_size: int = 64,
    vocab_size: int = 10000,
    padding_length: int = 100,
    epochs: int = 100,
    optimizer: str = "adam",
    batch_size: int = 32,
    loss: str = "categorical_crossentropy",
    metrics: list = ["accuracy"],
    patience: int = 3,
    dropout: float = 0.0,
    verbose: bool = True
) -> Sequential:
    """
    Trains a Convolutional Neural Network (CNN) model for text classification.

    Parameters:
        model_name (str): The name of the model.
        x_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
        x_val (np.ndarray): The validation features.
        y_val (np.ndarray): The validation labels.
        vectorizer (TextVectorization): The vectorizer to use.
        num_classes (int): The number of classes to predict.
        filters (int): The number of filters for the Conv1D layer.
        kernel_size (int): The kernel size for the Conv1D layer.
        activation (str): The activation function for the Conv1D and Dense layers.
        embedding_size (int): The size of the embedding layer.
        dense_size (int): The size of the Dense layer after the Conv1D layer.
        vocab_size (int): The size of the vocabulary for the Embedding layer.
        padding_length (int): The length to which text sequences are padded.
        epochs (int): The number of epochs to train the model.
        optimizer (str): The optimizer to use.
        batch_size (int): The batch size for training.
        loss (str): The loss function to use.
        metrics (list): The metrics to use for evaluation.
        patience (int): The patience for the Early Stopping callback.
        dropout (float): The dropout rate for the Dropout layer.
        verbose (bool): Whether to print verbose output.

    Returns:
        Sequential: The trained CNN model.
    """
    if verbose:
        console.print(f"\n [bold white] Training model: {model_name} [/bold white]")

    model = build_cnn(
        vectorizer=vectorizer,
        num_classes=num_classes,
        filters=filters,
        kernel_size=kernel_size,
        activation=activation,
        embedding_size=embedding_size,
        dense_size=dense_size,
        vocab_size=vocab_size,
        padding_length=padding_length,
        dropout=dropout
    )

    callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[callback],
        verbose=int(verbose)
    )

    if verbose:
        history = pd.DataFrame(history.history)[["loss", "val_loss"]]
        plt.clf()
        history.plot(figsize=(8, 5))
        plt.title(f"Loss curve - {model_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.savefig(
            Path(__file__).parent / "plots" / f"{model_name}_loss.png"
        )

    return model

def build_lstm(
    vectorizer: TextVectorization,
    num_classes: int = 4,
    embedding_size: int = 100,
    lstm_size: int = 64,
    dense_size: int = 64,
    vocab_size: int = 10000,
    padding_length: int = 100,
    dropout: float = 0.0,
) -> Sequential:
    return Sequential([
        Input(shape=(), dtype=tf.string),
        vectorizer,
        Embedding(vocab_size, embedding_size),
        Bidirectional(LSTM(lstm_size, return_sequences=False)),
        Dense(dense_size, activation="relu"),
        Dropout(dropout),
        Dense(num_classes, activation="softmax")
    ])


def training_lstm(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    vectorizer: TextVectorization,
    num_classes: int = 4,
    embedding_size: int = 100,
    lstm_size: int = 64,
    dense_size: int = 64,
    vocab_size: int = 10000,
    padding_length: int = 100,
    epochs: int = 100,
    optimizer: str = "adam",
    batch_size: int = 32,
    loss: str = "categorical_crossentropy",
    metrics: list = ["accuracy"],
    patience: int = 3,
    dropout: float = 0.0,
    verbose: bool = True,
    plot: bool = True
) -> Sequential:
    """
    Train a Long Short-Term Memory (LSTM) model for text classification.

    Parameters:
        model_name (str): The name of the model.
        x_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
        x_val (np.ndarray): The validation features.
        y_val (np.ndarray): The validation labels.
        vectorizer (TextVectorization): The vectorizer to use.
        num_classes (int, optional): The number of classes to predict. Defaults to 4.
        embedding_size (int, optional): The size of the embedding layer. Defaults to 100.
        lstm_size (int, optional): The size of the LSTM layer. Defaults to 64.
        dense_size (int, optional): The size of the Dense layer after the LSTM layer. Defaults to 64.
        vocab_size (int, optional): The size of the vocabulary for the Embedding layer. Defaults to 10000.
        padding_length (int, optional): The length to which text sequences are padded. Defaults to 100.
        epochs (int, optional): The number of epochs to train the model. Defaults to 100.
        optimizer (str, optional): The optimizer to use. Defaults to "adam".
        batch_size (int, optional): The batch size for training. Defaults to 32.
        loss (str, optional): The loss function to use. Defaults to "categorical_crossentropy".
        metrics (list, optional): The metrics to use for evaluation. Defaults to ["accuracy"].
        patience (int, optional): The patience for the Early Stopping callback. Defaults to 3.
        dropout (float, optional): The dropout rate for the Dropout layer. Defaults to 0.0.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        plot (bool, optional): Whether to plot the loss curve. Defaults to True.

    Returns:
        Sequential: The trained model.
    """
    if verbose:
        console.print(f"\n [bold white] Training model: {model_name} [/bold white]")

    model = build_lstm(
        vectorizer=vectorizer,
        num_classes=num_classes,
        embedding_size=embedding_size,
        lstm_size=lstm_size,
        dense_size=dense_size,
        vocab_size=vocab_size,
        padding_length=padding_length,
        dropout=dropout
    )

    callback = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True
    )

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[callback],
        verbose=int(verbose)
    )

    if verbose:
        history = pd.DataFrame(history.history)[["loss", "val_loss"]]
        plt.clf()
        history.plot(figsize=(8, 5))
        plt.title(f"Loss curve - {model_name}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.gca().set_ylim(0, 1)
        plt.savefig(
            Path(__file__).parent / "plots" / f"{model_name}_loss.png"
        )

    return model
    