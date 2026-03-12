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
):
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
    