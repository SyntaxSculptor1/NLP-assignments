import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, GlobalMaxPool1D, Conv1D, Input, Embedding, TextVectorization, Bidirectional, LSTM
from keras.callbacks import EarlyStopping

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
    

def perform_cnn(
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

    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[callback],
        verbose=int(verbose)
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
        Bidirectional(LSTM(lstm_size, return_sequences=True)),
        Dense(dense_size, activation="relu"),
        Dropout(dropout),
        Dense(num_classes, activation="softmax")
    ])


def perform_lstm(
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
    verbose: bool = True
) -> Sequential:
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

    model.fit(
        x=x_train,
        y=y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_val, y_val),
        callbacks=[callback],
        verbose=int(verbose)
    )

    return model
    