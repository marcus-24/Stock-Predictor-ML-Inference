import mlflow
import pandas as pd
import yfinance as yf
import tensorflow as tf
import keras
import os
import warnings

warnings.filterwarnings("ignore", message="You are saving your model as an HDF5 file.")

from models import build_bidirec_lstm_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # needed to suppress out of rand warnings

DATA_COLS = ["Open", "High", "Low", "Close"]
BATCH_SIZE = 32
N_PAST = 10
N_FUTURE = 10
N_SHIFT = 1
EPOCHS = 1000
TRAIN_PERCENT = 0.75
N_FEATURES = len(DATA_COLS)


def sequential_window_dataset(
    df: pd.DataFrame, batch_size: int, n_past: int, n_future: int, shift: int
) -> tf.data.Dataset:

    return (
        tf.data.Dataset.from_tensor_slices(
            df.values
        )  # transform array to tensor dataset type
        .window(
            size=n_past + n_future, shift=shift, drop_remainder=True
        )  # window features
        .flat_map(lambda w: w.batch(n_past + n_future))  #
        .map(
            lambda w: (w[:n_past], w[n_past:])
        )  # split into features and labels (window past )
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )  # create batch and prefetch next batch while processing current batch


if __name__ == "__main__":

    df = yf.Ticker("AAPL").history(interval="1d", start="2016-01-01", end="2024-01-30")

    """Split Data"""
    split_idx = int(TRAIN_PERCENT * df.shape[0])
    split_time = df.index[split_idx]

    x_train = df.loc[:split_time, DATA_COLS]
    train_time = x_train.index.to_numpy()
    x_val = df.loc[split_time:, DATA_COLS]
    val_time = x_val.index.to_numpy()

    train_set = sequential_window_dataset(
        x_train, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=N_SHIFT
    )
    val_set = sequential_window_dataset(
        x_val, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=N_SHIFT
    )

    """Create model"""
    model = build_bidirec_lstm_model(
        train_set, n_past=N_PAST, n_features=N_FEATURES, batch_size=BATCH_SIZE
    )

    """Setup MLFlow Tracking settings"""
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("/mlflow-tf-stock-forecast")
    # mlflow.tensorflow.autolog(every_n_iter=100, log_models=True, checkpoints=True)
    mlflow.tensorflow.autolog(checkpoint_save_best_only=True)

    """Train model"""
    early_stopping = keras.callbacks.EarlyStopping(patience=10)

    history = model.fit(
        train_set,
        epochs=EPOCHS,
        validation_data=val_set,
        callbacks=[early_stopping],
    )
