import mlflow
import numpy as np
import yfinance as yf
import tensorflow as tf
import keras
from sklearn.preprocessing import StandardScaler

from models import build_bidirec_lstm_model

DATA_COLS = ["Open", "High", "Low", "Close"]
BATCH_SIZE = 32
N_PAST = 24
N_FUTURE = 24
N_SHIFT = 1
EPOCHS = 1000
N_FEATURES = len(DATA_COLS)

def sequential_window_dataset(series: np.ndarray, 
                              batch_size: int,
                              n_past: int,
                              n_future:int,
                              shift: int) -> tf.data.Dataset:
    """_summary_

    Args:
        series (np.ndarray): _description_
        batch_size (int): _description_
        n_past (int): _description_
        n_future (int): _description_
        shift (int): _description_

    Returns:
        tf.data.Dataset: _description_
    """
    ds = tf.data.Dataset.from_tensor_slices(series)  # transform array to tensor dataset type
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)  # window features (add one for future label)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))  # creates batches for features
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))  # split into features and labels (window past )
    return ds.batch(batch_size).prefetch(1)  # create batch and prefetch next batch while processing current batch




if __name__ == '__main__':
    # os.system('mlflow server --host 127.0.0.1 --port 8080')
    
    df = (yf.Ticker('AAPL')
            .history(interval='1d', 
                     start="2020-01-01", 
                     end="2024-01-30"))

    '''Split Data'''
    split_idx = int(0.75 * df.shape[0])
    split_time = df.index[split_idx]

    x_train = df.loc[:split_time, DATA_COLS]
    train_time = x_train.index.to_numpy()
    x_val = df.loc[split_time:, DATA_COLS]
    val_time = x_val.index.to_numpy()

    '''Normalize data'''
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)

    train_set = sequential_window_dataset(x_train, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=N_SHIFT)
    val_set = sequential_window_dataset(x_val, batch_size=BATCH_SIZE, n_past=N_PAST, n_future=N_FUTURE, shift=N_SHIFT)

    '''Create model'''
    model = build_bidirec_lstm_model(n_past=N_PAST,
                                     n_features=N_FEATURES,
                                     batch_size=BATCH_SIZE)
    
    '''Setup MLFlow Tracking settings'''
    mlflow.set_tracking_uri("http://127.0.0.1:8080")
    mlflow.set_experiment("/mlflow-tf-stock-forecast")
    # mlflow.tensorflow.autolog(every_n_iter=100, log_models=True, checkpoints=True)
    mlflow.tensorflow.autolog()


    '''Train model'''
    early_stopping = keras.callbacks.EarlyStopping(patience=10)

    history = model.fit(train_set, 
                    epochs=EPOCHS, 
                    validation_data=val_set,
                    callbacks=[early_stopping])

    
    
    