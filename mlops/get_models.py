from flask_caching import Cache
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # turn off one DNN custom operations
import keras
from keras import models

from configs.loadsettings import HuggingFaceSettings


# created a separate file to avoid circular imports
cache = Cache()

MODEL_URL = HuggingFaceSettings().MODEL_URL


def load_model(url: str = MODEL_URL) -> models.Sequential:
    cached_model = cache.get("my_model")
    if not cached_model:
        # Data not in cache, fetch it from the model
        model = keras.saving.load_model(url)
        cache.set("my_model", model, timeout=600)  # Cache for 600 seconds
    else:
        model = cached_model

    return model
