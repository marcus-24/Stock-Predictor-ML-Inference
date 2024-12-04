from keras import layers, models, optimizers, losses  # TODO: figure out pylance errors


def build_bidirec_lstm_model(n_past: int,
                n_features: int,
                batch_size: int) -> models.Sequential:
	model = models.Sequential([
            layers.InputLayer(input_shape=(n_past, n_features),
                                batch_size=batch_size),
            layers.Bidirectional(layers.LSTM(20, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(20, return_sequences=True)),
            layers.Dense(n_features)
    ])

	optimizer = optimizers.SGD(learning_rate=0.001, momentum=0.9)
	model.compile(loss=losses.Huber(),
				  optimizer=optimizer,
				  metrics=["mae"])
	model.summary()

	return model