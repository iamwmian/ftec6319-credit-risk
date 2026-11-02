# 2 x 2 x 2 x 2 x 2 = 32 models. The layer counts use separate loops.
# Dropout of 100% means every node is kept. Dropout of 50% drops half the nodes.
for n_layers in (2, 4):
    for n_nodes in (4, 6):
        for activation in ("relu", "tanh"):
            for dropout_pct in (50, 100):
                for batch_size in (100, 10000):
                    network = keras.Sequential()
                    network.add(keras.Input(shape=(36,)))
                    for _ in range(n_layers):
                        network.add(keras.layers.Dense(n_nodes, activation=activation))
                        if dropout_pct == 50:
                            network.add(keras.layers.Dropout(0.5, seed=42))
                    network.add(keras.layers.Dense(1, activation="sigmoid"))
                    network.compile(optimizer="adam", loss="binary_crossentropy")
                    network.fit(X_train, y_train, epochs=20, batch_size=batch_size, shuffle=True)
                    # AUC is recorded on train, Test 1, and Test 2.
