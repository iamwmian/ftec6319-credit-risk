"""FTEC 6319 credit risk project, part 12: neural-net grid search.

Part 11 put the 36 kept features on one scale. This step tries every
combination of the settings named in the brief and records the AUC on the
train sample, Test 1, and Test 2. Each net is fit on train only. Test 1 and
Test 2 are scored afterward, so they do not teach the model.

The grid has 2 x 2 x 2 x 2 x 2 = 32 models:

- hidden layers: 2, 4
- nodes in each hidden layer: 4, 6
- activation in the hidden layers: ReLU, tanh
- dropout on the hidden layers: 50%, or 100% (every node kept)
- batch size: 100, 10000

A hidden layer is a set of nodes sitting between the 36 features and the
default probability. More layers and more nodes give the net more room to
fit patterns. ReLU and tanh are the two ways a node turns its inputs into
a number. Dropout at 50% ignores a random half of the hidden nodes on each
training update, which keeps the net from relying on one node. Dropout at
100% leaves every node in place. The batch size is how many accounts are
shown before the weights are updated.

The optimizer is Adam, the loss is cross-entropy, and every net trains for
20 epochs. Those three are fixed, so they do not multiply the grid. The
output is one probability of default. Cross-entropy for a yes/no target is
binary cross-entropy, and the output node uses a sigmoid so that number
stays between 0 and 1.

The brief asks for a separate loop for each layer count, because a 2-layer
net and a 4-layer net are different shapes. The results table is rewritten
after every model. A stopped run can continue from the rows already saved.
Seed 42 fixes the random dropout and the order of the training batches.
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_DIR / "samples"
RESULT_DIR = PROJECT_DIR / "results"

TRAIN_PATH = SAMPLE_DIR / "nn_train.csv"
TEST1_PATH = SAMPLE_DIR / "nn_test1.csv"
TEST2_PATH = SAMPLE_DIR / "nn_test2.csv"
GRID_PATH = RESULT_DIR / "10_nn_grid_search.csv"

SEED = 42
EPOCHS = 20

# A 2-layer loop and a 4-layer loop. The other four settings sit inside both.
LAYER_COUNTS = (2, 4)
NODE_COUNTS = (4, 6)
ACTIVATIONS = ("relu", "tanh")
# 100 means no dropout layer. 50 means each hidden node is dropped half the time.
DROPOUT_PCTS = (50, 100)
BATCH_SIZES = (100, 10000)

GRID_COLUMNS = [
    "n_hidden_layers",
    "n_nodes",
    "activation",
    "dropout_pct",
    "batch_size",
    "auc_train",
    "auc_test1",
    "auc_test2",
]


def set_seed():
    """Reset the random draws so a resumed combo rebuilds the same net."""
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    keras.utils.set_random_seed(SEED)


def load_sample(path):
    """Read one processed sample. The id is not an input to the net."""
    frame = pd.read_csv(path)
    if "target" not in frame.columns:
        raise RuntimeError(f"{path.name} has no target column")
    y = frame["target"].to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise RuntimeError(f"{path.name} target is not a 0/1 default indicator")
    X = frame.drop(columns=["target", "customer_ID"]).to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise RuntimeError(f"{path.name} still has missing or infinite feature values")
    return X, y


def combo_key(n_layers, n_nodes, activation, dropout_pct, batch_size):
    """Identity of one grid cell, used to skip work already saved."""
    return (
        int(n_layers),
        int(n_nodes),
        str(activation),
        int(dropout_pct),
        int(batch_size),
    )


def load_completed():
    """Return the saved grid, or an empty table if this is the first run."""
    if not GRID_PATH.exists():
        return pd.DataFrame(columns=GRID_COLUMNS)
    completed = pd.read_csv(GRID_PATH)
    missing = [column for column in GRID_COLUMNS if column not in completed.columns]
    if missing:
        raise RuntimeError(f"Existing grid file is missing columns: {missing}")
    return completed[GRID_COLUMNS]


def completed_keys(completed):
    keys = set()
    for row in completed.itertuples(index=False):
        keys.add(
            combo_key(
                row.n_hidden_layers,
                row.n_nodes,
                row.activation,
                row.dropout_pct,
                row.batch_size,
            )
        )
    return keys


def build_model(n_features, n_layers, n_nodes, activation, dropout_pct):
    """Stack the hidden layers, then one node that outputs a probability."""
    network = keras.Sequential()
    network.add(layers.Input(shape=(n_features,)))
    for _ in range(n_layers):
        network.add(layers.Dense(n_nodes, activation=activation))
        # 100% in the brief means every node is kept, so no dropout layer.
        if dropout_pct == 50:
            network.add(layers.Dropout(0.5, seed=SEED))
        elif dropout_pct != 100:
            raise RuntimeError(f"Dropout percent must be 50 or 100, got {dropout_pct}")
    network.add(layers.Dense(1, activation="sigmoid"))
    network.compile(optimizer="adam", loss="binary_crossentropy")
    return network


def auc(model, X, y, batch_size):
    """Ranking accuracy of the predicted default probability."""
    probability = model.predict(X, batch_size=batch_size, verbose=0).ravel()
    return float(roc_auc_score(y, probability))


def fit_one(X_train, y_train, X_test1, y_test1, X_test2, y_test2, settings):
    """Fit one combination on train and score all three samples."""
    n_layers, n_nodes, activation, dropout_pct, batch_size = settings
    set_seed()
    model = build_model(
        n_features=X_train.shape[1],
        n_layers=n_layers,
        n_nodes=n_nodes,
        activation=activation,
        dropout_pct=dropout_pct,
    )
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    row = {
        "n_hidden_layers": n_layers,
        "n_nodes": n_nodes,
        "activation": activation,
        "dropout_pct": dropout_pct,
        "batch_size": batch_size,
        "auc_train": auc(model, X_train, y_train, batch_size),
        "auc_test1": auc(model, X_test1, y_test1, batch_size),
        "auc_test2": auc(model, X_test2, y_test2, batch_size),
    }
    del model
    keras.backend.clear_session()
    return row


def save_grid(completed):
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = GRID_PATH.with_name(GRID_PATH.name + ".tmp")
    completed.to_csv(temporary, index=False)
    temporary.replace(GRID_PATH)


def setting_grid(n_layers):
    """The 16 combinations for one layer count."""
    return [
        (n_layers, n_nodes, activation, dropout_pct, batch_size)
        for n_nodes in NODE_COUNTS
        for activation in ACTIVATIONS
        for dropout_pct in DROPOUT_PCTS
        for batch_size in BATCH_SIZES
    ]


def run_layer_count(n_layers, completed, done, samples, start, total):
    """Train every setting that uses this many hidden layers."""
    X_train, y_train, X_test1, y_test1, X_test2, y_test2 = samples
    grid = setting_grid(n_layers)
    for offset, settings in enumerate(grid):
        number = start + offset
        if combo_key(*settings) in done:
            continue
        row = fit_one(
            X_train, y_train, X_test1, y_test1, X_test2, y_test2, settings
        )
        completed = pd.concat([completed, pd.DataFrame([row])], ignore_index=True)
        save_grid(completed)
        done.add(combo_key(*settings))
        print(
            f"{number}/{total} layers={row['n_hidden_layers']} "
            f"nodes={row['n_nodes']} activation={row['activation']} "
            f"dropout={row['dropout_pct']}% batch={row['batch_size']} "
            f"AUC train={row['auc_train']:.4f} "
            f"test1={row['auc_test1']:.4f} test2={row['auc_test2']:.4f}",
            flush=True,
        )
    return completed, done


def main():
    X_train, y_train = load_sample(TRAIN_PATH)
    X_test1, y_test1 = load_sample(TEST1_PATH)
    X_test2, y_test2 = load_sample(TEST2_PATH)
    if X_train.shape[1] != X_test1.shape[1] or X_train.shape[1] != X_test2.shape[1]:
        raise RuntimeError("The three samples do not have the same features")
    print(
        f"features: {X_train.shape[1]} | train: {len(X_train):,} | "
        f"test1: {len(X_test1):,} | test2: {len(X_test2):,}",
        flush=True,
    )

    completed = load_completed()
    done = completed_keys(completed)
    total = sum(len(setting_grid(n_layers)) for n_layers in LAYER_COUNTS)
    print(f"combinations: {total} | already saved: {len(done)}", flush=True)
    samples = (X_train, y_train, X_test1, y_test1, X_test2, y_test2)

    # Separate loops: the 2-layer nets and the 4-layer nets are different shapes.
    start = 1
    for n_layers in LAYER_COUNTS:
        completed, done = run_layer_count(
            n_layers, completed, done, samples, start, total
        )
        start += len(setting_grid(n_layers))

    if len(completed) != total:
        raise RuntimeError(f"Grid has {len(completed)} rows, expected {total}")
    print(f"saved {GRID_PATH}", flush=True)


if __name__ == "__main__":
    main()
