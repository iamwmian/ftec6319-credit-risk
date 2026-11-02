"""FTEC 6319 credit risk project, part 11: prepare features for the neural net.

A neural net needs every cell to be a number, and it treats a larger number
as a larger input. XGBoost could leave missing cells as NaN and could split
on a feature whether that feature was a 12-month sum or a single month.
This step puts the 36 kept features on a common scale.

The brief asks for three operations. There is no one-hot step to repeat:
none of the 11 categorical fields were in the random column draw, so every
kept feature is already numeric.

1. Outliers are capped. Each feature is floored at its 1st percentile and
   capped at its 99th percentile. The percentiles use the observed numbers
   only. A missing cell is not a value of zero, so it does not set the fence.
2. Each feature is standardized with StandardScaler: subtract its average
   and divide by its spread, so a typical observed value lands near 0 and
   one standard deviation lands near 1. Missing cells are skipped while the
   average and the spread are calculated, and they stay missing here.
3. Missing cells become 0. After standardization, 0 is the train average of
   that feature, so a missing cell enters the neural net as an ordinary
   value rather than as an extreme one.

Filling the gaps with zero before the cap would treat a rare observed value
as an outlier. Three kept features are missing for more than 99% of
accounts. After those gaps were written as zero, both percentiles were zero
and the cap erased the few real values.

The percentiles, the average, and the spread are calculated on the train
sample only. Test 1 and Test 2 are transformed with those same numbers.
Refitting them on a test sample would let that sample's own extremes decide
the scale the model is judged on.

The raw American Express fields were already anonymized and scaled. The
features here are summaries of those fields, and a 12-month sum is much
larger than a one-month value. Standardization puts those summaries on one
scale before the neural net.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_DIR / "samples"
RESULT_DIR = PROJECT_DIR / "results"

SELECTED_FEATURES_PATH = RESULT_DIR / "03_selected_features.csv"
TRAIN_PATH = SAMPLE_DIR / "train.csv"
TEST1_PATH = SAMPLE_DIR / "test1.csv"
TEST2_PATH = SAMPLE_DIR / "test2.csv"

NN_TRAIN_PATH = SAMPLE_DIR / "nn_train.csv"
NN_TEST1_PATH = SAMPLE_DIR / "nn_test1.csv"
NN_TEST2_PATH = SAMPLE_DIR / "nn_test2.csv"
RULES_PATH = RESULT_DIR / "09_nn_processing.csv"

# The brief's fences: floor at the 1st percentile, cap at the 99th.
LOW_PERCENTILE = 0.01
HIGH_PERCENTILE = 0.99
ID_COLUMNS = ["customer_ID", "target"]


def selected_features():
    """The 36 features kept when either importance screen was above 0.5%."""
    table = pd.read_csv(SELECTED_FEATURES_PATH)
    if not bool(table["kept"].all()):
        raise RuntimeError("03_selected_features.csv contains a feature that was not kept")
    features = table["feature"].tolist()
    if len(features) != len(set(features)):
        raise RuntimeError("03_selected_features.csv has duplicate feature names")
    return features


def load_sample(path, features):
    """Read the account id, the default flag, and the kept features."""
    frame = pd.read_csv(
        path,
        usecols=[*ID_COLUMNS, *features],
        dtype={"customer_ID": "string"},
    )
    if frame["customer_ID"].duplicated().any():
        raise RuntimeError(f"{path.name} has more than one row for an account")
    target = frame["target"].to_numpy()
    if not set(np.unique(target)).issubset({0, 1}):
        raise RuntimeError(f"{path.name} target is not a 0/1 default indicator")
    return frame


def fit_fences(train_values):
    """1st and 99th percentiles of each train column, ignoring missing cells."""
    if np.isnan(train_values).all(axis=0).any():
        raise RuntimeError("A kept feature is missing for every train account")
    low = np.nanquantile(train_values, LOW_PERCENTILE, axis=0)
    high = np.nanquantile(train_values, HIGH_PERCENTILE, axis=0)
    if np.any(low > high):
        raise RuntimeError("A feature has a 1st percentile above its 99th percentile")
    return low, high


def fill_missing(values):
    """Missing cells become 0, which is the train average after standardization."""
    filled = values.copy()
    filled[np.isnan(filled)] = 0.0
    return filled


def save_table(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def processed_frame(source, values, features):
    """Put the scaled numbers back next to the untouched id and target."""
    scaled = pd.DataFrame(values, columns=features)
    return pd.concat(
        [source[ID_COLUMNS].reset_index(drop=True), scaled],
        axis=1,
    )


def main():
    features = selected_features()
    train = load_sample(TRAIN_PATH, features)
    test1 = load_sample(TEST1_PATH, features)
    test2 = load_sample(TEST2_PATH, features)

    train_values = train[features].astype(float).to_numpy()
    test1_values = test1[features].astype(float).to_numpy()
    test2_values = test2[features].astype(float).to_numpy()
    missing_train = np.isnan(train_values).sum(axis=0)
    observed = ~np.isnan(train_values)

    # Step 1. Fences come from the observed train values. np.clip leaves a
    # missing cell missing, so the cap does not invent a zero.
    low, high = fit_fences(train_values)
    below = ((train_values < low) & observed).sum(axis=0)
    above = ((train_values > high) & observed).sum(axis=0)
    train_clipped = np.clip(train_values, low, high)
    test1_clipped = np.clip(test1_values, low, high)
    test2_clipped = np.clip(test2_values, low, high)

    # Step 2. The average and the spread ignore missing cells. StandardScaler
    # divides by the count of observed train values, not by count minus one.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_clipped)
    test1_scaled = scaler.transform(test1_clipped)
    test2_scaled = scaler.transform(test2_clipped)

    # Step 3. A missing cell becomes 0, the center of the standardized feature.
    train_scaled = fill_missing(train_scaled)
    test1_scaled = fill_missing(test1_scaled)
    test2_scaled = fill_missing(test2_scaled)

    if not np.isfinite(train_scaled).all() or not np.isfinite(test1_scaled).all() or not np.isfinite(test2_scaled).all():
        raise RuntimeError("Processed features still contain missing or infinite values")
    flat = np.isclose(train_scaled.std(axis=0), 0.0)
    if flat.any():
        names = [features[i] for i, wiped in enumerate(flat) if wiped]
        raise RuntimeError(f"Processing removed all variation from: {names}")

    rules = pd.DataFrame(
        {
            "feature": features,
            "train_missing_count": missing_train.astype(int),
            "train_missing_rate": missing_train / len(train),
            "p01": low,
            "p99": high,
            "train_below_p01": below.astype(int),
            "train_above_p99": above.astype(int),
            "train_mean": scaler.mean_,
            "train_scale": scaler.scale_,
        }
    )
    save_table(RULES_PATH, rules)
    save_table(NN_TRAIN_PATH, processed_frame(train, train_scaled, features))
    save_table(NN_TEST1_PATH, processed_frame(test1, test1_scaled, features))
    save_table(NN_TEST2_PATH, processed_frame(test2, test2_scaled, features))

    print(f"features: {len(features)}", flush=True)
    print(
        f"rows: train={len(train)} test1={len(test1)} test2={len(test2)}",
        flush=True,
    )
    print(
        f"train cells filled with 0: {int(missing_train.sum())} "
        f"of {len(train) * len(features)}",
        flush=True,
    )
    print(
        f"train values floored: {int(below.sum())}  capped: {int(above.sum())}",
        flush=True,
    )
    print(f"saved {NN_TRAIN_PATH}", flush=True)
    print(f"saved {NN_TEST1_PATH}", flush=True)
    print(f"saved {NN_TEST2_PATH}", flush=True)
    print(f"saved {RULES_PATH}", flush=True)


if __name__ == "__main__":
    main()
