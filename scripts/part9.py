"""FTEC 6319 credit risk project, part 9: XGBoost grid search.

Part 8 kept 36 features. This step tries every combination of the settings
named in the brief and records the AUC on the train sample, Test 1, and
Test 2. The models are fit on train only. Test 1 and Test 2 are scored
afterward, so they do not teach the model.

The grid has 3 x 2 x 2 x 2 x 3 = 72 models:

- trees: 50, 100, 300
- learning rate: 0.01, 0.1
- share of accounts used in each tree: 50%, 80%
- share of features used in each tree: 50%, 100%
- weight of a defaulted account: 1, 5, 10

AUC is the chance that a defaulted account gets a higher predicted
probability than a non-defaulted account. A value of 0.50 is no better
than chance. A value of 1 is a perfect ranking.

The results table is rewritten after every model. A stopped run can
continue from the combinations already in that file.

Tree depth is not in this grid, so it stays at XGBoost's default. Seed 42
fixes the random row and feature draws inside each tree.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_DIR / "samples"
RESULT_DIR = PROJECT_DIR / "results"

TRAIN_PATH = SAMPLE_DIR / "train.csv"
TEST1_PATH = SAMPLE_DIR / "test1.csv"
TEST2_PATH = SAMPLE_DIR / "test2.csv"
SELECTED_FEATURES_PATH = RESULT_DIR / "03_selected_features.csv"
GRID_PATH = RESULT_DIR / "04_xgb_grid_search.csv"

SEED = 42

# Order matches the brief: trees, learning rate, row share, feature share, weight.
TREE_COUNTS = (50, 100, 300)
LEARNING_RATES = (0.01, 0.1)
# Stored in the table as percents. The model receives the fraction.
ROW_SHARES = (50, 80)
FEATURE_SHARES = (50, 100)
DEFAULT_WEIGHTS = (1, 5, 10)

GRID_COLUMNS = [
    "n_trees",
    "learning_rate",
    "subsample_pct",
    "feature_pct",
    "weight_of_default",
    "auc_train",
    "auc_test1",
    "auc_test2",
]


def selected_features():
    """Column names kept in part 8."""
    selected = pd.read_csv(SELECTED_FEATURES_PATH)
    features = selected["feature"].tolist()
    if len(features) != len(set(features)):
        raise RuntimeError("03_selected_features.csv has duplicate feature names")
    if not features:
        raise RuntimeError("03_selected_features.csv has no features")
    return features


def load_sample(path, features):
    """Read one sample and keep the target plus the 36 selected features."""
    frame = pd.read_csv(path, usecols=["target", *features])
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise RuntimeError(f"{path.name} is missing features: {missing}")
    y = frame["target"].to_numpy()
    if not set(y).issubset({0, 1}):
        raise RuntimeError(f"{path.name} target is not a 0/1 default indicator")
    return frame[features], y


def combo_key(n_trees, learning_rate, row_share, feature_share, weight):
    """Identity of one grid cell, used to skip work already saved."""
    return (
        int(n_trees),
        round(float(learning_rate), 2),
        int(row_share),
        int(feature_share),
        int(weight),
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
                row.n_trees,
                row.learning_rate,
                row.subsample_pct,
                row.feature_pct,
                row.weight_of_default,
            )
        )
    return keys


def auc(model, X, y):
    """Ranking accuracy of the predicted default probability."""
    probability = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, probability))


def fit_one(X_train, y_train, X_test1, y_test1, X_test2, y_test2, settings):
    """Fit one combination on train and score all three samples."""
    n_trees, learning_rate, row_share, feature_share, weight = settings
    model = XGBClassifier(
        n_estimators=n_trees,
        learning_rate=learning_rate,
        # 50% of accounts -> subsample 0.5. 80% -> 0.8.
        subsample=row_share / 100,
        # 50% of features in each tree, or every feature when the share is 100%.
        colsample_bytree=feature_share / 100,
        # A defaulted account counts this many times a non-defaulted account.
        scale_pos_weight=weight,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return {
        "n_trees": n_trees,
        "learning_rate": learning_rate,
        "subsample_pct": row_share,
        "feature_pct": feature_share,
        "weight_of_default": weight,
        "auc_train": auc(model, X_train, y_train),
        "auc_test1": auc(model, X_test1, y_test1),
        "auc_test2": auc(model, X_test2, y_test2),
    }


def save_grid(completed):
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = GRID_PATH.with_name(GRID_PATH.name + ".tmp")
    completed.to_csv(temporary, index=False)
    temporary.replace(GRID_PATH)


def main():
    features = selected_features()
    X_train, y_train = load_sample(TRAIN_PATH, features)
    X_test1, y_test1 = load_sample(TEST1_PATH, features)
    X_test2, y_test2 = load_sample(TEST2_PATH, features)
    print(
        f"features: {len(features)} | train: {len(X_train):,} | "
        f"test1: {len(X_test1):,} | test2: {len(X_test2):,}",
        flush=True,
    )

    completed = load_completed()
    done = completed_keys(completed)
    grid = [
        (n_trees, learning_rate, row_share, feature_share, weight)
        for n_trees in TREE_COUNTS
        for learning_rate in LEARNING_RATES
        for row_share in ROW_SHARES
        for feature_share in FEATURE_SHARES
        for weight in DEFAULT_WEIGHTS
    ]
    print(f"combinations: {len(grid)} | already saved: {len(done)}", flush=True)

    for number, settings in enumerate(grid, start=1):
        if combo_key(*settings) in done:
            continue
        row = fit_one(
            X_train, y_train, X_test1, y_test1, X_test2, y_test2, settings
        )
        completed = pd.concat([completed, pd.DataFrame([row])], ignore_index=True)
        save_grid(completed)
        done.add(combo_key(*settings))
        print(
            f"{number}/{len(grid)} trees={row['n_trees']} "
            f"lr={row['learning_rate']} rows={row['subsample_pct']}% "
            f"features={row['feature_pct']}% weight={row['weight_of_default']} "
            f"AUC train={row['auc_train']:.4f} "
            f"test1={row['auc_test1']:.4f} test2={row['auc_test2']:.4f}",
            flush=True,
        )

    if len(completed) != len(grid):
        raise RuntimeError(
            f"Grid has {len(completed)} rows, expected {len(grid)}"
        )
    print(f"saved {GRID_PATH}", flush=True)


if __name__ == "__main__":
    main()
