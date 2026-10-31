"""FTEC 6319 credit risk project, part 8: keep the predictive features.

The train sample has 374 summaries of the 17 raw features. Most of those
summaries repeat the same history. This step fits two XGBoost models on the
train sample and keeps a feature when either model gives it more than 0.5%
of the total importance. Later models use only that shorter list.

Importance here is XGBoost's gain: how much the feature reduces the loss
when the trees split on it. The shares add up to 100%. A feature passes
when its share is above 0.5%, which is a share above 0.005.

Missing values stay as NaN. XGBoost sends missing values down the split
direction it learned for them, so they are not filled in with zero.

The first model uses XGBoost's own default tree settings. The second model
uses the settings named in the brief: 300 trees, learning rate 0.5, depth 4,
half the rows and half the columns in each tree, and a weight of 5 on
defaulted accounts. Seed 42 makes both fits rebuild the same importances.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_DIR / "samples"
RESULT_DIR = PROJECT_DIR / "results"
TRAIN_PATH = SAMPLE_DIR / "train.csv"

DEFAULT_IMPORTANCE_PATH = RESULT_DIR / "01_xgb_default_importance.csv"
SPECIFIED_IMPORTANCE_PATH = RESULT_DIR / "02_xgb_specified_importance.csv"
SELECTED_FEATURES_PATH = RESULT_DIR / "03_selected_features.csv"

SEED = 42
# The brief says "higher than 0.5%". Importance shares sum to 1, so the
# cutoff is 0.5 percentage points, not 50%.
IMPORTANCE_CUTOFF = 0.005

# These columns identify the account or describe the window. They are not
# candidate predictors.
DROP_COLUMNS = ["customer_ID", "sample", "n_statements", "latest_statement"]


def load_train_matrix():
    """Return the train predictors and the default flag.

    Empty cells in the CSV are read as NaN and left that way.
    """
    train = pd.read_csv(
        TRAIN_PATH,
        dtype={"customer_ID": "string", "latest_statement": "string"},
    )
    if "target" not in train.columns:
        raise RuntimeError("train.csv has no target column")

    y = train["target"].to_numpy()
    if not set(np.unique(y)).issubset({0, 1}):
        raise RuntimeError("target is not a 0/1 default indicator")

    X = train.drop(columns=["target", *DROP_COLUMNS])
    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    if non_numeric:
        raise RuntimeError(f"Non-numeric columns were left in the model: {non_numeric}")
    if X.isna().sum().sum() == 0:
        raise RuntimeError("Every model cell is filled. Missing values should remain NaN.")
    return X, y


def importance_table(model, feature_names):
    """One row per feature. importance sums to 1 across the table."""
    shares = np.asarray(model.feature_importances_, dtype=float)
    if shares.shape != (len(feature_names),):
        raise RuntimeError("Feature importance length does not match the predictors")
    total = shares.sum()
    if not np.isclose(total, 1.0, atol=1e-5):
        raise RuntimeError(f"Feature importances sum to {total}, not 1")

    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": shares,
            # Shown as a percent so the 0.5% rule can be read off the file.
            "importance_pct": shares * 100,
        }
    )
    table["above_cutoff"] = table["importance"] > IMPORTANCE_CUTOFF
    return table.sort_values("importance", ascending=False).reset_index(drop=True)


def fit_default_model(X, y):
    """First screen: XGBoost defaults, with the seed fixed."""
    model = XGBClassifier(
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def fit_specified_model(X, y):
    """Second screen: the tree settings and default weight from the brief.

    subsample=0.5 uses a random half of the accounts in each tree.
    colsample_bytree=0.5 uses a random half of the features in each tree.
    scale_pos_weight=5 makes a defaulted account count five times as much
    as a non-defaulted account. The later grid search uses this same weight.
    """
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.5,
        max_depth=4,
        subsample=0.5,
        colsample_bytree=0.5,
        scale_pos_weight=5,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def select_features(default_importance, specified_importance):
    """Keep a feature when either model puts more than 0.5% of gain on it."""
    compared = default_importance[["feature", "importance", "above_cutoff"]].merge(
        specified_importance[["feature", "importance", "above_cutoff"]],
        on="feature",
        suffixes=("_default", "_specified"),
    )
    compared["kept"] = compared["above_cutoff_default"] | compared["above_cutoff_specified"]
    compared = compared.sort_values(
        ["kept", "importance_specified", "importance_default"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return compared


def save_table(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    X, y = load_train_matrix()
    feature_names = X.columns.tolist()
    print(
        f"train accounts: {len(X):,} | candidate features: {len(feature_names):,}",
        flush=True,
    )
    print(f"missing cells left as NaN: {int(X.isna().sum().sum()):,}", flush=True)

    print("fitting the default XGBoost model", flush=True)
    default_model = fit_default_model(X, y)
    default_importance = importance_table(default_model, feature_names)

    print("fitting the specified XGBoost model", flush=True)
    specified_model = fit_specified_model(X, y)
    specified_importance = importance_table(specified_model, feature_names)

    compared = select_features(default_importance, specified_importance)
    kept = compared.loc[compared["kept"]].copy()
    if kept.empty:
        raise RuntimeError("No feature has importance above 0.5% in either model")

    save_table(DEFAULT_IMPORTANCE_PATH, default_importance)
    save_table(SPECIFIED_IMPORTANCE_PATH, specified_importance)
    save_table(SELECTED_FEATURES_PATH, kept)

    print(f"saved {DEFAULT_IMPORTANCE_PATH}", flush=True)
    print(f"saved {SPECIFIED_IMPORTANCE_PATH}", flush=True)
    print(f"saved {SELECTED_FEATURES_PATH}", flush=True)
    print(
        "above 0.5% in the default model: "
        f"{int(default_importance['above_cutoff'].sum()):,}",
        flush=True,
    )
    print(
        "above 0.5% in the specified model: "
        f"{int(specified_importance['above_cutoff'].sum()):,}",
        flush=True,
    )
    print(f"features kept for the later models: {len(kept):,}", flush=True)


if __name__ == "__main__":
    main()
