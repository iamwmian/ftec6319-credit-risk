"""FTEC 6319 credit risk project, part 10: choose and save the XGBoost model.

Part 9 scored 72 settings. This step picks one setting from bias and
variance, fits it again on the train sample, and saves that model.

Bias is a model that ranks poorly on every sample. It is measured by the
average of the train, Test 1, and Test 2 AUCs. A higher average means less
bias.

Variance is a model whose AUC changes from sample to sample, usually
because the train score is higher than the test scores. It is measured by
the standard deviation of those three AUCs. A smaller standard deviation
means less variance.

The chosen setting is the one with the largest value of

    average AUC - standard deviation of AUC

That rewards a high ranking score and penalizes a model that only looks
strong on the train sample. The three AUCs are the full set being compared,
so the standard deviation divides by 3, not by 2.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_DIR / "samples"
RESULT_DIR = PROJECT_DIR / "results"

GRID_PATH = RESULT_DIR / "04_xgb_grid_search.csv"
COMPARISON_PATH = RESULT_DIR / "05_xgb_grid_bias_variance.csv"
CHOICE_PATH = RESULT_DIR / "06_xgb_final_parameters.csv"
MODEL_PATH = RESULT_DIR / "07_xgb_final_model.json"
SELECTED_FEATURES_PATH = RESULT_DIR / "03_selected_features.csv"
TRAIN_PATH = SAMPLE_DIR / "train.csv"
TEST1_PATH = SAMPLE_DIR / "test1.csv"
TEST2_PATH = SAMPLE_DIR / "test2.csv"

SEED = 42
AUC_COLUMNS = ["auc_train", "auc_test1", "auc_test2"]


def add_bias_variance(grid):
    """Add the average AUC, its spread, and the selection score."""
    scored = grid.copy()
    scored["mean_auc"] = scored[AUC_COLUMNS].mean(axis=1)
    # ddof=0 because these three scores are the whole comparison, not a sample.
    scored["std_auc"] = scored[AUC_COLUMNS].std(axis=1, ddof=0)
    scored["train_test2_gap"] = scored["auc_train"] - scored["auc_test2"]
    scored["selection_score"] = scored["mean_auc"] - scored["std_auc"]
    return scored


def choose_setting(scored):
    """Return the row with the best bias-variance score.

    Ties go to the higher Test 2 AUC, then to the smaller train-to-Test 2 gap.
    """
    ordered = scored.sort_values(
        ["selection_score", "auc_test2", "train_test2_gap"],
        ascending=[False, False, True],
    )
    return ordered.iloc[0]


def selected_features():
    features = pd.read_csv(SELECTED_FEATURES_PATH)["feature"].tolist()
    if len(features) != len(set(features)):
        raise RuntimeError("03_selected_features.csv has duplicate feature names")
    return features


def load_sample(path, features):
    frame = pd.read_csv(path, usecols=["target", *features])
    y = frame["target"].to_numpy()
    if not set(y).issubset({0, 1}):
        raise RuntimeError(f"{path.name} target is not a 0/1 default indicator")
    return frame[features], y


def fit_chosen(X_train, y_train, choice):
    """Refit the winning settings. Depth stays at XGBoost's default."""
    model = XGBClassifier(
        n_estimators=int(choice["n_trees"]),
        learning_rate=float(choice["learning_rate"]),
        subsample=float(choice["subsample_pct"]) / 100,
        colsample_bytree=float(choice["feature_pct"]) / 100,
        scale_pos_weight=int(choice["weight_of_default"]),
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def auc(model, X, y):
    probability = model.predict_proba(X)[:, 1]
    return float(roc_auc_score(y, probability))


def save_table(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    scored = add_bias_variance(pd.read_csv(GRID_PATH))
    choice = choose_setting(scored)
    scored["chosen"] = False
    scored.loc[choice.name, "chosen"] = True

    features = selected_features()
    X_train, y_train = load_sample(TRAIN_PATH, features)
    X_test1, y_test1 = load_sample(TEST1_PATH, features)
    X_test2, y_test2 = load_sample(TEST2_PATH, features)

    model = fit_chosen(X_train, y_train, choice)
    refit = {
        "auc_train": auc(model, X_train, y_train),
        "auc_test1": auc(model, X_test1, y_test1),
        "auc_test2": auc(model, X_test2, y_test2),
    }
    # The same seed and settings should rebuild the grid row.
    for column, value in refit.items():
        if not np.isclose(value, choice[column], atol=1e-4):
            raise RuntimeError(
                f"Refit {column} is {value:.4f}, grid row has {choice[column]:.4f}"
            )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    save_table(COMPARISON_PATH, scored.sort_values("selection_score", ascending=False))
    save_table(CHOICE_PATH, choice.to_frame().T)

    print(
        "chosen: "
        f"{int(choice['n_trees'])} trees, "
        f"learning rate {choice['learning_rate']}, "
        f"{int(choice['subsample_pct'])}% of accounts, "
        f"{int(choice['feature_pct'])}% of features, "
        f"default weight {int(choice['weight_of_default'])}",
        flush=True,
    )
    print(
        f"AUC train={refit['auc_train']:.4f} "
        f"test1={refit['auc_test1']:.4f} "
        f"test2={refit['auc_test2']:.4f}",
        flush=True,
    )
    print(
        f"mean AUC={choice['mean_auc']:.4f} "
        f"std AUC={choice['std_auc']:.4f} "
        f"train-test2 gap={choice['train_test2_gap']:.4f}",
        flush=True,
    )
    print(f"saved {MODEL_PATH}", flush=True)
    print(f"saved {COMPARISON_PATH}", flush=True)
    print(f"saved {CHOICE_PATH}", flush=True)


if __name__ == "__main__":
    main()
