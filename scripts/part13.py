"""FTEC 6319 credit risk project, part 13: choose and save the neural net.

Part 12 scored 32 settings. This step picks one setting from bias and
variance, fits it again on the train sample, and saves that model.

The rule is the same one used for XGBoost. Bias is measured by the average
of the train, Test 1, and Test 2 AUCs. A higher average means less bias.
Variance is measured by the standard deviation of those three AUCs. A
smaller standard deviation means less variance.

The chosen setting is the one with the largest value of

    average AUC - standard deviation of AUC

The three AUCs are the full set being compared, so the standard deviation
divides by 3, not by 2. Ties go to the higher Test 2 AUC, then to the
smaller gap between train and Test 2.

The refit uses the same seed and the same layer stack as the grid, so the
saved net is the setting the table selected.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from part12 import EPOCHS, auc, build_model, load_sample, set_seed
from part12 import TEST1_PATH, TEST2_PATH, TRAIN_PATH

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_DIR / "results"

GRID_PATH = RESULT_DIR / "10_nn_grid_search.csv"
COMPARISON_PATH = RESULT_DIR / "11_nn_grid_bias_variance.csv"
CHOICE_PATH = RESULT_DIR / "12_nn_final_parameters.csv"
MODEL_PATH = RESULT_DIR / "13_nn_final_model.keras"

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


def save_table(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def fit_chosen(X_train, y_train, choice):
    """Refit the winning settings on the processed train sample."""
    set_seed()
    model = build_model(
        n_features=X_train.shape[1],
        n_layers=int(choice["n_hidden_layers"]),
        n_nodes=int(choice["n_nodes"]),
        activation=str(choice["activation"]),
        dropout_pct=int(choice["dropout_pct"]),
    )
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=int(choice["batch_size"]),
        shuffle=True,
        verbose=0,
    )
    return model


def main():
    scored = add_bias_variance(pd.read_csv(GRID_PATH))
    choice = choose_setting(scored)
    scored["chosen"] = False
    scored.loc[choice.name, "chosen"] = True

    X_train, y_train = load_sample(TRAIN_PATH)
    X_test1, y_test1 = load_sample(TEST1_PATH)
    X_test2, y_test2 = load_sample(TEST2_PATH)

    model = fit_chosen(X_train, y_train, choice)
    batch_size = int(choice["batch_size"])
    refit = {
        "auc_train": auc(model, X_train, y_train, batch_size),
        "auc_test1": auc(model, X_test1, y_test1, batch_size),
        "auc_test2": auc(model, X_test2, y_test2, batch_size),
    }
    # The same seed and settings should rebuild the grid row.
    for column, value in refit.items():
        if not np.isclose(value, choice[column], atol=1e-4):
            raise RuntimeError(
                f"Refit {column} is {value:.4f}, grid row has {choice[column]:.4f}"
            )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    save_table(COMPARISON_PATH, scored.sort_values("selection_score", ascending=False))
    save_table(CHOICE_PATH, choice.to_frame().T)

    print(
        "chosen: "
        f"{int(choice['n_hidden_layers'])} layers, "
        f"{int(choice['n_nodes'])} nodes, "
        f"{choice['activation']}, "
        f"dropout {int(choice['dropout_pct'])}%, "
        f"batch {int(choice['batch_size'])}",
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
