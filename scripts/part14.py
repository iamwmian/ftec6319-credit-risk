"""FTEC 6319 credit risk project, part 14: choose XGBoost or the neural net.

The brief's "models of step 11 and step 14" are the two saved finals:
the XGBoost model from part 10 and the neural net from part 13. This step
does not train again. Both models already have a train AUC, a Test 1 AUC,
and a Test 2 AUC.

The choice uses the same score as the two grid searches:

    average of the three AUCs - standard deviation of the three AUCs

The larger score wins. A tie goes to the higher Test 2 AUC, then to the
smaller gap between train and Test 2.
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULT_DIR = PROJECT_DIR / "results"

XGB_PATH = RESULT_DIR / "06_xgb_final_parameters.csv"
NN_PATH = RESULT_DIR / "12_nn_final_parameters.csv"
COMPARISON_PATH = RESULT_DIR / "14_final_model_comparison.csv"

SCORE_COLUMNS = [
    "auc_train",
    "auc_test1",
    "auc_test2",
    "mean_auc",
    "std_auc",
    "train_test2_gap",
    "selection_score",
]


def score_row(path, model_name):
    """Keep the selection columns from one saved final model."""
    table = pd.read_csv(path)
    if len(table) != 1:
        raise RuntimeError(f"{path.name} should contain one model")
    missing = [column for column in SCORE_COLUMNS if column not in table.columns]
    if missing:
        raise RuntimeError(f"{path.name} is missing columns: {missing}")
    row = table.iloc[0][SCORE_COLUMNS].to_dict()
    row["model"] = model_name
    return row


def main():
    comparison = pd.DataFrame(
        [
            score_row(XGB_PATH, "xgboost"),
            score_row(NN_PATH, "neural_net"),
        ]
    )
    comparison = comparison.sort_values(
        ["selection_score", "auc_test2", "train_test2_gap"],
        ascending=[False, False, True],
    )
    comparison["chosen"] = False
    comparison.iloc[0, comparison.columns.get_loc("chosen")] = True
    comparison = comparison[["model", *SCORE_COLUMNS, "chosen"]]

    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = COMPARISON_PATH.with_name(COMPARISON_PATH.name + ".tmp")
    comparison.to_csv(temporary, index=False)
    temporary.replace(COMPARISON_PATH)

    winner = comparison.iloc[0]
    print(
        f"chosen: {winner['model']} "
        f"score={winner['selection_score']:.4f} "
        f"mean={winner['mean_auc']:.4f} "
        f"spread={winner['std_auc']:.4f}",
        flush=True,
    )
    print(f"saved {COMPARISON_PATH}", flush=True)


if __name__ == "__main__":
    main()
