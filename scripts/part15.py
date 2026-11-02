"""FTEC 6319 credit risk project, part 15: strategy thresholds.

The brief's strategy section is this step. The function scores one
threshold. The run below applies it to the train sample and saves the
two chosen thresholds in the strategy folder.

An account i has four numbers:

    Y_i     observed default, 1 if the account defaulted and 0 otherwise
    PD_i    the model's probability of default
    B_i     estimated monthly balance, the last-6-month average of B_37
    S_i     estimated monthly spend, the last-6-month average of S_27

t is a threshold between 0 and 1. The account is accepted when PD_i < t
and rejected when PD_i > t. A lower t accepts fewer accounts.

t_C is the conservative threshold and t_A is the aggressive threshold,
with t_C < t_A. Written on one line, Ct and At can be read as products,
so the symbols here keep the threshold in the subscript.

Monthly revenue for one account, before the default wipeout, is

    r_i = 0.02 * B_i + 0.001 * S_i

The 0.02 is the monthly interest rate that matches 24% a year. The 0.001
is the 0.1% charge on spend. Expected revenue over the next 12 months is

    R_i = 12 * r_i    if Y_i = 0
    R_i = 0           if Y_i = 1

Rejected accounts are left out of both totals. For a threshold t,

    DR(t)  = sum of Y_i over accepted accounts / number accepted
    Rev(t) = sum of R_i over accepted accounts

The firm requires DR(t) <= 0.10. On the train sample:

    t_A = threshold with the largest Rev(t) among those with DR(t) <= 0.10
    t_C = threshold with the largest Rev(t) among those with DR(t) <= 0.05

The 5% cap is half the firm's limit. It is the conservative book. The
search checks t = 0.01, 0.02, ..., 0.99.

S_27 is the only spend feature in the development sample. B_37 is the
balance feature both importance screens kept. The brief asks for the
last-6-month average, so the columns are S_27_Ave_6 and B_37_Ave_6.
A missing spend average is treated as 0: there is no observed spend to
charge. These averages are anonymized model units, not dollars. The
formula still uses 0.02 and 0.001, as written.

Thresholds are chosen on the train sample only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
STRATEGY_DIR = PROJECT_DIR / "strategy"
RESULT_DIR = PROJECT_DIR / "results"
SAMPLE_DIR = PROJECT_DIR / "samples"

MODEL_PATH = RESULT_DIR / "07_xgb_final_model.json"
FEATURES_PATH = RESULT_DIR / "03_selected_features.csv"
TRAIN_PATH = SAMPLE_DIR / "train.csv"

BOOK_PATH = STRATEGY_DIR / "train_book.csv"
SEARCH_PATH = STRATEGY_DIR / "threshold_search.csv"
CHOICE_PATH = STRATEGY_DIR / "chosen_strategies.csv"

SPEND_COLUMN = "S_27_Ave_6"
BALANCE_COLUMN = "B_37_Ave_6"
TARGET_COLUMN = "target"
PROBABILITY_COLUMN = "pd"
BALANCE_NAME = "monthly_balance"
SPEND_NAME = "monthly_spend"

# 24% a year is 2% a month. The spend charge is 0.1% of monthly spend.
MONTHLY_BALANCE_RATE = 0.02
SPEND_RATE = 0.001
REVENUE_MONTHS = 12

FIRM_DEFAULT_LIMIT = 0.10
CONSERVATIVE_DEFAULT_LIMIT = 0.05
THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def portfolio_result(data, target_name, probability_name, balance_name, spend_name, threshold):
    """Default rate and 12-month revenue for one threshold.

    The return value is (default_rate, expected_revenue). default_rate is
    the share of accepted accounts with Y = 1. expected_revenue sums R_i
    over accepted accounts. An empty accepted set has no default rate.
    """
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    for column in (target_name, probability_name, balance_name, spend_name):
        if column not in data.columns:
            raise ValueError(f"data has no column named {column}")

    accepted = data.loc[data[probability_name] < threshold]
    if len(accepted) == 0:
        return float("nan"), 0.0

    default_rate = float(accepted[target_name].mean())
    monthly = (
        MONTHLY_BALANCE_RATE * accepted[balance_name]
        + SPEND_RATE * accepted[spend_name]
    )
    revenue = REVENUE_MONTHS * monthly
    # A defaulted account contributes no revenue, even though it stays in DR(t).
    revenue = revenue.where(accepted[target_name] == 0, 0.0)
    return default_rate, float(revenue.sum())


def selected_features():
    features = pd.read_csv(FEATURES_PATH)["feature"].tolist()
    if len(features) != len(set(features)):
        raise RuntimeError("03_selected_features.csv has duplicate feature names")
    return features


def load_model():
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def build_train_book(model, features):
    """Score the train accounts and attach the monthly balance and spend."""
    extra = [column for column in (SPEND_COLUMN, BALANCE_COLUMN) if column not in features]
    frame = pd.read_csv(
        TRAIN_PATH,
        usecols=["customer_ID", TARGET_COLUMN, *features, *extra],
        dtype={"customer_ID": "string"},
    )
    if frame["customer_ID"].duplicated().any():
        raise RuntimeError("train.csv has more than one row for an account")

    missing_spend = int(frame[SPEND_COLUMN].isna().sum())
    frame[SPEND_COLUMN] = frame[SPEND_COLUMN].fillna(0.0)
    if frame[BALANCE_COLUMN].isna().any():
        raise RuntimeError(f"{BALANCE_COLUMN} has missing values")

    probability = model.predict_proba(frame[features])[:, 1]
    book = pd.DataFrame(
        {
            "customer_ID": frame["customer_ID"],
            TARGET_COLUMN: frame[TARGET_COLUMN].to_numpy(),
            PROBABILITY_COLUMN: probability,
            BALANCE_NAME: frame[BALANCE_COLUMN].to_numpy(),
            SPEND_NAME: frame[SPEND_COLUMN].to_numpy(),
        }
    )
    return book, missing_spend


def search_thresholds(book):
    """Evaluate every threshold from 0.01 through 0.99."""
    rows = []
    for threshold in THRESHOLDS:
        default_rate, revenue = portfolio_result(
            book,
            TARGET_COLUMN,
            PROBABILITY_COLUMN,
            BALANCE_NAME,
            SPEND_NAME,
            threshold,
        )
        accepted = int((book[PROBABILITY_COLUMN] < threshold).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "n_accepted": accepted,
                "accept_rate": accepted / len(book),
                "default_rate": default_rate,
                "expected_revenue": revenue,
                "within_firm_limit": bool(default_rate <= FIRM_DEFAULT_LIMIT),
                "within_conservative_limit": bool(default_rate <= CONSERVATIVE_DEFAULT_LIMIT),
            }
        )
    return pd.DataFrame(rows)


def choose_strategies(search):
    """Pick t_A under the 10% cap and t_C under the 5% cap."""
    aggressive_pool = search.loc[search["within_firm_limit"]].sort_values(
        ["expected_revenue", "threshold"],
        ascending=[False, False],
    )
    if aggressive_pool.empty:
        raise RuntimeError("No threshold keeps the default rate at or below 10%")
    aggressive = aggressive_pool.iloc[0]

    conservative_pool = search.loc[
        search["within_conservative_limit"] & (search["threshold"] < aggressive["threshold"])
    ].sort_values(
        ["expected_revenue", "threshold"],
        ascending=[False, True],
    )
    if conservative_pool.empty:
        raise RuntimeError("No threshold keeps the default rate at or below 5%")
    conservative = conservative_pool.iloc[0]

    chosen = pd.DataFrame(
        [
            {"strategy": "conservative", "symbol": "t_C", **conservative.to_dict()},
            {"strategy": "aggressive", "symbol": "t_A", **aggressive.to_dict()},
        ]
    )
    if not chosen.loc[0, "threshold"] < chosen.loc[1, "threshold"]:
        raise RuntimeError("Conservative threshold is not below the aggressive threshold")
    return chosen


def save_table(path, table):
    temporary = path.with_name(path.name + ".tmp")
    table.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    features = selected_features()
    book, missing_spend = build_train_book(load_model(), features)
    search = search_thresholds(book)
    chosen = choose_strategies(search)

    save_table(BOOK_PATH, book)
    save_table(SEARCH_PATH, search)
    save_table(CHOICE_PATH, chosen)

    print(f"train accounts: {len(book):,}", flush=True)
    print(f"missing {SPEND_COLUMN} filled with 0: {missing_spend:,}", flush=True)
    print(
        f"balance column: {BALANCE_COLUMN}   spend column: {SPEND_COLUMN}",
        flush=True,
    )
    for row in chosen.itertuples(index=False):
        print(
            f"{row.symbol} {row.strategy}: t={row.threshold:.2f} "
            f"accepted={row.n_accepted:,.0f} "
            f"default rate={row.default_rate:.4f} "
            f"revenue={row.expected_revenue:.2f}",
            flush=True,
        )
    print(f"saved {BOOK_PATH}", flush=True)
    print(f"saved {SEARCH_PATH}", flush=True)
    print(f"saved {CHOICE_PATH}", flush=True)


if __name__ == "__main__":
    main()
