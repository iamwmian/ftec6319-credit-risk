"""FTEC 6319 credit risk project, part 6: account-level features.

The development sample has one row per monthly statement. This step keeps
one row per account and replaces the monthly values with summaries.

The 17 model features in the development sample are all numeric. Part 5's
one-hot encoding has no columns to change, so there are no categorical
response-rate features to build.

For each numeric feature the summaries are:

- Average, sum, minimum, maximum, and median over the last 3, 6, 9, and 12
  calendar months. The brief's examples use those four windows and the
  average, sum, minimum, and maximum. Median is included because the
  feature-engineering slide lists it with those summaries.
- The value on the account's most recent statement. The brief calls this
  April 2018. In this file the latest statement month is March 2018, and
  some accounts stop earlier, so the column is named Latest.
- The change from the same calendar month one year earlier:
  (latest - year-ago) / year-ago. The brief writes this as
  (April 2018 - April 2017) / April 2017.

A window is anchored on each account's own latest statement month. The last
6 months ending in March 2018 are October 2017 through March 2018. An
account whose whole history is shorter still uses every month it has inside
that window. An account with only four statements therefore gets a 6-month
average from those four statements, as the brief requires.

Missing monthly values stay out of the summaries. A summary is missing when
every value in its window is missing. A year-over-year change is missing
when the year-ago statement is absent or its value is zero.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_DIR / "samples"
DEVELOPMENT_SAMPLE_PATH = SAMPLE_DIR / "development_sample.csv"
OUTPUT_PATH = SAMPLE_DIR / "account_features.csv"

# Calendar months ending at the account's own latest statement month.
WINDOWS = (3, 6, 9, 12)

# Names follow the brief: X_1_Ave_6, X_1_Sum_3, X_1_Min_6, X_1_Max_9.
STAT_LABELS = {
    "mean": "Ave",
    "sum": "Sum",
    "min": "Min",
    "max": "Max",
    "median": "Med",
}

KEY_COLUMNS = ["customer_ID", "S_2", "target"]
# stmt_month is only the clock used to decide which rows fall in a window.
HELPER_COLUMNS = {"stmt_month"}

# These would need one-hot encoding before this step. None are in the sample.
CATEGORICAL_COLUMNS = [
    "B_30",
    "B_38",
    "D_114",
    "D_116",
    "D_117",
    "D_120",
    "D_126",
    "D_63",
    "D_64",
    "D_66",
    "D_68",
]


def feature_columns(statements):
    """Return the numeric profile columns, in their order in the sample."""
    features = [
        column
        for column in statements.columns
        if column not in KEY_COLUMNS and column not in HELPER_COLUMNS
    ]
    categorical = [column for column in features if column in CATEGORICAL_COLUMNS]
    if categorical:
        raise RuntimeError(
            "One-hot encode these categorical columns before aggregation: "
            + ", ".join(categorical)
        )
    if not features:
        raise RuntimeError("The development sample has no profile features")
    return features


def load_statements():
    """Read the saved development sample and add a calendar-month column."""
    statements = pd.read_csv(
        DEVELOPMENT_SAMPLE_PATH,
        dtype={"customer_ID": "string", "S_2": "string", "target": "int64"},
    )
    statements["S_2"] = pd.to_datetime(statements["S_2"])
    # One statement per calendar month. The window rules use the month, not
    # the day inside the month.
    statements["stmt_month"] = statements["S_2"].dt.to_period("M")
    return statements


def account_keys(statements):
    """One row per account: the default flag, history length, and latest date.

    n_statements and latest_statement explain the windows. They are not
    profile features and should be dropped before the models.
    """
    target_count = statements.groupby("customer_ID", sort=False)["target"].nunique()
    if int(target_count.max()) != 1:
        raise RuntimeError("An account has more than one target value")

    keys = statements.groupby("customer_ID", sort=False).agg(
        target=("target", "first"),
        n_statements=("S_2", "size"),
        latest_statement=("S_2", "max"),
    )
    keys["latest_statement"] = keys["latest_statement"].dt.strftime("%Y-%m-%d")
    return keys


def window_aggregates(statements, features, n_months):
    """Summarize each feature over the last n_months calendar months.

    The start month is the latest month minus (n_months - 1), so a 6-month
    window contains six months, including the latest one. Months before that
    start are left out. Months inside it that the account does not have are
    simply absent; the summary uses the months that exist.
    """
    latest_month = statements.groupby("customer_ID", sort=False)["stmt_month"].transform(
        "max"
    )
    start_month = latest_month - (n_months - 1)
    in_window = statements.loc[
        statements["stmt_month"].ge(start_month),
        ["customer_ID", *features],
    ]
    aggregated = in_window.groupby("customer_ID", sort=False)[features].agg(
        ["mean", "sum", "min", "max", "median", "count"]
    )

    # One DataFrame, built here, rather than a column added on each pass.
    # count is the number of non-missing values. Sum would otherwise turn
    # an all-missing window into 0.
    data = {}
    for feature in features:
        observed = aggregated[(feature, "count")]
        for stat, label in STAT_LABELS.items():
            values = aggregated[(feature, stat)]
            if stat == "sum":
                values = values.where(observed.gt(0))
            data[f"{feature}_{label}_{n_months}"] = values.to_numpy()

    summarized = pd.DataFrame(data, index=aggregated.index)
    if summarized.index.has_duplicates:
        raise RuntimeError(f"The {n_months}-month summary has duplicate accounts")
    return summarized


def latest_and_change(statements, features):
    """Most recent value, and the change from 12 calendar months earlier."""
    latest_rows = (
        statements.sort_values(["customer_ID", "S_2"])
        .groupby("customer_ID", sort=False)
        .tail(1)
        .set_index("customer_ID")
    )
    latest = latest_rows[features].copy()
    latest.columns = [f"{feature}_Latest" for feature in features]

    latest_month = statements.groupby("customer_ID", sort=False)["stmt_month"].transform(
        "max"
    )
    # Same calendar month in the prior year: March 2018 looks back to March 2017.
    year_ago_rows = statements.loc[statements["stmt_month"].eq(latest_month - 12)]
    year_ago_rows = (
        year_ago_rows.sort_values(["customer_ID", "S_2"])
        .groupby("customer_ID", sort=False)
        .tail(1)
        .set_index("customer_ID")
    )
    year_ago = year_ago_rows.reindex(latest.index)[features]
    change = (latest_rows[features] - year_ago) / year_ago
    # A zero year-ago value cannot be the denominator. Leave the change missing.
    change = change.where(year_ago.ne(0))
    change.columns = [f"{feature}_YoY" for feature in features]
    return latest, change


def column_order(features):
    """Place each raw feature's summaries together, windows from short to long."""
    ordered = ["customer_ID", "target", "n_statements", "latest_statement"]
    for feature in features:
        for label in STAT_LABELS.values():
            for n_months in WINDOWS:
                ordered.append(f"{feature}_{label}_{n_months}")
        ordered.append(f"{feature}_Latest")
        ordered.append(f"{feature}_YoY")
    return ordered


def build_account_features(statements):
    """Return one row per account, with every summary column."""
    features = feature_columns(statements)
    keys = account_keys(statements)
    frames = [keys]
    for n_months in WINDOWS:
        summarized = window_aggregates(statements, features, n_months)
        if len(summarized) != len(keys):
            raise RuntimeError(
                f"The {n_months}-month window kept {len(summarized):,} accounts, "
                f"expected {len(keys):,}"
            )
        frames.append(summarized.reindex(keys.index))

    latest, change = latest_and_change(statements, features)
    if len(latest) != len(keys):
        raise RuntimeError("The latest-statement table does not have every account")
    frames.extend([latest.reindex(keys.index), change.reindex(keys.index)])

    accounts = pd.concat(frames, axis=1).reset_index()
    accounts = accounts[column_order(features)].sort_values("customer_ID")
    accounts = accounts.reset_index(drop=True)
    return accounts


def _window_values(account_rows, feature, n_months):
    """Values of one feature inside the same month window the summaries use."""
    latest_month = account_rows["stmt_month"].max()
    in_window = account_rows["stmt_month"].ge(latest_month - (n_months - 1))
    return account_rows.loc[in_window, feature]


def _values_agree(actual, expected):
    if pd.isna(expected):
        return pd.isna(actual)
    return bool(np.isclose(actual, expected, rtol=0, atol=1e-9))


def check_account(statements, accounts, customer_id, checks):
    """Recompute selected summaries from the monthly rows for one account."""
    monthly = statements.loc[statements["customer_ID"].eq(customer_id)].copy()
    row = accounts.loc[accounts["customer_ID"].eq(customer_id)].iloc[0]
    for feature, n_months, stat in checks:
        values = _window_values(monthly, feature, n_months)
        if stat == "mean":
            expected = values.mean()
            column = f"{feature}_Ave_{n_months}"
        elif stat == "sum":
            expected = values.sum(min_count=1)
            column = f"{feature}_Sum_{n_months}"
        elif stat == "min":
            expected = values.min()
            column = f"{feature}_Min_{n_months}"
        elif stat == "max":
            expected = values.max()
            column = f"{feature}_Max_{n_months}"
        else:
            raise RuntimeError(f"Unknown check {stat}")
        if not _values_agree(row[column], expected):
            raise RuntimeError(
                f"{customer_id[:12]} {column}: got {row[column]}, expected {expected}"
            )

    feature = checks[0][0]
    latest_position = monthly["S_2"].idxmax()
    if not _values_agree(row[f"{feature}_Latest"], monthly.at[latest_position, feature]):
        raise RuntimeError(f"{feature}_Latest does not match the newest statement")

    year_ago_month = monthly["stmt_month"].max() - 12
    year_ago = monthly.loc[monthly["stmt_month"].eq(year_ago_month), feature]
    latest_value = monthly.at[latest_position, feature]
    if year_ago.empty or pd.isna(year_ago.iloc[0]) or year_ago.iloc[0] == 0:
        expected_change = np.nan
    else:
        expected_change = (latest_value - year_ago.iloc[0]) / year_ago.iloc[0]
    if not _values_agree(row[f"{feature}_YoY"], expected_change):
        raise RuntimeError(
            f"{feature}_YoY: got {row[f'{feature}_YoY']}, expected {expected_change}"
        )


def check_accounts(statements, accounts):
    """Confirm the row count, the target, and the window math on real accounts."""
    if accounts["customer_ID"].duplicated().any():
        raise RuntimeError("The account table has duplicate customer_ID values")
    source_ids = set(statements["customer_ID"].astype(str))
    result_ids = set(accounts["customer_ID"].astype(str))
    if source_ids != result_ids:
        raise RuntimeError("The account table does not match the sampled accounts")

    source_target = statements.groupby("customer_ID", sort=False)["target"].first()
    compared = accounts[["customer_ID", "target"]].merge(
        source_target.rename("source_target"),
        left_on="customer_ID",
        right_index=True,
        how="left",
    )
    if not compared["target"].eq(compared["source_target"]).all():
        raise RuntimeError("target on the account table does not match the sample")

    history = statements.groupby("customer_ID").size()
    # B_12 has no missing values in this sample, so the window math is visible.
    feature = "B_12"
    long_id = history[history.eq(13)].index[0]
    short_id = history[history.eq(4)].index[0]
    single_id = history[history.eq(1)].index[0]
    check_account(
        statements,
        accounts,
        long_id,
        [
            (feature, 12, "mean"),
            (feature, 6, "min"),
            (feature, 9, "max"),
            (feature, 3, "sum"),
        ],
    )
    check_account(
        statements,
        accounts,
        short_id,
        [(feature, 6, "mean"), (feature, 3, "mean"), (feature, 12, "mean")],
    )
    check_account(
        statements,
        accounts,
        single_id,
        [(feature, 3, "mean"), (feature, 6, "sum"), (feature, 12, "max")],
    )

    # A 13-statement account covers March 2017 through March 2018. The 12-month
    # window starts in April 2017, so March 2017 must stay outside that average.
    long_rows = statements.loc[statements["customer_ID"].eq(long_id)]
    long_end = long_rows["stmt_month"].max()
    if long_rows["stmt_month"].min() != long_end - 12:
        raise RuntimeError("Expected the 13-statement account to start 12 months back")
    oldest = long_rows.loc[long_rows["stmt_month"].eq(long_end - 12), feature].iloc[0]
    overall = long_rows[feature].mean()
    last_year = accounts.loc[
        accounts["customer_ID"].eq(long_id), f"{feature}_Ave_12"
    ].iloc[0]
    if np.isclose(last_year, overall) and not np.isclose(oldest, overall):
        raise RuntimeError("Ave_12 included the statement from 12 months earlier")


def main():
    statements = load_statements()
    features = feature_columns(statements)
    print(
        f"statements: {len(statements):,} | accounts: {statements['customer_ID'].nunique():,}",
        flush=True,
    )
    print(f"numeric features ({len(features)}): {features}", flush=True)
    print("categorical features in the sample: none", flush=True)

    accounts = build_account_features(statements)
    check_accounts(statements, accounts)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".tmp")
    accounts.to_csv(temporary, index=False)
    temporary.replace(OUTPUT_PATH)

    summary_columns = [
        column
        for column in accounts.columns
        if column not in {"customer_ID", "target", "n_statements", "latest_statement"}
    ]
    print(f"saved {OUTPUT_PATH}", flush=True)
    print(f"rows: {len(accounts):,}", flush=True)
    print(f"columns: {accounts.shape[1]:,}", flush=True)
    print(f"summary columns: {len(summary_columns):,}", flush=True)
    print(
        "statements per account: "
        f"min {int(accounts['n_statements'].min())}, "
        f"median {float(accounts['n_statements'].median()):.0f}, "
        f"max {int(accounts['n_statements'].max())}",
        flush=True,
    )
    yoy_columns = [column for column in summary_columns if column.endswith("_YoY")]
    yoy_present = accounts[yoy_columns].notna().any(axis=1).sum()
    print(
        f"accounts with a year-ago statement used in at least one change: {yoy_present:,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
