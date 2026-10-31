"""FTEC 6319 credit risk project, part 7: train, Test 1, and Test 2.

account_features.csv has one row per account. This step puts each account
into one sample:

- 70% Train, used to fit the models
- 15% Test 1
- 15% Test 2

Test 1 and Test 2 are held out. Later steps score both so the comparison is
not based on a single test sample.

The draw is stratified on target. Each sample keeps about the same default
rate, so a difference in AUC is not an accident of one sample containing
more defaults. Seed 42 makes the same accounts fall into the same samples
on a re-run.

The split is at the account level. Splitting monthly statements would put
some months of an account in train and the rest in a test sample.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DIR = PROJECT_DIR / "samples"
ACCOUNT_FEATURES_PATH = SAMPLE_DIR / "account_features.csv"
TRAIN_PATH = SAMPLE_DIR / "train.csv"
TEST1_PATH = SAMPLE_DIR / "test1.csv"
TEST2_PATH = SAMPLE_DIR / "test2.csv"

SEED = 42
TRAIN_FRACTION = 0.70
# The remaining 30% is divided in half, which is 15% of all accounts each.
HOLDOUT_FRACTION = 1 - TRAIN_FRACTION
TEST2_FRACTION_OF_HOLDOUT = 0.50

SAMPLE_TRAIN = "Train"
SAMPLE_TEST1 = "Test1"
SAMPLE_TEST2 = "Test2"


def load_accounts():
    """Read the one-row-per-account table from part 6."""
    accounts = pd.read_csv(
        ACCOUNT_FEATURES_PATH,
        dtype={"customer_ID": "string", "latest_statement": "string"},
    )
    if accounts["customer_ID"].duplicated().any():
        raise RuntimeError("account_features.csv has duplicate customer_ID values")
    if not set(accounts["target"].unique()).issubset({0, 1}):
        raise RuntimeError("target is not a 0/1 default indicator")
    return accounts


def assign_samples(accounts):
    """Return train, Test 1, and Test 2 with a sample name on each row.

    The first split holds out 30%. The second split cuts that holdout in
    half. Both splits shuffle and both use seed 42. stratify keeps the
    default rate in each piece close to the rate in the full account table.
    """
    train, holdout = train_test_split(
        accounts,
        test_size=HOLDOUT_FRACTION,
        random_state=SEED,
        stratify=accounts["target"],
    )
    test1, test2 = train_test_split(
        holdout,
        test_size=TEST2_FRACTION_OF_HOLDOUT,
        random_state=SEED,
        stratify=holdout["target"],
    )

    train = train.copy()
    test1 = test1.copy()
    test2 = test2.copy()
    train["sample"] = SAMPLE_TRAIN
    test1["sample"] = SAMPLE_TEST1
    test2["sample"] = SAMPLE_TEST2

    # Stable row order inside each file. Membership comes from the split,
    # not from this sort.
    train = train.sort_values("customer_ID").reset_index(drop=True)
    test1 = test1.sort_values("customer_ID").reset_index(drop=True)
    test2 = test2.sort_values("customer_ID").reset_index(drop=True)
    return train, test1, test2


def check_split(accounts, train, test1, test2):
    """Every account is in exactly one sample, and the sizes add up."""
    pieces = {
        SAMPLE_TRAIN: train,
        SAMPLE_TEST1: test1,
        SAMPLE_TEST2: test2,
    }
    seen = []
    for name, piece in pieces.items():
        if piece["customer_ID"].duplicated().any():
            raise RuntimeError(f"{name} contains duplicate accounts")
        if not piece["sample"].eq(name).all():
            raise RuntimeError(f"{name} has a row labeled as another sample")
        seen.append(set(piece["customer_ID"].astype(str)))

    overlap = (seen[0] & seen[1]) | (seen[0] & seen[2]) | (seen[1] & seen[2])
    if overlap:
        raise RuntimeError(f"{len(overlap):,} accounts appear in more than one sample")

    assigned = seen[0] | seen[1] | seen[2]
    source = set(accounts["customer_ID"].astype(str))
    if assigned != source:
        raise RuntimeError(
            "The three samples do not contain the same accounts as account_features.csv"
        )

    feature_columns = [column for column in accounts.columns if column != "sample"]
    for piece in pieces.values():
        piece_features = [column for column in piece.columns if column != "sample"]
        if piece_features != feature_columns:
            raise RuntimeError("A sample is missing feature columns from the account table")


def save_sample(path, sample):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    sample.to_csv(temporary, index=False)
    temporary.replace(path)


def describe(name, sample, total_accounts):
    default_rate = sample["target"].mean()
    share = len(sample) / total_accounts
    print(
        f"{name}: {len(sample):,} accounts ({share:.2%}), "
        f"default rate {default_rate:.4%}",
        flush=True,
    )


def main():
    accounts = load_accounts()
    train, test1, test2 = assign_samples(accounts)
    check_split(accounts, train, test1, test2)

    save_sample(TRAIN_PATH, train)
    save_sample(TEST1_PATH, test1)
    save_sample(TEST2_PATH, test2)

    total = len(accounts)
    print(f"accounts: {total:,}", flush=True)
    print(f"full-sample default rate: {accounts['target'].mean():.4%}", flush=True)
    describe(SAMPLE_TRAIN, train, total)
    describe(SAMPLE_TEST1, test1, total)
    describe(SAMPLE_TEST2, test2, total)
    print(f"saved {TRAIN_PATH}", flush=True)
    print(f"saved {TEST1_PATH}", flush=True)
    print(f"saved {TEST2_PATH}", flush=True)


if __name__ == "__main__":
    main()
