"""FTEC 6319 credit risk project: build the development sample.

train_labels.csv has one row per account. train_data.csv has one row per
monthly statement, up to 13 statements per account. The development sample is:

1. Exactly 10% of the train_data columns. customer_ID is always included
   because it is the account key. S_2, the statement date, is always included
   because the later time-window features are defined from it. The remaining
   columns in that 10% are a random draw of model features.
2. A random 20% of train_labels rows. Those rows are accounts, so this draw
   does not drop months from an account's history.
3. An inner merge on customer_ID. Every statement for a sampled account is
   kept, and the default flag is attached to each of those statements.

Seed 42 makes a re-run rebuild the same sample.
"""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
SAMPLE_DIR = PROJECT_DIR / "samples"
# Raw Kaggle files live in data/. train_data.csv was removed after use.
# See data/train_data.txt. Derived samples live in samples/.
TRAIN_DATA_PATH = DATA_DIR / "train_data.csv"
TRAIN_LABELS_PATH = DATA_DIR / "train_labels.csv"
OUTPUT_PATH = SAMPLE_DIR / "development_sample.csv"

SEED = 42
COLUMN_FRACTION = 0.10
LABEL_FRACTION = 0.20
CHUNK_SIZE = 500_000

# Not numeric in the source file. Every other train_data column is a float.
STRING_COLUMNS = {"customer_ID", "S_2", "D_63", "D_64"}

# Named in the assignment. Stored as codes here. One-hot encoding comes later.
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


def choose_columns(all_columns, rng):
    """Return exactly 10% of train_data columns, in source-file order."""
    if "customer_ID" not in all_columns:
        raise RuntimeError("train_data.csv has no customer_ID column")
    if "S_2" not in all_columns:
        raise RuntimeError("train_data.csv has no S_2 column")

    n_keep = int(round(COLUMN_FRACTION * len(all_columns)))
    locked = ["customer_ID", "S_2"]
    n_random = n_keep - len(locked)
    if n_random <= 0:
        raise RuntimeError(
            f"10% of {len(all_columns)} columns is {n_keep}, which leaves "
            "no room for a random feature draw"
        )

    pool = [column for column in all_columns if column not in locked]
    drawn = rng.choice(np.array(pool, dtype=object), size=n_random, replace=False)
    drawn = set(drawn.tolist())
    random_features = [column for column in all_columns if column in drawn]
    selected = locked + random_features

    if len(selected) != n_keep or len(set(selected)) != n_keep:
        raise RuntimeError("Column sample is not exactly 10% of train_data")
    return selected


def choose_labels(labels, rng_seed):
    """Return a random 20% of label rows, which are accounts."""
    if list(labels.columns) != ["customer_ID", "target"]:
        raise RuntimeError(f"Unexpected label columns: {list(labels.columns)}")
    if labels["customer_ID"].duplicated().any():
        raise RuntimeError("train_labels.csv has duplicate customer_ID values")
    if not set(labels["target"].unique()).issubset({0, 1}):
        raise RuntimeError("target is not a 0/1 default indicator")

    n_keep = int(round(LABEL_FRACTION * len(labels)))
    sampled = labels.sample(n=n_keep, random_state=rng_seed).reset_index(drop=True)
    if sampled["customer_ID"].duplicated().any():
        raise RuntimeError("Label sample has duplicate accounts")
    return sampled


def read_statements(selected_columns, account_ids):
    """Read the selected columns and keep rows for the sampled accounts only."""
    dtype = {
        column: "string" if column in STRING_COLUMNS else "float64"
        for column in selected_columns
    }
    kept = []
    rows_seen = 0
    rows_kept = 0

    reader = pd.read_csv(
        TRAIN_DATA_PATH,
        usecols=selected_columns,
        dtype=dtype,
        chunksize=CHUNK_SIZE,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        rows_seen += len(chunk)
        matched = chunk[chunk["customer_ID"].isin(account_ids)]
        rows_kept += len(matched)
        if len(matched):
            kept.append(matched)
        print(
            f"chunk {chunk_number}: scanned {rows_seen:,} statement rows, "
            f"kept {rows_kept:,}",
            flush=True,
        )

    if not kept:
        raise RuntimeError("No statement rows matched the sampled accounts")

    print(f"finished scan: {rows_seen:,} statement rows in train_data", flush=True)
    return pd.concat(kept, ignore_index=True)


def main():
    all_columns = list(pd.read_csv(TRAIN_DATA_PATH, nrows=0).columns)
    selected_columns = choose_columns(all_columns, np.random.default_rng(SEED))

    labels = pd.read_csv(
        TRAIN_LABELS_PATH,
        dtype={"customer_ID": "string", "target": "int64"},
    )
    label_sample = choose_labels(labels, SEED)
    account_ids = set(label_sample["customer_ID"].astype(str))

    print(f"train_data columns: {len(all_columns)}", flush=True)
    print(f"columns kept ({len(selected_columns)}): {selected_columns}", flush=True)
    print(
        f"label rows: {len(labels):,} | sampled accounts: {len(label_sample):,}",
        flush=True,
    )

    statements = read_statements(selected_columns, account_ids)
    development = statements.merge(label_sample, on="customer_ID", how="inner")

    if len(development) != len(statements):
        raise RuntimeError(
            "The merge changed the number of statement rows. "
            "The label key is not unique, or customer_ID values did not match."
        )

    found_ids = set(development["customer_ID"].astype(str))
    missing = account_ids - found_ids
    if missing:
        raise RuntimeError(
            f"{len(missing):,} sampled accounts have no rows in train_data"
        )
    extra = found_ids - account_ids
    if extra:
        raise RuntimeError(f"{len(extra):,} accounts are outside the label sample")

    history = development.groupby("customer_ID", sort=False).size()
    if history.min() < 1 or history.max() > 13:
        raise RuntimeError(
            "Statements per account are outside 1 to 13: "
            f"min {history.min()}, max {history.max()}"
        )

    development = development.sort_values(
        ["customer_ID", "S_2"], kind="mergesort"
    ).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT_PATH.with_name(OUTPUT_PATH.name + ".tmp")
    development.to_csv(temporary, index=False)
    temporary.replace(OUTPUT_PATH)

    account_target = development.groupby("customer_ID", sort=False)["target"].first()
    categorical_kept = [
        column for column in selected_columns if column in CATEGORICAL_COLUMNS
    ]

    print(f"saved {OUTPUT_PATH}", flush=True)
    print(f"statement rows: {len(development):,}", flush=True)
    print(f"accounts: {development['customer_ID'].nunique():,}", flush=True)
    print(
        "statements per account: "
        f"min {int(history.min())}, "
        f"median {float(history.median()):.0f}, "
        f"max {int(history.max())}",
        flush=True,
    )
    print(f"output columns: {list(development.columns)}", flush=True)
    print(f"account default rate in sample: {account_target.mean():.4%}", flush=True)
    print(f"default rate in all labels: {labels['target'].mean():.4%}", flush=True)
    print(
        f"default rate in the 20% label draw: {label_sample['target'].mean():.4%}",
        flush=True,
    )
    print(f"categorical columns kept: {categorical_kept}", flush=True)


if __name__ == "__main__":
    main()
