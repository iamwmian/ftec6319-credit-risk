# Data

The model uses the American Express Default Prediction competition files. This folder holds them.

`train_labels.csv` is here. `train_data.csv` is not. It is about 15 GB. It was downloaded, used to build `samples/development_sample.csv`, and then deleted. `train_data.txt` in this folder is the placeholder. Both filenames are listed in `.gitignore` and must not be pushed. They are Kaggle competition data.

Download them from the competition data page:

https://www.kaggle.com/competitions/amex-default-prediction/data

Competition overview:

https://www.kaggle.com/competitions/amex-default-prediction/overview

| File | What it is | Where it is |
|---|---|---|
| `train_data.csv` | Monthly account activity from April 2017 through April 2018. Many rows per account. | Removed after use. See `train_data.txt`. |
| `train_labels.csv` | One row per account. Default status as of April 2018. | This folder |

## Download again

Only needed if you rebuild the development sample.

1. Sign in to Kaggle.
2. Open the data link above and accept the competition rules.
3. Download `train_data.csv`. `train_labels.csv` is already in this folder.
4. Move `train_data.csv` into this folder. Do not rename it. Delete it again after `scripts/part4.py` finishes.

Kaggle CLI, after you have an API token at `~/.kaggle/kaggle.json`:

```bash
kaggle competitions download -c amex-default-prediction -f train_data.csv
```

Unzip if Kaggle gives you a `.zip`, then leave `train_data.csv` in this folder until the sample is rebuilt.
