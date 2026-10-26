# Data

The model uses the American Express Default Prediction competition files. Both files stay on your machine. They are listed in `.gitignore` and must not be pushed. `train_data.csv` is about 15 GB, which GitHub will not accept, and the files are Kaggle competition data.

Download them from the competition data page:

https://www.kaggle.com/competitions/amex-default-prediction/data

Competition overview:

https://www.kaggle.com/competitions/amex-default-prediction/overview

| File | What it is | Put it here |
|---|---|---|
| `train_data.csv` | Monthly account activity from April 2017 through April 2018. Many rows per account. | This folder, next to this file |
| `train_labels.csv` | One row per account. Default status as of April 2018. | This folder, next to this file |

## Download

1. Sign in to Kaggle.
2. Open the data link above and accept the competition rules.
3. Download `train_data.csv` and `train_labels.csv`.
4. Move both files into this `CR_Project` folder. Do not rename them.

Kaggle CLI, after you have an API token at `~/.kaggle/kaggle.json`:

```bash
kaggle competitions download -c amex-default-prediction -f train_labels.csv
kaggle competitions download -c amex-default-prediction -f train_data.csv
```

Unzip if Kaggle gives you a `.zip`, then leave the `.csv` files in this folder.
