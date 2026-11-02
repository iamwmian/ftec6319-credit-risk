"""Build the numbered tables and figures for the 17-slide deck.

Outputs go to CR_Project/presentation. Slide numbers in the file names
follow the brief. Figures use a white background and large type so they
can be dropped onto a slide.
"""

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from tensorflow import keras
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from part15 import portfolio_result
from plot_bias_variance import axis_limits

SAMPLE_DIR = PROJECT_DIR / "samples"
RESULT_DIR = PROJECT_DIR / "results"
OUT = PROJECT_DIR / "presentation"

CATEGORIES = [
    ("D", "Delinquency"),
    ("S", "Spend"),
    ("P", "Payment"),
    ("B", "Balance"),
    ("R", "Risk"),
]
CATEGORICALS = [
    "B_30", "B_38", "D_63", "D_64", "D_66", "D_68",
    "D_114", "D_116", "D_117", "D_120", "D_126",
]
STAT_ORDER = ["Ave", "Sum", "Min", "Max", "Med", "Latest", "YoY"]
FIRM_LIMIT = 0.10
CONSERVATIVE_T = 0.32
AGGRESSIVE_T = 0.54
SLIDE17_THRESHOLDS = [round(x, 1) for x in np.arange(0.1, 0.9, 0.1)]

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#222222",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
})


def comma(value):
    return f"{int(round(value)):,}"


def pct(value):
    return f"{100 * float(value):.2f}%"


def auc_text(value):
    return f"{float(value):.3f}"


def money(value):
    return f"{float(value):,.2f}"


def stat_text(value):
    if pd.isna(value):
        return ""
    if abs(float(value)) < 1:
        return f"{float(value):.3f}"
    return f"{float(value):,.2f}"


def save_table(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    print(f"saved {path.name}", flush=True)


def selected_features():
    return pd.read_csv(RESULT_DIR / "03_selected_features.csv")["feature"].tolist()


def parse_engineered(name):
    for stat in ("Ave", "Sum", "Min", "Max", "Med"):
        token = f"_{stat}_"
        if token in name:
            root, _window = name.split(token, 1)
            return root, stat
    if name.endswith("_Latest"):
        return name[: -len("_Latest")], "Latest"
    if name.endswith("_YoY"):
        return name[: -len("_YoY")], "YoY"
    raise ValueError(f"Cannot parse engineered feature {name}")


def category_of(root):
    return root.split("_", 1)[0]


def load_split(path, features):
    extra = [column for column in ("S_27_Ave_6", "B_37_Ave_6") if column not in features]
    frame = pd.read_csv(
        path,
        usecols=["customer_ID", "target", "n_statements", *features, *extra],
        dtype={"customer_ID": "string"},
    )
    return frame


def score_book(model, frame, features):
    spend_missing = int(frame["S_27_Ave_6"].isna().sum())
    book = pd.DataFrame(
        {
            "customer_ID": frame["customer_ID"],
            "target": frame["target"].to_numpy(),
            "pd": model.predict_proba(frame[features])[:, 1],
            "monthly_balance": frame["B_37_Ave_6"].to_numpy(),
            "monthly_spend": frame["S_27_Ave_6"].fillna(0.0).to_numpy(),
        }
    )
    return book, spend_missing


def strategy_row(book, threshold):
    default_rate, revenue = portfolio_result(
        book, "target", "pd", "monthly_balance", "monthly_spend", threshold
    )
    accepted = int((book["pd"] < threshold).sum())
    return accepted, default_rate, revenue


def history_table(frames):
    history = pd.concat(
        [frame[["n_statements", "target"]] for frame in frames.values()],
        ignore_index=True,
    )
    history["n_statements"] = history["n_statements"].astype(int)
    rows = [{
        "History": "All applications",
        "Accounts": comma(len(history)),
        "Default rate": pct(history["target"].mean()),
    }]
    for months in range(13, 0, -1):
        group = history.loc[history["n_statements"] == months]
        history_label = "1 month" if months == 1 else f"{months} months"
        rows.append({
            "History": history_label,
            "Accounts": comma(len(group)),
            "Default rate": pct(group["target"].mean()) if len(group) else "",
        })
    return pd.DataFrame(rows)


def category_tables(engineered_columns, selected):
    raw_roots = []
    stats = []
    for column in engineered_columns:
        root, stat = parse_engineered(column)
        raw_roots.append(root)
        stats.append(stat)
    raw = pd.DataFrame({"root": raw_roots, "category": [category_of(r) for r in raw_roots]})
    raw_unique = raw.drop_duplicates("root")
    selected_rows = []
    for feature in selected:
        root, stat = parse_engineered(feature)
        selected_rows.append({"feature": feature, "root": root, "category": category_of(root), "statistic": stat})
    selected_frame = pd.DataFrame(selected_rows)
    raw_counts = raw_unique.groupby("category").size()
    engineered_counts = raw.groupby("category").size()
    selected_counts = selected_frame.groupby("category").size()
    rows = []
    for code, label in CATEGORIES:
        rows.append({
            "Category": label,
            "Code": code,
            "Raw features": int(raw_counts.get(code, 0)),
            "Engineered features": int(engineered_counts.get(code, 0)),
            "Selected for modeling": int(selected_counts.get(code, 0)),
        })
    by_category = pd.DataFrame(rows)
    stat_rows = []
    engineered_stats = pd.Series(stats)
    for stat in STAT_ORDER:
        stat_rows.append({
            "Statistic": stat,
            "Engineered features": int((engineered_stats == stat).sum()),
            "Selected for modeling": int((selected_frame["statistic"] == stat).sum()),
        })
    return by_category, pd.DataFrame(stat_rows), raw_unique, selected_frame


def feature_selection_figure(default_kept, specified_kept, union_kept, path):
    figure, axis = plt.subplots(figsize=(12, 6.2))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 6)
    axis.axis("off")
    boxes = [
        (0.4, 2.3, 2.6, 1.6, f"374 summaries\nfrom 17 raw fields", "#F3F4F6"),
        (4.0, 3.5, 3.2, 1.5, f"Default XGBoost\n{default_kept} features above 0.5%", "#E8EEF5"),
        (4.0, 1.0, 3.2, 1.5, f"Specified XGBoost\n{specified_kept} features above 0.5%", "#E8EEF5"),
        (8.6, 2.3, 3.0, 1.6, f"Union kept\n{union_kept} features", "#F4DDD2"),
    ]
    for x, y, w, h, text, color in boxes:
        axis.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
            facecolor=color, edgecolor="#222222", linewidth=1.2,
        ))
        axis.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=15)
    for start, end in (((3.0, 3.1), (4.0, 4.2)), ((3.0, 3.1), (4.0, 1.8)), ((7.2, 4.2), (8.6, 3.3)), ((7.2, 1.8), (8.6, 2.9))):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, color="#222222", linewidth=1.2))
    axis.set_title("Feature selection", fontsize=20, pad=8)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_mean_spread(grid_path, title, path):
    grid = pd.read_csv(grid_path)
    chosen = grid.loc[grid["chosen"]].iloc[0]
    others = grid.loc[~grid["chosen"]]
    score = float(chosen["selection_score"])
    figure, axis = plt.subplots(figsize=(11, 7.2))
    x_left, x_right, y_top = axis_limits(grid)
    shown = others.loc[others["mean_auc"] >= x_left]
    axis.scatter(shown["mean_auc"], shown["std_auc"], s=70, color="#8A8F98", label="Other settings", zorder=2)
    axis.scatter([chosen["mean_auc"]], [chosen["std_auc"]], s=160, color="#C2410C", label="Chosen model", zorder=3)
    axis.plot([x_left, x_right], [x_left - score, x_right - score], linestyle="--", color="#222222", linewidth=1.2, label="Winning score", zorder=1)
    axis.set_xlim(x_left, x_right)
    axis.set_ylim(0, y_top)
    axis.set_title(title)
    axis.set_xlabel("Average AUC")
    axis.set_ylabel("Spread of AUC")
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _p: f"{value:.3f}"))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _p: f"{value:.3f}"))
    hidden = int((grid["mean_auc"] < x_left).sum())
    if hidden:
        axis.text(0.98, 0.02, f"{hidden} weaker settings are left of this window", transform=axis.transAxes, ha="right", va="bottom", fontsize=13, color="#444444")
    axis.legend(frameon=False, loc="upper left")
    axis.grid(True, color="#E6E6E6", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_train_test2(grid, title, path, label_fn):
    figure, axis = plt.subplots(figsize=(11, 7.2))
    chosen = grid.loc[grid["chosen"]].iloc[0]
    best_test = grid.loc[grid["auc_test2"].idxmax()]
    # One failed setting would otherwise pin every useful model into a corner.
    shown = grid.loc[grid["auc_test2"] >= 0.85]
    if len(shown) < 5:
        shown = grid
    others = shown.loc[~shown["chosen"]]
    axis.scatter(others["auc_train"], others["auc_test2"], s=70, color="#8A8F98", label="Other settings", zorder=2)
    axis.scatter([chosen["auc_train"]], [chosen["auc_test2"]], s=170, color="#C2410C", label="Chosen on average minus spread", zorder=4)
    same = np.isclose(best_test["auc_test2"], chosen["auc_test2"]) and np.isclose(best_test["auc_train"], chosen["auc_train"])
    if not same and float(best_test["auc_test2"]) >= 0.85:
        axis.scatter([best_test["auc_train"]], [best_test["auc_test2"]], s=140, facecolors="none", edgecolors="#1D4E89", linewidths=2.2, label="Highest Test 2 AUC", zorder=3)
    x_left = float(shown["auc_train"].min()) - 0.002
    x_right = float(shown["auc_train"].max()) + 0.002
    y_low = float(shown["auc_test2"].min()) - 0.001
    y_high = float(shown["auc_test2"].max()) + 0.001
    if y_high > x_left and x_right > y_low:
        line_low = max(x_left, y_low)
        line_high = min(x_right, y_high)
        if (line_high - line_low) > 0.2 * (x_right - x_left):
            axis.plot([line_low, line_high], [line_low, line_high], linestyle=":", color="#222222", linewidth=1.2, label="Train AUC = Test 2 AUC", zorder=1)
    axis.set_xlim(x_left, x_right)
    axis.set_ylim(y_low, y_high)
    hidden = len(grid) - len(shown)
    if hidden:
        axis.text(0.98, 0.02, f"{hidden} weak settings are outside this window", transform=axis.transAxes, ha="right", va="bottom", fontsize=13, color="#444444")
    axis.set_title(title)
    axis.set_xlabel("Train AUC")
    axis.set_ylabel("Test 2 AUC")
    axis.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _p: f"{value:.3f}"))
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _p: f"{value:.3f}"))
    legend_loc = "upper right" if (x_right - float(chosen["auc_train"])) > 0.008 else "upper left"
    axis.legend(frameon=False, loc=legend_loc)
    axis.grid(True, color="#E6E6E6", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return chosen, best_test, same


def score_intervals(edges):
    """Two decimals, or more when a bin's two ends would print as the same number."""
    labels = None
    for decimals in (2, 3, 4):
        labels = []
        readable = True
        for i in range(len(edges) - 1):
            left = f"{edges[i]:.{decimals}f}"
            right = f"{edges[i + 1]:.{decimals}f}"
            readable = readable and left != right
            labels.append(f"{left}–{right}")
        if readable and len(labels) == len(set(labels)):
            return labels
    return labels


def rank_bins(probabilities, targets, edges):
    bins = pd.cut(probabilities, bins=edges, include_lowest=True, labels=False)
    rows = []
    for bin_id in range(len(edges) - 1):
        mask = bins == bin_id
        count = int(mask.sum())
        rate = float(targets[mask].mean()) if count else np.nan
        rows.append((bin_id + 1, count, rate))
    return rows


def rank_chart(table, title, path):
    figure, axis = plt.subplots(figsize=(12, 7.2))
    bins = sorted(table["Bin"].unique())
    x = np.arange(len(bins))
    width = 0.25
    colors = {"Train": "#4B5563", "Test 1": "#1D4E89", "Test 2": "#C2410C"}
    for offset, sample in enumerate(("Train", "Test 1", "Test 2")):
        part = table.loc[table["Sample"] == sample].set_index("Bin").loc[bins]
        axis.bar(x + (offset - 1) * width, part["Default rate value"] * 100, width=width, color=colors[sample], label=sample)
    labels = table.loc[table["Sample"] == "Train"].set_index("Bin").loc[bins, "Score interval"]
    axis.set_xticks(x)
    axis.set_xticklabels(labels, rotation=30, ha="right", fontsize=11)
    axis.set_xlabel("Predicted probability of default")
    axis.set_ylabel("Actual default rate")
    axis.set_title(title)
    axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _p: f"{value:.0f}%"))
    axis.legend(frameon=False)
    axis.grid(True, axis="y", color="#E6E6E6", linewidth=0.8)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def finish_plot(path):
    figure = plt.gcf()
    figure.set_size_inches(12, 7.5)
    figure.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    features = selected_features()
    frames = {
        "Train": load_split(SAMPLE_DIR / "train.csv", features),
        "Test 1": load_split(SAMPLE_DIR / "test1.csv", features),
        "Test 2": load_split(SAMPLE_DIR / "test2.csv", features),
    }
    model = XGBClassifier()
    model.load_model(RESULT_DIR / "07_xgb_final_model.json")
    books = {}
    missing_spend = {}
    for sample, frame in frames.items():
        books[sample], missing_spend[sample] = score_book(model, frame, features)
        print(f"scored {sample}: {len(books[sample]):,}", flush=True)

    # Slide 1. Chosen strategies on all three samples.
    slide1 = []
    for strategy, threshold in (("Conservative", CONSERVATIVE_T), ("Aggressive", AGGRESSIVE_T)):
        row = {"Strategy": strategy, "Threshold": f"{threshold:.2f}"}
        for sample in ("Train", "Test 1", "Test 2"):
            accepted, default_rate, revenue = strategy_row(books[sample], threshold)
            row[f"{sample} accepted"] = comma(accepted)
            row[f"{sample} default rate"] = pct(default_rate)
            row[f"{sample} revenue"] = money(revenue)
            row[f"_{sample}_default"] = default_rate
        slide1.append(row)
    slide1_table = pd.DataFrame(slide1)
    save_table(OUT / "01_strategy_results.csv", slide1_table.drop(columns=[c for c in slide1_table.columns if c.startswith("_")]))

    aggressive_holds = all(slide1[1][f"_{sample}_default"] <= FIRM_LIMIT for sample in ("Test 1", "Test 2"))
    if aggressive_holds:
        recommendation = (
            "Recommend the aggressive strategy. On Train, Test 1, and Test 2 its default rate stays at or below 10%, "
            "and it produces more 12-month revenue than the conservative strategy."
        )
    else:
        recommendation = (
            "Recommend the conservative strategy. The aggressive book stays within 10% on the train sample, "
            "but it exceeds 10% on a held-out sample. The conservative book is the one that keeps the firm's limit."
        )
    (OUT / "01_recommendation.txt").write_text(recommendation + "\n")

    # Slide 2.
    save_table(OUT / "02_history_length.csv", history_table(frames))

    # Slides 3 and 4 category counts. SHAP top 5 is filled after the explainer runs.
    header = pd.read_csv(SAMPLE_DIR / "train.csv", nrows=0).columns
    skip = {"customer_ID", "target", "n_statements", "latest_statement", "sample"}
    engineered_columns = [column for column in header if column not in skip]
    by_category, by_stat, raw_unique, selected_frame = category_tables(engineered_columns, features)
    raw_list = raw_unique.copy()
    raw_list["Category"] = raw_list["category"].map(dict(CATEGORIES))
    raw_list = raw_list.rename(columns={"root": "Raw feature"})[["Raw feature", "Category"]].sort_values(["Category", "Raw feature"])
    save_table(OUT / "03_raw_features.csv", raw_list)
    save_table(OUT / "03_raw_feature_categories.csv", by_category[["Category", "Raw features"]].rename(columns={"Raw features": "Features"}))
    save_table(OUT / "04_engineered_feature_categories.csv", by_category[["Category", "Engineered features", "Selected for modeling"]])
    save_table(OUT / "04_statistic_types.csv", by_stat)

    # Slide 5.
    present = set(engineered_columns) | set(raw_unique["root"])
    one_hot = pd.DataFrame({
        "Categorical variable": CATEGORICALS,
        "In the development sample": ["Yes" if name in present else "No" for name in CATEGORICALS],
        "One-hot columns created": [0 for _ in CATEGORICALS],
    })
    save_table(OUT / "05_categorical_treatment.csv", one_hot)

    # Slide 6.
    default_imp = pd.read_csv(RESULT_DIR / "01_xgb_default_importance.csv")
    specified_imp = pd.read_csv(RESULT_DIR / "02_xgb_specified_importance.csv")
    selected_imp = pd.read_csv(RESULT_DIR / "03_selected_features.csv")
    default_kept = int(default_imp["above_cutoff"].sum())
    specified_kept = int(specified_imp["above_cutoff"].sum())
    union_kept = int(selected_imp["kept"].sum())
    feature_selection_figure(default_kept, specified_kept, union_kept, OUT / "06_feature_selection.png")
    with pd.ExcelWriter(OUT / "06_feature_importance.xlsx") as writer:
        default_imp.to_excel(writer, sheet_name="Default model", index=False)
        specified_imp.to_excel(writer, sheet_name="Specified model", index=False)
        selected_imp.to_excel(writer, sheet_name="Kept union", index=False)
    print("saved 06_feature_importance.xlsx", flush=True)
    print("saved 06_feature_selection.png", flush=True)

    # Slides 8 and 14 scatters. Slide 7 and 13 text come from these grids.
    xgb_grid = pd.read_csv(RESULT_DIR / "05_xgb_grid_bias_variance.csv")
    nn_grid = pd.read_csv(RESULT_DIR / "11_nn_grid_bias_variance.csv")
    plot_mean_spread(RESULT_DIR / "05_xgb_grid_bias_variance.csv", "XGBoost: average AUC and spread", OUT / "08_xgb_mean_vs_spread.png")
    plot_mean_spread(RESULT_DIR / "11_nn_grid_bias_variance.csv", "Neural net: average AUC and spread", OUT / "14_nn_mean_vs_spread.png")
    _c1, xgb_best_test, xgb_same = plot_train_test2(xgb_grid, "XGBoost: train AUC and Test 2 AUC", OUT / "08_xgb_train_vs_test2.png", None)
    _c2, nn_best_test, nn_same = plot_train_test2(nn_grid, "Neural net: train AUC and Test 2 AUC", OUT / "14_nn_train_vs_test2.png", None)
    print("saved scatter plots", flush=True)

    # Slide 9 and 15 parameter cards are written with the rank-order results.
    def parameter_card_xgb():
        row = pd.read_csv(RESULT_DIR / "06_xgb_final_parameters.csv").iloc[0]
        return pd.DataFrame([
            ["Trees", f"{int(row['n_trees'])}"],
            ["Learning rate", f"{row['learning_rate']:.2f}"],
            ["Accounts per tree", f"{int(row['subsample_pct'])}%"],
            ["Features per tree", f"{int(row['feature_pct'])}%"],
            ["Weight of a default", f"{int(row['weight_of_default'])}"],
            ["Train AUC", auc_text(row["auc_train"])],
            ["Test 1 AUC", auc_text(row["auc_test1"])],
            ["Test 2 AUC", auc_text(row["auc_test2"])],
            ["Average AUC", auc_text(row["mean_auc"])],
            ["Spread of AUC", auc_text(row["std_auc"])],
            ["Selection score", auc_text(row["selection_score"])],
        ], columns=["Item", "Value"])

    def parameter_card_nn():
        row = pd.read_csv(RESULT_DIR / "12_nn_final_parameters.csv").iloc[0]
        dropout = "None" if int(row["dropout_pct"]) == 100 else "50%"
        return pd.DataFrame([
            ["Hidden layers", f"{int(row['n_hidden_layers'])}"],
            ["Nodes in each layer", f"{int(row['n_nodes'])}"],
            ["Activation", str(row["activation"])],
            ["Dropout", dropout],
            ["Batch size", comma(row["batch_size"])],
            ["Epochs", "20"],
            ["Optimizer", "Adam"],
            ["Loss", "Binary cross-entropy"],
            ["Train AUC", auc_text(row["auc_train"])],
            ["Test 1 AUC", auc_text(row["auc_test1"])],
            ["Test 2 AUC", auc_text(row["auc_test2"])],
            ["Average AUC", auc_text(row["mean_auc"])],
            ["Spread of AUC", auc_text(row["std_auc"])],
            ["Selection score", auc_text(row["selection_score"])],
        ], columns=["Item", "Value"])

    def write_rank(probabilities, targets, stem, title):
        _codes, edges = pd.qcut(probabilities["Train"], 10, retbins=True, duplicates="drop")
        edges = np.unique(edges.astype(float))
        intervals = score_intervals(edges)
        rows = []
        for sample in ("Train", "Test 1", "Test 2"):
            for bin_id, count, rate in rank_bins(probabilities[sample], targets[sample], edges):
                rows.append({
                    "Sample": sample,
                    "Bin": bin_id,
                    "Score interval": intervals[bin_id - 1],
                    "Accounts": comma(count),
                    "Default rate": pct(rate) if count else "",
                    "Default rate value": rate,
                })
        table = pd.DataFrame(rows)
        rank_chart(table, title, OUT / f"{stem}.png")
        save_table(OUT / f"{stem}.csv", table.drop(columns=["Default rate value"]))
        return table

    xgb_prob = {sample: books[sample]["pd"].to_numpy() for sample in books}
    xgb_y = {sample: books[sample]["target"].to_numpy() for sample in books}
    save_table(OUT / "09_xgb_final_model.csv", parameter_card_xgb())
    write_rank(xgb_prob, xgb_y, "09_xgb_rank_order", "XGBoost rank order")

    # Slides 10 and 11, then the slide 4 top-five table.
    test2_x = frames["Test 2"][features]
    print("computing SHAP on Test 2", flush=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(test2_x)
    shap_values = np.asarray(shap_values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top5 = [features[i] for i in order[:5]]
    ranking = pd.DataFrame({
        "Rank": np.arange(1, len(features) + 1),
        "Feature": [features[i] for i in order],
        "Mean absolute SHAP": [stat_text(mean_abs[i]) for i in order],
    })
    save_table(OUT / "10_xgb_shap_ranking.csv", ranking)
    stat_rows = []
    for feature in top5:
        series = test2_x[feature]
        observed = series.dropna()
        stat_rows.append({
            "Feature": feature,
            "Min": stat_text(observed.min()),
            "1st percentile": stat_text(observed.quantile(0.01)),
            "5th percentile": stat_text(observed.quantile(0.05)),
            "Median": stat_text(observed.median()),
            "95th percentile": stat_text(observed.quantile(0.95)),
            "99th percentile": stat_text(observed.quantile(0.99)),
            "Max": stat_text(observed.max()),
            "Mean": stat_text(observed.mean()),
            "Missing": pct(series.isna().mean()),
        })
    save_table(OUT / "04_top5_feature_statistics.csv", pd.DataFrame(stat_rows))

    plt.rcParams.update({"font.size": 13})
    shap.summary_plot(shap_values, test2_x, max_display=10, show=False)
    plt.gca().set_title("What moves the score on Test 2")
    finish_plot(OUT / "10_xgb_shap_beeswarm.png")
    print("saved 10_xgb_shap_beeswarm.png", flush=True)

    border = books["Test 2"].copy()
    between = border.loc[(border["pd"] >= CONSERVATIVE_T) & (border["pd"] < AGGRESSIVE_T)]
    pick_from = between if len(between) else border
    pick_index = (pick_from["pd"] - 0.43).abs().idxmin()
    row_position = border.index.get_loc(pick_index)
    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)):
        expected = np.asarray(expected).reshape(-1)[0]
    explanation = shap.Explanation(
        values=shap_values[row_position],
        base_values=float(expected),
        data=test2_x.iloc[row_position].to_numpy(),
        feature_names=features,
    )
    picked = border.loc[pick_index]
    shap.plots.waterfall(explanation, max_display=8, show=False)
    plt.gca().set_title(f"Test 2 account with predicted default probability {float(picked['pd']):.2f}")
    finish_plot(OUT / "11_xgb_shap_waterfall.png")
    local = pd.DataFrame({
        "Feature": features,
        "SHAP value": shap_values[row_position],
        "Feature value": test2_x.iloc[row_position].to_numpy(),
    }).sort_values("SHAP value", key=np.abs, ascending=False)
    top = local.head(8).copy()
    top.insert(0, "customer_ID", picked["customer_ID"])
    top.insert(1, "target", int(picked["target"]))
    top.insert(2, "pd", f"{float(picked['pd']):.3f}")
    top["SHAP value"] = top["SHAP value"].map(stat_text)
    top["Feature value"] = top["Feature value"].map(stat_text)
    save_table(OUT / "11_xgb_shap_observation.csv", top)
    print("saved waterfall", flush=True)

    # Slides 12 to 15.
    process = pd.read_csv(RESULT_DIR / "09_nn_processing.csv")
    steps = pd.DataFrame([
        ["1", "Cap", "Floor each feature at its train 1st percentile and cap it at its train 99th percentile. Empty cells do not set the fence."],
        ["2", "Standardize", "Subtract the train average and divide by the train spread. Empty cells are skipped and stay empty."],
        ["3", "Fill", "Replace the remaining empty cells with 0, which is the train average after standardization."],
    ], columns=["Step", "Action", "Rule"])
    save_table(OUT / "12_nn_processing_steps.csv", steps)
    examples = process.loc[process["feature"].isin(["B_37_Latest", "D_110_Sum_12", "S_27_Max_12"])].copy()
    if examples.empty:
        examples = process.head(3)
    examples = pd.DataFrame({
        "Feature": examples["feature"],
        "Missing": examples["train_missing_rate"].map(pct),
        "1st percentile": examples["p01"].map(stat_text),
        "99th percentile": examples["p99"].map(stat_text),
        "Train mean": examples["train_mean"].map(stat_text),
        "Train scale": examples["train_scale"].map(stat_text),
    })
    save_table(OUT / "12_nn_processing_examples.csv", examples)

    nn_model = keras.models.load_model(RESULT_DIR / "13_nn_final_model.keras")
    nn_prob = {}
    nn_y = {}
    for sample, filename in (("Train", "nn_train.csv"), ("Test 1", "nn_test1.csv"), ("Test 2", "nn_test2.csv")):
        nn_frame = pd.read_csv(SAMPLE_DIR / filename)
        nn_y[sample] = nn_frame["target"].to_numpy()
        nn_x = nn_frame.drop(columns=["customer_ID", "target"])
        nn_prob[sample] = nn_model.predict(nn_x, verbose=0).ravel()
        print(f"nn scored {sample}", flush=True)
    save_table(OUT / "15_nn_final_model.csv", parameter_card_nn())
    write_rank(nn_prob, nn_y, "15_nn_rank_order", "Neural net rank order")

    # Slide 16.
    comparison = pd.read_csv(RESULT_DIR / "14_final_model_comparison.csv")
    pretty = pd.DataFrame({
        "Model": comparison["model"].map({"xgboost": "XGBoost", "neural_net": "Neural net"}),
        "Train AUC": comparison["auc_train"].map(auc_text),
        "Test 1 AUC": comparison["auc_test1"].map(auc_text),
        "Test 2 AUC": comparison["auc_test2"].map(auc_text),
        "Average AUC": comparison["mean_auc"].map(auc_text),
        "Spread": comparison["std_auc"].map(auc_text),
        "Selection score": comparison["selection_score"].map(auc_text),
        "Chosen": comparison["chosen"].map({True: "Yes", False: "No"}),
    })
    save_table(OUT / "16_final_model_comparison.csv", pretty)

    # Slide 17 threshold grid.
    slide17 = []
    for threshold in SLIDE17_THRESHOLDS:
        row = {"Threshold": f"{threshold:.1f}"}
        for sample in ("Train", "Test 1", "Test 2"):
            accepted, default_rate, revenue = strategy_row(books[sample], threshold)
            row[f"{sample} accepted"] = comma(accepted)
            row[f"{sample} default rate"] = pct(default_rate) if accepted else ""
            row[f"{sample} revenue"] = money(revenue)
        slide17.append(row)
    save_table(OUT / "17_strategy_thresholds.csv", pd.DataFrame(slide17))

    facts = {
        "recommendation": recommendation,
        "aggressive_holds": aggressive_holds,
        "default_kept": default_kept,
        "specified_kept": specified_kept,
        "union_kept": union_kept,
        "top5": top5,
        "xgb_same": bool(xgb_same),
        "nn_same": bool(nn_same),
        "xgb_best_test_auc": float(xgb_best_test["auc_test2"]),
        "nn_best_test_auc": float(nn_best_test["auc_test2"]),
        "picked_pd": float(picked["pd"]),
        "picked_y": int(picked["target"]),
        "missing_spend": missing_spend,
        "slide1": slide1_table.drop(columns=[c for c in slide1_table.columns if c.startswith("_")]).to_dict("records"),
    }
    pd.Series({key: str(value) for key, value in facts.items()}).to_csv(OUT / "00_build_facts.csv", header=["value"])
    for name in (
        "default_rate_by_threshold.png",
        "revenue_by_threshold.png",
        "revenue_vs_default_rate.png",
    ):
        shutil.copy(PROJECT_DIR / "strategy" / name, OUT / f"17_{name}")
        print(f"saved 17_{name}", flush=True)
    print(recommendation, flush=True)
    print("top5", top5, flush=True)
    print("xgb same point", xgb_same, "nn same point", nn_same, flush=True)


if __name__ == "__main__":
    main()
