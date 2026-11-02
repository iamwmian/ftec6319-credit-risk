# 3 x 2 x 2 x 2 x 3 = 72 models. Each model is fit on train only.
TREE_COUNTS = (50, 100, 300)
LEARNING_RATES = (0.01, 0.1)
ROW_SHARES = (50, 80)       # percent of accounts in each tree
FEATURE_SHARES = (50, 100)  # percent of features in each tree
DEFAULT_WEIGHTS = (1, 5, 10)

for n_trees in TREE_COUNTS:
    for learning_rate in LEARNING_RATES:
        for row_share in ROW_SHARES:
            for feature_share in FEATURE_SHARES:
                for weight in DEFAULT_WEIGHTS:
                    model = XGBClassifier(
                        n_estimators=n_trees,
                        learning_rate=learning_rate,
                        subsample=row_share / 100,
                        colsample_bytree=feature_share / 100,
                        scale_pos_weight=weight,
                        random_state=42,
                    )
                    model.fit(X_train, y_train)
                    # AUC is recorded on train, Test 1, and Test 2.
