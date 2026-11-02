# Slide guide

White background, one idea per slide, and the file named with that slide's number. Counts use thousands separators. Rates use two decimals. AUC and other small scores use three decimals. Revenue is in the units of the anonymized features. It is not a dollar figure.

The model we sell is the XGBoost model. The strategy we sell is the conservative one, threshold 0.32.

## Slide 1. Executive summary

Files: `01_strategy_results.csv`, `01_recommendation.txt`.

The firm wants more revenue from accepted accounts, and it will not accept a book whose default rate is above 10%. Default here means the account did not pay as agreed, so its interest and spend revenue for the year is zero. The model ranks every application by its probability of default. The threshold is the highest probability the firm is willing to accept.

| Strategy | Threshold | Train accepted | Train default rate | Train revenue | Test 1 accepted | Test 1 default rate | Test 1 revenue | Test 2 accepted | Test 2 default rate | Test 2 revenue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Conservative | 0.32 | 42,729 | 4.87% | 541.72 | 9,159 | 5.80% | 115.12 | 9,161 | 5.95% | 114.28 |
| Aggressive | 0.54 | 49,207 | 9.86% | 766.53 | 10,566 | 10.90% | 162.03 | 10,584 | 11.00% | 161.14 |

Propose the conservative strategy. On the train sample the aggressive book stays just under 10% and earns more. On both held-out samples it goes through 10%, to 10.90% and 11.00%. The conservative book stays near 6% out of sample, inside the firm's limit. The 5% line was our tighter target while we searched on the train sample. The test samples land a little above 5% and remain far below 10%.

## Slide 2. Data

File: `02_history_length.csv`.

The development sample is 91,783 accounts, a 20% draw, with a default rate of 25.65%. Each account is scored at its latest statement. In this file that month is March 2018 for every training account, so the six-month balance and spend window runs from October 2017 through March 2018. The brief calls this the April 2018 vintage. We use the month the file actually contains.

A full 13-month history is the healthy book: 77,348 accounts, default rate 22.94%. Accounts with a shorter history default at 31% to 47%. We kept them. Dropping them would remove the applicants the model most needs to recognize.

| History | Accounts | Default rate |
| --- | --- | --- |
| All applications | 91,783 | 25.65% |
| 13 months | 77,348 | 22.94% |
| 12 months | 2,115 | 37.87% |
| 11 months | 1,159 | 44.09% |
| 10 months | 1,329 | 46.58% |
| 9 months | 1,278 | 43.58% |
| 8 months | 1,169 | 45.08% |
| 7 months | 1,046 | 41.49% |
| 6 months | 1,109 | 41.30% |
| 5 months | 933 | 39.44% |
| 4 months | 938 | 43.07% |
| 3 months | 1,158 | 35.75% |
| 2 months | 1,217 | 31.22% |
| 1 month | 984 | 33.13% |

The split is 70% train, 15% Test 1, 15% Test 2, stratified on default, seed 42. Train has 64,248 accounts. Test 1 has 13,767. Test 2 has 13,768.

## Slide 3. Features

Files: `03_raw_feature_categories.csv`, `03_raw_features.csv`.

These are the 17 raw fields in the development sample, before any summary was built. Payment was on the category list and contributed no field to this draw.

| Category | Features |
| --- | --- |
| Delinquency | 10 |
| Spend | 1 |
| Payment | 0 |
| Balance | 4 |
| Risk | 2 |

The names are in `03_raw_features.csv`. Spend is `S_27`. The statement date was used to build history and is not one of these 17 predictors.

## Slide 4. Feature engineering

Files: `04_engineered_feature_categories.csv`, `04_statistic_types.csv`, `04_top5_feature_statistics.csv`.

Each raw field became an average, sum, minimum, maximum, and median over 3, 6, 9, and 12 months, plus the latest value and the year-over-year change. That is 17 × 22 = 374 summaries.

| Category | Engineered features | Selected for modeling |
| --- | --- | --- |
| Delinquency | 220 | 23 |
| Spend | 22 | 1 |
| Payment | 0 | 0 |
| Balance | 88 | 8 |
| Risk | 44 | 4 |

| Statistic | Engineered features | Selected for modeling |
| --- | --- | --- |
| Average | 68 | 9 |
| Sum | 68 | 2 |
| Minimum | 68 | 4 |
| Maximum | 68 | 7 |
| Median | 68 | 8 |
| Latest | 17 | 4 |
| Year over year | 17 | 2 |

The top five attributes below are the ones our later analysis ranked as most important. Say that, and leave the model name for slide 10. Statistics are on Test 2.

| Feature | Min | 1st | 5th | Median | 95th | 99th | Max | Mean | Missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| D_48_Latest | -0.009 | 0.001 | 0.013 | 0.310 | 0.955 | 1.05 | 1.98 | 0.406 | 12.46% |
| B_37_Latest | -0.473 | 0.000 | 0.002 | 0.033 | 0.656 | 1.07 | 1.33 | 0.136 | 0.00% |
| B_28_YoY | -1.000 | -0.927 | -0.742 | 0.133 | 5.33 | 23.82 | 2,151.49 | 1.74 | 13.69% |
| D_48_Ave_12 | -0.009 | 0.006 | 0.021 | 0.283 | 0.910 | 0.976 | 1.94 | 0.371 | 6.71% |
| D_48_Ave_3 | -0.010 | 0.003 | 0.016 | 0.292 | 0.946 | 1.02 | 1.97 | 0.393 | 9.32% |

`B_28_YoY` has a very long right tail. That is why its max is in the thousands while its median is 0.133.

## Slide 5. One-hot encoding

Files: `05_categorical_treatment.csv`, `05_one_hot_code.py`.

The brief names 11 categorical fields. None of them were in the 17 raw fields we drew, so one-hot encoding created zero columns. Put the short code on the slide and the table under it. Every row is the same result: not in the sample, zero columns created.

## Slide 6. Feature selection

Files: `06_feature_selection.png`, `06_feature_importance.xlsx`. The selected counts are the last column of the slide 4 category table.

Two XGBoost models scored all 374 summaries on the train sample. Importance is gain, and the shares add to 100%. A feature is kept when either model gives it more than 0.5%.

The default-settings model kept 28 features. The specified model, 300 trees, learning rate 0.50, depth 4, half the rows, half the columns, default weight 5, kept 14. Six features cleared 0.5% in both models. The union is 36 features. Six raw fields contributed nothing above the cutoff: `B_32`, `D_46`, `D_94`, `D_104`, `D_143`, and `R_28`.

The Excel file has one sheet per model and a third sheet for the 36 kept features. Attach it. Do not paste it onto the slide. The picture is the slide.

## Slide 7. XGBoost grid search

Files: `07_xgb_grid_settings.csv`, `07_xgb_grid_code.py`.

Seventy-two models. Each one was fit on the train sample only, then scored on train, Test 1, and Test 2. The table of settings explains why each dial is in the search. Tree depth stayed at XGBoost's own default because it was not one of the dials. Seed 42 fixes the row and feature draws. The results were saved after every model, so a stopped run could continue.

What the search taught us: raising the learning rate to 0.10 lifts the train AUC faster than the Test 2 AUC. Giving a default a weight of 5 or 10 did not win. The winning setting is a short model, 50 trees, learning rate 0.10, half the accounts, all of the features, and a default weight of 1.

## Slide 8. Choosing the XGBoost model

Files: `08_xgb_mean_vs_spread.png`, `08_xgb_train_vs_test2.png`, `08_scatter_choice.txt`.

Put the two plots side by side, or on two slides if the type gets small. The caption file has the sentence for each plot.

The average-minus-spread plot chooses the orange point. The train-versus-Test 2 plot would choose the open blue point, which has a higher Test 2 AUC and a larger train gap. We keep the orange point.

## Slide 9. Final XGBoost model

Files: `09_xgb_final_model.csv`, `09_xgb_rank_order.png`, `09_xgb_rank_order.csv`.

| Item | Value |
| --- | --- |
| Trees | 50 |
| Learning rate | 0.10 |
| Accounts per tree | 50% |
| Features per tree | 100% |
| Weight of a default | 1 |
| Train AUC | 0.936 |
| Test 1 AUC | 0.922 |
| Test 2 AUC | 0.921 |
| Average AUC | 0.926 |
| Spread of AUC | 0.007 |
| Selection score | 0.920 |

The bars use ten bins cut on the train scores. The same cuts are applied to Test 1 and Test 2. The actual default rate rises from about 0.1% in the safest bin to about 90% in the riskiest bin, on all three samples. The two safest bins on Test 1 are nearly tied, at 0.37% and 0.35%. From there the bars step up.

## Slide 10. SHAP beeswarm

Files: `10_xgb_shap_beeswarm.png`, `10_xgb_shap_ranking.csv`.

Test 2, all 13,768 accounts. A point to the right of zero pushes this account toward default. Red is a high feature value and blue is a low one.

`D_48_Latest` is first. A high latest delinquency value pushes the score up, and a low value pushes it down. `B_37_Latest` is second and has the same direction. `B_28_YoY` is third. The ranking file has all 36 features. The top five mean absolute SHAP values are 0.888, 0.576, 0.234, 0.171, and 0.145.

## Slide 11. SHAP waterfall

Files: `11_xgb_shap_waterfall.png`, `11_xgb_shap_observation.csv`.

One Test 2 account, predicted probability 0.430, and it did not default. It sits between the two strategy thresholds, so only the aggressive strategy would accept it. The chart is on the model's raw score. The average raw score is -1.105, which is about the 25.65% base rate. This account ends at -0.281, which is probability 0.43.

The latest balance, `B_37_Latest` = 0.215, is the largest push toward default, +0.48. The year-over-year balance change and the latest delinquency value push the same way. A higher 12-month peak spend, `S_27_Max_12` = 0.508, pulls the score back down by 0.20. To improve this score, the account would need a lower latest balance and a lower latest delinquency value.

## Slide 12. Neural net data processing

Files: `12_nn_processing_steps.csv`, `12_nn_processing_examples.csv`.

The same 36 features. No new one-hot columns. Three steps, and the fences and the scale are learned on the train sample only.

1. Cap each feature at the train 1st and 99th percentiles of the values that are actually present.
2. Subtract the train average and divide by the train spread. Empty cells stay empty.
3. Fill the remaining empty cells with 0. After the scaling, 0 is the train average.

`B_37_Latest` is complete, so its train scale is the real spread, 0.223. `D_110_Sum_12` is missing on 99.11% of train accounts. Its observed 1st percentile is 0.173 and its 99th is 12.07, so the cap still has a range. Filling with zero before the cap would have crushed that feature.

## Slide 13. Neural net grid search

Files: `13_nn_grid_settings.csv`, `13_nn_grid_code.py`.

Thirty-two models. Two layers and four layers are separate loops, because they are different shapes. Dropout of 100% means the dropout layer is left out, so every node is kept. It does not mean `Dropout(1.0)`, which would drop every node. Adam, binary cross-entropy, and 20 epochs stay fixed. Seed 42 is reset before each model. The table was saved after every model.

What the search taught us: batch size 100 beat batch size 10,000. Keeping every node beat dropping half of them. One setting, 4 layers, 6 ReLU nodes, 50% dropout, batch 10,000, scored below 0.50 on all three samples. The winner is the small net: 2 layers, 6 ReLU nodes, no dropout, batch 100.

## Slide 14. Choosing the neural net

Files: `14_nn_mean_vs_spread.png`, `14_nn_train_vs_test2.png`, `14_scatter_choice.txt`.

Both plots choose the same model. The average-versus-spread window is zoomed to the strongest settings so the winning line stays visible. The train-versus-Test 2 plot shows that this net's two scores stay close, and that the orange point is the best Test 2 score as well.

## Slide 15. Final neural net

Files: `15_nn_final_model.csv`, `15_nn_rank_order.png`, `15_nn_rank_order.csv`.

| Item | Value |
| --- | --- |
| Hidden layers | 2 |
| Nodes in each layer | 6 |
| Activation | ReLU |
| Dropout | None |
| Batch size | 100 |
| Epochs | 20 |
| Optimizer | Adam |
| Loss | Binary cross-entropy |
| Train AUC | 0.921 |
| Test 1 AUC | 0.917 |
| Test 2 AUC | 0.916 |
| Average AUC | 0.918 |
| Spread of AUC | 0.002 |
| Selection score | 0.916 |

The bins are cut on the train probabilities and applied to both test samples. The default rate rises from well under 1% in the safest bin to about 87% in the riskiest bin, on all three samples.

## Slide 16. Final model

File: `16_final_model_comparison.csv`.

| Model | Train AUC | Test 1 AUC | Test 2 AUC | Average AUC | Spread | Selection score | Chosen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.936 | 0.922 | 0.921 | 0.926 | 0.007 | 0.920 | Yes |
| Neural net | 0.921 | 0.917 | 0.916 | 0.918 | 0.002 | 0.916 | No |

The same rule picks the winner inside each grid and between the two finalists: average of the three AUCs, minus the spread of those three. The neural net is the tighter model. Its average is lower by about 0.008, and that gap is larger than the extra spread of the XGBoost model. XGBoost wins. Both models rank-order. The strategy in the next slide uses the XGBoost probabilities.

## Slide 17. Strategy

Files: `17_strategy_function.py`, `17_strategy_thresholds.csv`, `17_default_rate_by_threshold.png`, `17_revenue_by_threshold.png`, `17_revenue_vs_default_rate.png`.

An account is accepted when its predicted probability is below the threshold. Monthly revenue is 2% of the six-month average balance plus 0.1% of the six-month average spend. A defaulted account adds nothing. The 12-month total is that monthly amount times 12. Balance is `B_37_Ave_6`. Spend is `S_27_Ave_6`, and a missing spend is treated as zero before this function runs. On the train sample that is 11,480 accounts.

The pictures search every threshold from 0.01 to 0.99 on the train sample. The conservative threshold, 0.32, is the highest-revenue point on that grid whose train default rate is still at or below 5%. The next grid step, 0.33, is 5.03%. The aggressive threshold, 0.54, is the highest-revenue point still at or below the firm's 10%. The next grid step, 0.55, is 10.16%. The 5% line is our choice. The 10% line is the firm's.

The table the brief asks for, evaluated at the same thresholds on all three samples:

| Threshold | Train accepted | Train default rate | Train revenue | Test 1 accepted | Test 1 default rate | Test 1 revenue | Test 2 accepted | Test 2 default rate | Test 2 revenue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 34,488 | 1.54% | 308.78 | 7,346 | 2.12% | 67.01 | 7,345 | 2.16% | 66.46 |
| 0.2 | 39,039 | 3.02% | 418.47 | 8,370 | 3.99% | 90.03 | 8,373 | 4.06% | 90.53 |
| 0.3 | 42,140 | 4.52% | 521.58 | 9,037 | 5.61% | 110.38 | 9,043 | 5.64% | 110.23 |
| 0.4 | 45,076 | 6.38% | 627.83 | 9,683 | 7.51% | 131.80 | 9,697 | 7.69% | 132.91 |
| 0.5 | 48,018 | 8.79% | 727.77 | 10,336 | 9.89% | 155.32 | 10,339 | 10.00% | 154.05 |
| 0.6 | 51,110 | 11.67% | 824.01 | 10,948 | 12.19% | 175.19 | 11,010 | 12.84% | 173.73 |
| 0.7 | 54,807 | 15.35% | 921.50 | 11,707 | 15.66% | 194.95 | 11,747 | 15.84% | 197.32 |
| 0.8 | 59,295 | 19.98% | 1,008.92 | 12,654 | 19.99% | 214.98 | 12,695 | 20.20% | 217.70 |

Thresholds 0.32 and 0.54 sit between these rows. Slide 1 has those two on all three samples. Close the deck with the same recommendation: 0.32, because 0.54 does not keep the 10% limit once the model sees new accounts.
