# Assembly instructions

Paste this file to Claude with the design template attached and with access to `CR_Project`. If folder access is not available, also attach `presentation/`, `strategy/part15.md`, and `Credit Risk Project.pdf`.

`00_slide_guide.md` stays the claim sheet. This file is the job. Do not merge them: the instructions below tell Claude to read the slide guide as evidence, so the guide cannot also be the prompt.

---

You are assembling a graded management presentation. You are not the analyst. The analysis is finished. Your job is to place the finished evidence into the attached design template so the 17 required slides read as one decision, told in order.

Do not change, round differently, recompute, reinterpret, or “improve” any number, threshold, model choice, chart, caption, or recommendation. If a sentence would require a number that is not already in the source files, leave the sentence out. If two sources disagree, stop and report the conflict. Do not pick a winner.

## What you are given

1. The attached PowerPoint design template. This is the only visual system you may use.
2. The project folder, especially:
   - Credit Risk Project.pdf (or the .docx of the same assignment). This says which artifact must appear on which slide.
   - presentation/00_slide_guide.md. This says what each slide claims. Follow it.
   - presentation/00_build_facts.csv and the numbered files 01 through 17. These are the numbers, pictures, code, and captions.
   - strategy/part15.md. This is the strategy write-up. Use its definitions and its caveats. Do not add new ones.

Source order when anything is unclear:

- Numbers, counts, rates, AUC, revenue, feature names, and chart images come from the numbered files in presentation/.
- The claim on each slide comes from presentation/00_slide_guide.md.
- Definitions of default, threshold, revenue, and the 5% versus 10% caps come from strategy/part15.md.
- Which object must physically appear on a slide (table, code, chart, attached workbook) comes from the assignment.
- The example counts printed inside the assignment’s sample tables are not our results. Never use them.

## The decision the deck is selling

Open on the decision and close on the same decision.

The firm wants more revenue from accepted accounts. It will not accept a book whose realized default rate is above 10%. Default means the account did not pay as agreed, so that account’s interest and spend revenue for the year is zero. The model ranks each application by probability of default. An account is accepted only when its probability is below the cutoff. A probability equal to the cutoff is rejected. The cutoff is not the default rate. A cutoff of 0.32 does not mean 32% of accepted accounts defaulted.

We sell the XGBoost model and the conservative strategy, threshold 0.32.

On the train sample the aggressive book, threshold 0.54, stays just under 10% and earns more. On Test 1 its default rate is 10.90%. On Test 2 it is 11.00%. The conservative book stays near 6% on both held-out samples, inside the firm’s limit. The 5% line was our tighter target while searching the train sample. The 10% line is the firm’s. Say that. Do not call the aggressive book acceptable out of sample, and do not say it “nearly” meets the limit.

Revenue is in the units of the anonymized features. It is not dollars and it is not profit. It does not subtract credit losses or operating costs. Do not add a currency symbol.

## Story, without new slides of analysis

Keep the assignment’s 17 slides, in that order, so a grader can find Slide 1 through Slide 17. You may add one title slide in front, using the template’s title layout only. Do not add section dividers, appendix slides, or extra charts. The story is the sequence and the first sentence of each slide.

Tell it in five beats. Each beat hands off to the next in one sentence. Do not announce the beats with labels such as “Act 1”.

1. Slide 1. The firm can say yes more often until defaults wipe out the revenue and break a 10% limit. We recommend the book that still holds on accounts the model has not seen.
2. Slides 2–6. Here is the book we learned from, what we could see, how 13 months became account-level signals, and which signals were strong enough to keep.
3. Slides 7–11. We searched 72 tree models, kept the stable one rather than the one with the highest Test 2 score, showed that it sorts risk from the safest bin to the riskiest, and opened one score so the audience can see what moves it.
4. Slides 12–16. The same 36 signals were given to a neural net. That net is tighter and a little less accurate. The same selection rule, average AUC minus the spread of the three AUCs, picks XGBoost.
5. Slide 17. The cutoff is a business choice on top of that score. Return to 0.32, because 0.54 does not keep the 10% limit once the model sees new accounts.

The first line of each slide states that slide’s point in plain language. The rest of the slide is the evidence the assignment requires. Speaker notes may carry one connective sentence into the next slide. The slide itself must still stand alone. Do not hide a required table, code block, chart, or caveat in the notes.

## Facts you must not smooth over

Use these as written. Do not “fix” them into a cleaner story.

- The development sample is 91,783 accounts, a 20% draw, default rate 25.65%. The split is 70% train, 15% Test 1, 15% Test 2, stratified on default, seed 42. Train has 64,248 accounts. Test 1 has 13,767. Test 2 has 13,768.
- Every training account’s newest statement in the file is March 2018, so the six-month window used here is October 2017 through March 2018. The brief calls this the April 2018 vintage and names November 2017 through April 2018. April 2018 is not in the file. Say both facts. Do not relabel the month.
- A full 13-month history is the healthier book: 77,348 accounts, default rate 22.94%. Shorter histories default at roughly 31% to 47%. We kept them. Dropping them would remove the applicants the model most needs to recognize.
- The draw has 17 raw fields: Delinquency 10, Spend 1, Payment 0, Balance 4, Risk 2. Payment was on the category list and contributed no field. Spend is S_27. The statement date was used to build history and is not one of the 17 predictors.
- Engineering produced 17 × 22 = 374 summaries: average, sum, minimum, maximum, and median over 3, 6, 9, and 12 months, plus the latest value and the year-over-year change. Selected counts are Delinquency 23, Spend 1, Payment 0, Balance 8, Risk 4.
- On Slide 4, the top five attributes are the ones later ranked most important. Name them and show their Test 2 statistics. Do not name XGBoost or SHAP until Slide 10. B_28_YoY has a very long right tail: its max is in the thousands and its median is 0.133.
- The brief names 11 categorical fields. None of them are in the 17 raw fields, so one-hot encoding created zero columns. Show that result for every row. Do not imply that encoding was skipped because it was unnecessary in principle. It created nothing because those fields were not in this draw.
- Feature selection: two XGBoost models scored all 374 summaries on train. Importance is gain, and the shares add to 100%. A feature is kept when either model gives it more than 0.5%. The default-settings model kept 28. The specified model (300 trees, learning rate 0.50, depth 4, half the rows, half the columns, default weight 5) kept 14. Six features cleared 0.5% in both. The union is 36. Six raw fields contributed nothing above the cutoff: B_32, D_46, D_94, D_104, D_143, and R_28. The picture is the slide. Attach 06_feature_importance.xlsx. Do not paste the workbook onto the slide.
- XGBoost search: 72 models, fit on train only, scored on train, Test 1, and Test 2. Tree depth stayed at XGBoost’s own default because it was not one of the dials. Seed 42 fixes the row and feature draws. The lesson in the slide guide is the lesson you may state: raising the learning rate to 0.10 lifts train AUC faster than Test 2 AUC; a default weight of 5 or 10 did not win; the winning setting is 50 trees, learning rate 0.10, half the accounts, all of the features, default weight 1. Explain the dials from 07_xgb_grid_settings.csv. Do not say a setting was chosen because the assignment listed it.
- Slide 8: the average-minus-spread plot chooses the orange point. The train-versus-Test 2 plot would choose the open blue point, which has a higher Test 2 AUC and a larger train gap. We keep the orange point. Use the sentences in 08_scatter_choice.txt. Do not let the higher Test 2 point become the chosen model.
- Final XGBoost: Train AUC 0.936, Test 1 0.922, Test 2 0.921, average 0.926, spread 0.007, selection score 0.920. Rank-order bins are cut on the train scores and applied unchanged to both test samples. The actual default rate rises from about 0.1% in the safest bin to about 90% in the riskiest bin, on all three samples. The two safest bins on Test 1 are nearly tied, at 0.37% and 0.35%. From there the bars step up.
- SHAP beeswarm is Test 2, all 13,768 accounts. A point to the right of zero pushes that account toward default. Red is a high feature value and blue is a low one. D_48_Latest is first. A high latest delinquency value pushes the score up, and a low value pushes it down. B_37_Latest is second and has the same direction. B_28_YoY is third. The top five mean absolute SHAP values are 0.888, 0.576, 0.234, 0.171, and 0.145.
- SHAP waterfall: one Test 2 account, predicted probability 0.430, and it did not default. It sits between 0.32 and 0.54, so only the aggressive strategy would accept it. The chart is on the model’s raw score. The average raw score is −1.105, about the 25.65% base rate. This account ends at −0.281, which is probability 0.43. B_37_Latest = 0.215 is the largest push toward default, +0.48. The year-over-year balance change and the latest delinquency value push the same way. S_27_Max_12 = 0.508 pulls the score down by 0.20. To improve this score, the account would need a lower latest balance and a lower latest delinquency value.
- Neural-net processing uses the same 36 features and no new one-hot columns. Fences and scale are learned on train only. Three steps, in this order: cap at the train 1st and 99th percentiles of values that are actually present; subtract the train average and divide by the train spread, leaving empty cells empty; then fill the remaining empty cells with 0, which is the train average after scaling. B_37_Latest is complete, so its train scale is the real spread, 0.223. D_110_Sum_12 is missing on 99.11% of train accounts. Its observed 1st percentile is 0.173 and its 99th is 12.07, so the cap still has a range. Filling with zero before the cap would have crushed that feature.
- Neural-net search: 32 models. Two layers and four layers are separate loops. Dropout of 100% means the dropout layer is left out, so every node is kept. It does not mean Dropout(1.0). Adam, binary cross-entropy, and 20 epochs stay fixed. Seed 42 is reset before each model. Batch size 100 beat batch size 10,000. Keeping every node beat dropping half of them. One setting — 4 layers, 6 ReLU nodes, 50% dropout, batch 10,000 — scored below 0.50 on all three samples. The winner is 2 layers, 6 ReLU nodes, no dropout, batch 100.
- On Slide 14 both plots choose that same model. Use 14_scatter_choice.txt. The average-versus-spread window is zoomed to the strongest settings. The orange point is also the best Test 2 score. The neural net’s train and Test 2 scores stay close.
- Final neural net: Train AUC 0.921, Test 1 0.917, Test 2 0.916, average 0.918, spread 0.002, selection score 0.916. The default rate rises from well under 1% in the safest bin to about 87% in the riskiest bin, on all three samples.
- Slide 16: the neural net is the tighter model. Its average is lower by about 0.008, and that gap is larger than the extra spread of the XGBoost model. XGBoost wins. Both models rank-order. The strategy uses the XGBoost probabilities. Do not recommend the neural net for being more stable. The selection score already charges XGBoost for its wider spread, and XGBoost still wins.
- Strategy inputs: balance is B_37_Ave_6, spend is S_27_Ave_6. Missing spend is set to 0 before the function runs. On the train sample that is 11,480 accounts. Monthly revenue is 2% of the six-month average balance plus 0.1% of the six-month average spend. A defaulted account adds nothing. The 12-month total is the monthly amount times 12. The 0.02 is a 24% annual interest rate written monthly. The 0.001 is a 0.1% charge on monthly spend. The brief’s dollar example may be used once, as an illustration of those two rates, and then you must say our features are not dollars.
- The conservative threshold, 0.32, is the highest-revenue point on the train grid whose train default rate is still at or below 5%. The next grid step, 0.33, is 5.03%. The aggressive threshold, 0.54, is the highest-revenue point still at or below 10%. The next grid step, 0.55, is 10.16%. Thresholds 0.32 and 0.54 sit between the rows of the 0.1-to-0.8 table. Slide 1 has those two strategies on all three samples. Slide 17 has the 0.1-to-0.8 table the brief asks for.
- Copy every figure in 01_strategy_results.csv and 17_strategy_thresholds.csv exactly, including thousands separators and two-decimal rates and revenues.

## Template rules

Inspect the attached template before you write a slide. Use its existing layouts, theme colors, fonts, logo, and footer. Do not create a new theme, a new palette, or new layouts.

Map each required slide onto the closest existing layout: title, one chart, two charts, table, code, or comparison. If the type would be too small, split that one required topic into 8a and 8b, or 14a and 14b, and say so. Do not split any other slide.

Do not paint a new background, gradient, or photograph. Keep whatever background the template master already has. Do not restyle the PNG charts. Insert them as images. Their orange point, open blue point, axis scales, and captions are part of the finding.

Tables use the template’s table style if it has one. Keep them readable. A table that does not fit at a readable size is split across the continuation of that same numbered slide, not squeezed. Counts use thousands separators. Rates use two decimals. AUC and other small scores use three decimals, as already printed in the CSVs. Do not re-round.

Code on Slides 5, 7, 13, and 17 comes from these files, unchanged:

- presentation/05_one_hot_code.py
- presentation/07_xgb_grid_code.py
- presentation/13_nn_grid_code.py
- presentation/17_strategy_function.py

Set the code in a monospaced face already in the template, or in Menlo or Consolas if the template has none. Do not rewrite the code into pseudocode. Do not paste a longer script from elsewhere in the project.

One idea per slide. No small type. No walls of bullets. No text copied from a notebook. Titles are claims, not labels. “XGBoost grid search” is a label. “A short model beat a heavier one” is a claim, and only if the slide guide supports it.

Charts and tables get the titles already implied by the slide guide. Do not invent a punchier title that changes the claim.

## Build order

1. Read the template layouts.
2. Read presentation/00_slide_guide.md from start to finish.
3. Read strategy/part15.md.
4. Read each numbered file before building that slide.
5. Build Slides 1 through 17 in order.
6. Check the finished deck against the list below before you deliver it.

## Check before you deliver

Confirm each of these is true. If one is false, fix the slide. Do not fix it by changing the evidence.

- Slide 1 recommends conservative 0.32, shows both strategies on train, Test 1, and Test 2, and says the aggressive book breaks 10% out of sample.
- Slide 2 uses our history table, including 91,783 and 25.65%, not the sample counts printed in the assignment.
- Slide 2 says March 2018 is the month in the file and that the brief calls the vintage April 2018.
- Slide 3 shows Payment as 0 fields.
- Slide 4 shows 374 engineered features and the top-five statistics without naming the model.
- Slide 5 shows 11 categorical names, zero columns, and the one-hot code.
- Slide 6 shows the feature-selection picture, states 28, 14, and 36, and attaches the workbook instead of pasting it.
- Slides 7 and 13 include the grid code and the lesson from the slide guide, including the dropout meaning and the setting that scored below 0.50.
- Slides 8 and 14 use the supplied scatter images and the supplied caption sentences. The chosen XGBoost point is the orange point, not the open blue point.
- Slides 9 and 15 show the final settings, the three AUCs, and the supplied rank-order chart. Bins were cut on train and applied to both tests.
- Slides 10 and 11 use the supplied SHAP images and the account with probability 0.430 that did not default.
- Slide 12 shows the three processing steps in order and the reason empty cells were filled after scaling.
- Slide 16 chooses XGBoost with selection scores 0.920 and 0.916, and says the strategy uses XGBoost probabilities.
- Slide 17 shows the strategy function, the 0.1-to-0.8 table, and the same recommendation as Slide 1.
- No dollar sign was added. No result was described as profit. No threshold was described as a default rate.
- No chart was redrawn. No CSV value was replaced by a value from the assignment’s example tables.

Deliver the PowerPoint file only. Do not also deliver a rewritten analysis, a new CSV, or a memo that restates the findings in different numbers.
