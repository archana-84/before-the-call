# Before the Call — Model Report

## Predicting campaign response before outreach

## 1. Executive summary

Before the Call is an end-to-end data science project using the UCI Bank Marketing dataset. It investigates whether historical term-deposit subscription responses can be ranked using information available before a marketing call.

The project deliberately excludes call duration and other information that would not be available at the proposed decision time. Models are evaluated chronologically rather than with a random split because the dataset is documented as ordered over time and exhibits substantial temporal change.

The selected model was a regularized logistic regression excluding economic indicators, followed by sigmoid probability calibration. It was selected and committed before the final test set was evaluated.

The model showed useful ranking performance during validation, but weak generalization in the later untouched test period:

- Test response prevalence: 44.50%
- Average precision: 0.4782
- No-skill average-precision reference: 0.4450
- Mean calibrated prediction: 21.88%
- Calibration gap: −22.62 percentage points
- Lift at 10% capacity: 0.97
- Lift at 20% capacity: 1.10

The final conclusion is that the model is not deployment-ready. The project is valuable as a demonstration of leakage prevention, chronological validation, capacity-based evaluation, probability calibration, distribution-shift detection, and responsible reporting.

## 2. Intended use

The intended analytical question is:

> Before a scheduled marketing call, can historical observations be ranked by their likelihood of ending in a term-deposit subscription when contact capacity is limited?

The project supports a retrospective demonstration of how a fixed contact capacity changes which historical observations would be prioritized.

The model is not intended for:

- estimating the causal effect of making a call;
- claiming campaign uplift or increased revenue;
- making current banking or financial decisions;
- deployment in the current United States market;
- automatically excluding people from access to financial products;
- making conclusions about individuals based on protected or demographic attributes.

## 3. Data source and historical context

Source: [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)

Selected variant:

- `bank-additional-full.csv`
- 41,188 campaign observations
- 20 input columns and one target
- Portuguese bank direct-marketing campaigns
- May 2008 through November 2010
- Rows documented as ordered chronologically

The selected “additional full” dataset includes client attributes, current campaign scheduling fields, previous campaign history, and economic context variables.

The data represents historical Portuguese banking campaigns. It does not establish current customer behavior, current economic conditions, or current US-market performance.

## 4. Unit of analysis and target

### Unit of analysis

One row represents one historical campaign/contact observation.

The data does not include a verified customer identifier. Therefore:

- rows cannot be confirmed as unique customers;
- repeat contacts may represent the same person;
- customer-level grouping cannot be enforced;
- repeated-customer leakage cannot be ruled out.

### Target

The target is `y`:

- `yes`: the historical observation ended in subscription;
- `no`: the historical observation did not end in subscription.

Overall target distribution:

| Outcome | Observations | Share |
|---|---:|---:|
| No | 36,548 | 88.73% |
| Yes | 4,640 | 11.27% |

## 5. Decision time

The proposed decision time is immediately before a scheduled call, after the contact channel and planned call date are assumed to be known, but before the conversation begins.

This decision-time definition controls which variables are eligible for modeling.

## 6. Leakage audit

### Excluded fields

| Feature | Reason |
|---|---|
| `duration` | Call duration is only known after the call occurs and is direct future-information leakage. |
| `campaign` | Its meaning at the exact scoring moment is ambiguous because it counts contacts during the current campaign. |
| `y` | This is the post-campaign outcome and prediction target. |

### Selected final-model fields

The selected no-economic model uses 13 source features:

- Client attributes: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
- Scheduling context: `contact`, `month`, `day_of_week`
- Previous campaign history: `pdays`, `previous`, `poutcome`

The following economic variables were evaluated but removed from the selected model:

- `emp.var.rate`
- `cons.price.idx`
- `cons.conf.idx`
- `euribor3m`
- `nr.employed`

All five economic variables moved outside their training ranges during validation. Although they improved some ranking metrics, they produced extremely unstable probability estimates under temporal extrapolation.

## 7. Data validation

The project validated:

- expected row and column counts;
- documented column names and order;
- allowed categorical values;
- numeric ranges;
- missing and infinite values;
- target categories;
- relationships among `previous`, `pdays`, and `poutcome`;
- exact duplicate rows.

Findings:

- Conventional null cells: 0
- Infinite numeric values: 0
- Exact duplicate rows: 12
- Duplicate rows were retained because no customer or contact identifier exists to prove that they are erroneous duplicates.
- The string `unknown` was retained as semantic missingness rather than converted to a conventional null.
- `pdays = 999` was not treated alone as proof of no previous contact because 4,110 such rows also recorded previous contacts.

## 8. Chronological splitting strategy

The source documentation states that rows are ordered chronologically. Exact dates are unavailable, so month-level periods were inferred from row ordering and documented campaign coverage.

| Partition | Approximate period | Observations | Subscribers | Response rate | Purpose |
|---|---|---:|---:|---:|---|
| Training | 2008-05 to 2008-11 | 27,680 | 1,338 | 4.83% | Fit preprocessing and base models |
| Validation | 2008-12 to 2009-05 | 8,544 | 1,093 | 12.79% | Compare models, features, and calibration |
| Test | 2009-06 to 2010-11 | 4,964 | 2,209 | 44.50% | One-time final evaluation |

The test boundary was locked before model comparison. Model and calibration choices were committed before the test set was opened.

The very large change in response prevalence is evidence of temporal distribution shift. A random split would mix these campaign regimes and produce an easier but less realistic evaluation.

## 9. Preprocessing

Preprocessing was fitted only on training data.

Categorical features were encoded using:

```python
OneHotEncoder(handle_unknown="ignore")

```

This setting allows categories that appear later but were not observed during training to be processed without refitting the encoder.

Numeric features were standardized using `StandardScaler`.

The final base model contains 52 encoded features derived from 13 source features.

## 10. Models compared

The project compared:

1. Training-prevalence dummy baseline
2. Unweighted logistic regression
3. Class-balanced logistic regression
4. Unweighted random forest
5. Class-balanced random forest

Four feature policies were also compared:

1. Full scheduled pre-call features
2. No economic indicators
3. No scheduling fields
4. Core client and previous-campaign fields only

Model selection considered:

- average precision;
- precision and recall at 10% and 20% capacity;
- lift at capacity;
- Brier score;
- probability calibration;
- temporal robustness;
- interpretability.

## 11. Validation results

### Baseline

The training-prior baseline assigned 4.83% to every validation observation.

- Validation prevalence: 12.79%
- Average precision: 0.1279
- Brier score: 0.1179
- Calibration gap: −7.96 percentage points
- Ranking lift: 1.00 by definition

### Full scheduled logistic model

The unweighted full logistic model produced the strongest overall validation ranking:

| Metric | Result |
|---|---:|
| Average precision | 0.2386 |
| Precision at 10% | 29.36% |
| Recall at 10% | 22.96% |
| Lift at 10% | 2.29 |
| Precision at 20% | 24.28% |
| Recall at 20% | 37.97% |
| Lift at 20% | 1.90 |

However, its mean validation prediction was only 0.05%. The model retained some ranking ability, but its probability estimates collapsed because several economic indicators moved completely outside their training ranges.

### No-economic logistic model

| Metric | Result |
|---|---:|
| Average precision | 0.2244 |
| Precision at 10% | 26.67% |
| Recall at 10% | 20.86% |
| Lift at 10% | 2.08 |
| Precision at 20% | 24.69% |
| Recall at 20% | 38.61% |
| Lift at 20% | 1.93 |
| Brier score | 0.1138 |

The no-economic policy sacrificed some top-10% validation performance but produced more stable probability behavior and the strongest top-20% result.

## 12. Calibration

Calibration methods were compared using a chronological subdivision of validation:

- Earlier 60%: calibration fitting
- Later 40%: calibration assessment

The later assessment period had a 9.89% response rate.

For the no-economic model:

| Method | Mean prediction | Calibration gap | AP | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| Uncalibrated | 3.63% | −6.26 pp | 0.1371 | 0.0925 | 0.3581 |
| Sigmoid | 8.31% | −1.58 pp | 0.1371 | 0.0886 | 0.3209 |
| Isotonic | 8.01% | −1.87 pp | 0.1263 | 0.0888 | 0.3217 |

Sigmoid calibration was selected because it improved the Brier score, log loss, and calibration gap while preserving ranking.

After selecting the method, the final sigmoid calibrator was fitted on the complete validation partition. Metrics calculated on that same complete validation partition are in-sample calibration metrics and are not presented as generalization estimates.

## 13. Locked final model

The final model was locked before test evaluation:

- Base model: unweighted regularized logistic regression
- Feature policy: no economic indicators
- Calibration method: sigmoid
- Base-model fitting data: training partition
- Calibration-fitting data: validation partition
- Test data used during selection or fitting: no
- Operational decision: rank observations and select the highest scores allowed by capacity
- Fixed probability threshold: none

The model-selection record is stored at:

```text
data/exports/final_model_selection.json
```

## 14. Untouched test performance

The untouched test period contained 4,964 observations and had a 44.50% response rate.

### Probability performance

| Variant | Mean prediction | Calibration gap | AP | Brier | Log loss |
|---|---:|---:|---:|---:|---:|
| Uncalibrated | 12.71% | −31.79 pp | 0.4782 | 0.3811 | 1.3032 |
| Sigmoid calibrated | 21.88% | −22.62 pp | 0.4782 | 0.3438 | 1.0261 |

Calibration improved the Brier score and log loss but did not overcome the large later-period shift.

The calibrated average precision of 0.4782 was only modestly above the no-skill reference of 0.4450.

### Capacity performance

| Capacity | Selected | Selected subscribers | Precision | Recall | Lift |
|---|---:|---:|---:|---:|---:|
| 10% | 497 | 214 | 43.06% | 9.69% | 0.97 |
| 20% | 993 | 484 | 48.74% | 21.91% | 1.10 |

At 10% capacity, the model performed slightly worse than random ranking. At 20%, it performed only modestly better.

These results do not support operational deployment.

## 15. Error analysis

Errors were defined in relation to the capacity decision:

- **False prioritization:** a selected historical non-subscriber
- **Missed subscriber:** a historical subscriber outside the selected capacity

| Capacity | False prioritizations | Missed subscribers |
|---|---:|---:|
| 10% | 283 | 1,995 |
| 20% | 509 | 1,725 |

A capacity limit necessarily excludes many observations. Therefore, a missed subscriber is not automatically an avoidable model mistake. The relevant question is whether the model concentrates substantially more subscribers inside the selected group than random ranking. On the test period, it generally did not.

## 16. Group diagnostics

### Contact channel

| Channel | Observations | Response rate | Mean prediction | Calibration gap | AP | Top-20% selection rate |
|---|---:|---:|---:|---:|---:|---:|
| Cellular | 4,265 | 47.15% | 22.17% | −24.98 pp | 0.5025 | 20.70% |
| Telephone | 699 | 28.33% | 20.11% | −8.21 pp | 0.2998 | 15.74% |

The model underpredicted both channels, especially cellular observations.

These differences may reflect campaign timing, channel assignment, and other confounding variables. They do not prove that one contact channel causes a higher subscription rate.

### Age

Observed test response rates were relatively similar across age groups, while predictions were substantially lower for every group. This indicates broad temporal underprediction rather than a calibration problem isolated to one age band.

Demographic analysis is descriptive only. It is not a causal explanation or formal fairness certification.

## 17. Feature contributions

The final model contains 52 encoded coefficients.

The largest positive coefficient was `month = oct`. However, October contained only 67 training observations and had an unusually high training response rate. This coefficient was unstable and did not transfer to the test period, where October was dramatically overpredicted.

Other large coefficients included rare categories such as:

- `marital = unknown`
- `education = illiterate`
- `job = unknown`

Coefficient interpretation boundaries:

- coefficients are conditional model associations;
- numeric coefficients represent a one-training-standard-deviation increase;
- one-hot coefficients are regularized category associations;
- magnitude is not proof of stability;
- odds ratios are not causal effects;
- demographic coefficients are not evidence of individual behavior.

## 18. Why performance weakened

The principal causes were:

1. **Large prevalence shift**  
   Response increased from 4.83% in training to 44.50% in testing.

2. **Feature distribution shift**  
   Campaign month, contact channel, previous-campaign history, and economic context changed substantially.

3. **Unseen and rare categories**  
   Some later campaign months were absent or rare during training.

4. **Campaign-period overfitting**  
   The model learned strong month effects that did not transfer reliably.

5. **Limited historical coverage**  
   Training represented only an early portion of a changing multi-year campaign.

6. **Missing customer identifiers**  
   Customer histories, repeated contacts, and grouped customer-level validation could not be constructed.

7. **Calibration cannot repair ranking drift**  
   Calibration changes the probability scale. It cannot restore relationships that have changed over time.

## 19. Deployment recommendation

Do not deploy the current model.

A future production candidate would require:

- recent and representative training data;
- verified customer and contact identifiers;
- exact timestamps;
- clear feature-availability contracts;
- rolling or walk-forward validation;
- comparison against simple operational rules;
- monitoring for prevalence, feature, ranking, and calibration drift;
- documented recalibration and retraining rules;
- subgroup monitoring;
- human review and governance.

If the decision question is whether contacting someone changes their probability of subscribing, a randomized experiment or credible uplift-modeling design would be required.

## 20. Reproducibility controls

The repository includes:

- official-source download and checksum validation;
- schema and data-quality checks;
- a documented chronological split;
- a feature leakage audit;
- training-only preprocessing;
- baseline and model comparisons;
- temporal calibration assessment;
- a model-selection manifest committed before testing;
- untouched final test evaluation;
- Python and SQL capacity calculations;
- automated tests;
- a historical Streamlit capacity simulator.

The automated tests verify:

- leakage exclusions;
- final feature-policy composition;
- pre-test model locking;
- demo-data integrity;
- exact top-10% and top-20% results;
- exported probability metrics;
- agreement between Python and SQL outputs.

## 21. Final conclusion

Before the Call demonstrates that strong validation performance is not sufficient evidence for deployment.

The project’s most important result is not that a particular model “won.” It is that a careful chronological test exposed a failure that a random split or leakage-prone feature set could easily hide.

The final model is useful for education, retrospective analysis, and demonstrating responsible model evaluation. It is not suitable for live campaign prioritization without substantially better and more recent data.