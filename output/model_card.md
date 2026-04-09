# Model Card: random_forest

**Type**: random
**Task**: classification

## Intended Use

Predict and classify natural disasters (wildfires, floods, droughts) and their
impact on crops using global satellite telemetry data from NASA FIRMS, EONET,
and POWER APIs, combined with historical disaster training data from Kaggle.
Focus on identifying high-risk zones for emergency dispatch and agricultural protection.


## Performance

- CV Score: 0.9380 ± 0.0084
- ROC-AUC: 0.9725
- Accuracy: 91.3%

## Limitations

Model trained on historical data; may not generalize to distribution shifts.

## Hyperparameters

```json
{
  "n_estimators": 178,
  "random_state": 42,
  "n_jobs": -1,
  "max_depth": 9,
  "min_samples_split": 8,
  "min_samples_leaf": 5
}
```