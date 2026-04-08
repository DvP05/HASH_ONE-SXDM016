# Model Card: logistic_regression

**Type**: logistic
**Task**: classification

## Intended Use

Predict and classify natural disasters (wildfires, floods, droughts) and their
impact on crops using global satellite telemetry data from NASA FIRMS, EONET,
and POWER APIs, combined with historical disaster training data from Kaggle.
Focus on identifying high-risk zones for emergency dispatch and agricultural protection.


## Performance

- CV Score: 0.9401 ± 0.0077
- ROC-AUC: 0.9430
- Accuracy: 90.0%

## Limitations

Model trained on historical data; may not generalize to distribution shifts.

## Hyperparameters

```json
{
  "max_iter": 1000,
  "random_state": 42
}
```