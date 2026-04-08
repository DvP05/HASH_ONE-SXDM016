# Model Card: random_forest

**Type**: random
**Task**: classification

## Intended Use

Predict the confidence of active fires based on NASA FIRMS telemetry data.
Focus on identifying high-confidence active fire vectors for emergency dispatch.


## Performance

- CV Score: 1.0000 ± 0.0000
- ROC-AUC: 1.0000
- Accuracy: 100.0%

## Limitations

Model trained on historical data; may not generalize to distribution shifts.

## Hyperparameters

```json
{
  "n_estimators": 100,
  "random_state": 42,
  "n_jobs": -1
}
```