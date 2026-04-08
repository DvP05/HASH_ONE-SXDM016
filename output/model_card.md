# Model Card: logistic_regression

**Type**: logistic
**Task**: classification

## Intended Use

Predict 30-day customer churn for a SaaS product.
Minimize false negatives (missed churners) â€” recall
is more important than precision for this use case.


## Performance

- CV Score: 0.8288 ± 0.0055
- ROC-AUC: 0.8296
- Accuracy: 76.8%

## Limitations

Analysis complete.

## Hyperparameters

```json
{
  "max_iter": 1000,
  "random_state": 42
}
```