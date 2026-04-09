# Autonomous Analysis — Insight Report

*Generated at: 2026-04-09T05:16:24.187342+00:00*

---

## Executive Summary

Analysis of Predict and classify natural disasters (wildfires, floods, droughts) and their
impact on crops using global satellite telemetry data from NASA FIRMS, EONET,
and POWER APIs, combined with historical disaster training data from Kaggle.
Focus on identifying high-risk zones for emergency dispatch and agricultural protection.
 is complete. The random_forest model achieves ROC-AUC = 0.973. Accuracy: 91.3%. Cross-validation score: 0.9380 ± 0.0084.

---

## Data Quality

**2 cleaning operations** were applied:

- **fill_numeric_missing**: Fill missing numeric values with column median. (0 rows affected)
- **cap_outliers_iqr**: Cap extreme outliers using IQR method. (0 rows affected)

**Dataset**: 9,048 rows × 38 columns
**Average missing rate**: 2.7%
**Class balance**: {'l': 0.6737, 'n': 0.2282, 'h': 0.098}

---

## Key Patterns

Dataset contains 9048 rows and 38 columns. Average missing rate is 2.7%.


---

## Model Performance

**Selected model**: random_forest

**Cross-validation**: 0.9380 ± 0.0084

**Metrics**: Accuracy: 91.3% | ROC-AUC: 0.9725 | F1-Score: 0.9111 | Precision: 0.9109 | Recall: 0.9129

### Top Predictive Features

1. **frp** (importance: 0.3294)
2. **frp_x_severity** (importance: 0.2875)
3. **brightness** (importance: 0.1607)
4. **crop_yield_loss_pct** (importance: 0.0751)
5. **precipitation** (importance: 0.0266)
6. **humidity** (importance: 0.0243)
7. **soil_moisture** (importance: 0.0237)
8. **month** (importance: 0.0182)
9. **temperature** (importance: 0.0148)
10. **wind_speed** (importance: 0.0131)

### Model Comparison

| Model | CV Score |
|-------|---------|
| random_forest | 0.9336 ± 0.0098 |
| lightgbm | 0.9330 ± 0.0078 |
| xgboost | 0.9323 ± 0.0081 |

---

## Recommendations


### [HIGH] Deploy Interventions for Top Risk Indicators

The model identifies frp, frp_x_severity, brightness as the strongest risk predictors. Prioritize immediate preventative measures and allocate disaster mitigation resources to these variables.

**Action Items:**
- Deploy rapid-response resources addressing 'frp'
- Design targeted mitigation strategies for these key risk factors
- Establish continuous monitoring for high-risk threshold breaches

*Confidence: 85%*


### [HIGH] Deploy Model with Monitoring

Deploy the trained model to a staging environment with A/B testing and set up drift detection to ensure performance is maintained over time.

**Action Items:**
- Set up model serving infrastructure
- Implement prediction monitoring and alerting
- Schedule monthly model retraining
- Create A/B test plan for recommendation validation

*Confidence: 80%*


---

## Next Steps

- Validate model on holdout/test data
- Collect more recent data for retraining
- A/B test recommendations in production
- Set up monitoring for model performance drift