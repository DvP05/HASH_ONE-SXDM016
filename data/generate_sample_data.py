"""
Synthetic SaaS Customer Churn Dataset Generator
Generates ~10k rows with realistic patterns, correlations,
missing values, outliers, and categorical inconsistencies.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pandas as pd


def generate_churn_data(n_rows: int = 10000, seed: int = 42,
                         output_path: str = "data/sample_churn_data.csv") -> pd.DataFrame:
    """Generate a synthetic SaaS customer churn dataset."""
    np.random.seed(seed)
    random.seed(seed)

    print(f"Generating {n_rows} rows of synthetic churn data...")

    # ── Base Features ──
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_rows + 1)]

    # Signup dates (last 5 years)
    start_date = pd.Timestamp("2021-01-01")
    end_date = pd.Timestamp("2025-12-31")
    days_range = (end_date - start_date).days
    signup_dates = [start_date + pd.Timedelta(days=random.randint(0, days_range))
                    for _ in range(n_rows)]

    # Tenure (correlated with signup date)
    tenure_days = [(pd.Timestamp("2026-01-01") - d).days for d in signup_dates]
    tenure_days = np.array(tenure_days) + np.random.normal(0, 30, n_rows)
    tenure_days = np.clip(tenure_days, 1, 2000).astype(int)

    # Plan types
    plans = np.random.choice(["Basic", "Standard", "Premium", "Enterprise"],
                             size=n_rows, p=[0.35, 0.30, 0.25, 0.10])

    # Monthly charges (correlated with plan)
    plan_charges = {"Basic": 29, "Standard": 59, "Premium": 99, "Enterprise": 199}
    monthly_charges = np.array([plan_charges[p] for p in plans]) + \
                      np.random.normal(0, 10, n_rows)
    monthly_charges = np.clip(monthly_charges, 9.99, 499.99).round(2)

    # Total charges (correlated with tenure and monthly)
    total_charges = monthly_charges * (tenure_days / 30) + np.random.normal(0, 100, n_rows)
    total_charges = np.clip(total_charges, 0, 50000).round(2)

    # Support tickets (higher for churners)
    base_tickets = np.random.poisson(2, n_rows)

    # Average response time
    avg_response_time = np.random.exponential(12, n_rows).round(1)
    avg_response_time = np.clip(avg_response_time, 0.5, 96)

    # Login frequency (per month)
    login_frequency = np.random.lognormal(2, 0.8, n_rows).round(1)
    login_frequency = np.clip(login_frequency, 0, 100)

    # Feature usage score (0-100)
    feature_usage = np.random.beta(2, 5, n_rows) * 100
    feature_usage = feature_usage.round(1)

    # Contract type
    contracts = np.random.choice(["Month-to-month", "One year", "Two year"],
                                  size=n_rows, p=[0.50, 0.30, 0.20])

    # Payment method
    payment_methods = np.random.choice(
        ["Credit card", "Bank transfer", "Electronic check", "Mailed check"],
        size=n_rows, p=[0.35, 0.25, 0.25, 0.15]
    )

    # Region
    regions = np.random.choice(
        ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East"],
        size=n_rows, p=[0.35, 0.25, 0.20, 0.12, 0.08]
    )

    # Demographics
    ages = np.random.normal(42, 12, n_rows).astype(int)
    ages = np.clip(ages, 18, 85)

    genders = np.random.choice(["Male", "Female", "Non-binary", "Prefer not to say"],
                                size=n_rows, p=[0.45, 0.42, 0.08, 0.05])

    num_products = np.random.choice([1, 2, 3, 4, 5], size=n_rows,
                                     p=[0.30, 0.30, 0.20, 0.15, 0.05])

    has_partner = np.random.choice([0, 1], size=n_rows, p=[0.52, 0.48])
    has_dependents = np.random.choice([0, 1], size=n_rows, p=[0.70, 0.30])

    online_security = np.random.choice(["Yes", "No", "No internet"],
                                        size=n_rows, p=[0.35, 0.50, 0.15])
    tech_support = np.random.choice(["Yes", "No", "No internet"],
                                     size=n_rows, p=[0.30, 0.55, 0.15])

    # ── Churn Target (with realistic correlations) ──
    churn_score = np.zeros(n_rows)

    # Higher churn probability factors:
    churn_score += (contracts == "Month-to-month") * 1.5
    churn_score += (monthly_charges > 80) * 0.8
    churn_score += (base_tickets > 4) * 1.2
    churn_score += (login_frequency < 5) * 1.0
    churn_score += (feature_usage < 20) * 0.9
    churn_score += (tenure_days < 180) * 0.7
    churn_score += (payment_methods == "Electronic check") * 0.6
    churn_score += (online_security == "No") * 0.4
    churn_score += (tech_support == "No") * 0.3
    churn_score += (avg_response_time > 24) * 0.5

    # Lower churn probability factors:
    churn_score -= (contracts == "Two year") * 1.5
    churn_score -= (tenure_days > 720) * 1.0
    churn_score -= (feature_usage > 60) * 0.8
    churn_score -= (login_frequency > 20) * 0.6
    churn_score -= (num_products > 2) * 0.4

    # Add noise
    churn_score += np.random.normal(0, 1.5, n_rows)

    # Convert to probability
    churn_prob = 1 / (1 + np.exp(-churn_score + 2))  # Sigmoid with offset
    churned = (np.random.random(n_rows) < churn_prob).astype(int)

    # Adjust support tickets for churners
    base_tickets = base_tickets + churned * np.random.poisson(2, n_rows)

    # ── Introduce Data Quality Issues ──

    # Missing values
    missing_indices = {
        "age": np.random.choice(n_rows, int(n_rows * 0.03), replace=False),
        "monthly_charges": np.random.choice(n_rows, int(n_rows * 0.02), replace=False),
        "avg_response_time_hrs": np.random.choice(n_rows, int(n_rows * 0.05), replace=False),
        "login_frequency": np.random.choice(n_rows, int(n_rows * 0.04), replace=False),
        "feature_usage_score": np.random.choice(n_rows, int(n_rows * 0.03), replace=False),
        "online_security": np.random.choice(n_rows, int(n_rows * 0.02), replace=False),
        "tech_support": np.random.choice(n_rows, int(n_rows * 0.02), replace=False),
    }

    # Outliers (a few extreme values)
    outlier_idx = np.random.choice(n_rows, 15, replace=False)
    monthly_charges_with_outliers = monthly_charges.copy()
    monthly_charges_with_outliers[outlier_idx[:5]] = np.random.uniform(800, 2000, 5)
    total_charges_with_outliers = total_charges.copy()
    total_charges_with_outliers[outlier_idx[5:10]] = np.random.uniform(80000, 150000, 5)

    # Categorical inconsistencies
    inconsistent_idx = np.random.choice(n_rows, 50, replace=False)
    regions_list = list(regions)
    alt_names = {
        "North America": ["NA", "N. America", "US/Canada"],
        "Europe": ["EU", "EMEA"],
        "Asia Pacific": ["APAC", "Asia"],
    }
    for idx in inconsistent_idx:
        original = regions_list[idx]
        if original in alt_names:
            regions_list[idx] = random.choice(alt_names[original])
    regions = np.array(regions_list)

    # ── Build DataFrame ──
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "signup_date": signup_dates,
        "tenure_days": tenure_days,
        "plan_type": plans,
        "monthly_charges": monthly_charges_with_outliers,
        "total_charges": total_charges_with_outliers,
        "num_support_tickets": base_tickets,
        "avg_response_time_hrs": avg_response_time,
        "login_frequency": login_frequency,
        "feature_usage_score": feature_usage,
        "contract_type": contracts,
        "payment_method": payment_methods,
        "region": regions,
        "age": ages.astype(float),
        "gender": genders,
        "num_products": num_products,
        "has_partner": has_partner,
        "has_dependents": has_dependents,
        "online_security": online_security,
        "tech_support": tech_support,
        "churned_30d": churned,
    })

    # Apply missing values
    for col, indices in missing_indices.items():
        if col in df.columns:
            df.loc[indices, col] = np.nan

    # Add a few duplicate rows
    dup_idx = np.random.choice(n_rows, 25, replace=False)
    duplicates = df.iloc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"[OK] Generated {len(df)} rows x {len(df.columns)} columns")
    print(f"   Churn rate: {churned.mean():.1%}")
    print(f"   Saved to: {output_path}")

    return df


if __name__ == "__main__":
    generate_churn_data()
