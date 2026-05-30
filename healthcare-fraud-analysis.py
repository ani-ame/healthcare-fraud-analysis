# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python (venv)
#     language: python
#     name: .venv
# ---

# %% [markdown]
# # Healthcare Fraud Detection

# %% [markdown]
# ## Business Question

# %% [markdown]
# > _What patterns most reliably identify fraudulent insurance claims?_

# %% [markdown]
# ## Dataset

# %% [markdown]
# > Sourced from [Kaggle](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset/data), this dataset comprises 10,000 simulated healthcare insurance claims that model real-world fraud scenarios. It includes a range of patient, provider, and financial attributes.

# %%
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)

# %% [markdown]
# ## Load Data

# %%
fraud_df = pd.read_csv("healthcare_fraud_detection.csv")

# %% [markdown]
# ## Raw Data Overview

# %%
df = fraud_df
print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
display(df.head())
df.info()
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

print(f"\nFraud breakdown:\n{df['Is_Fraud'].value_counts()}")
print(f"\nFraud rate: {df['Is_Fraud'].mean():.2%}")

# %% [markdown]
# ## Clean Data

# %%
df["Claim_Submission_Date"] = pd.to_datetime(df["Claim_Submission_Date"])

# %%
df["Insurance_Type"] = df["Insurance_Type"].fillna("Unknown")
df["Provider_Specialty"] = df["Provider_Specialty"].fillna("Unknown")

# %%
print(df["Prior_Visits_12m"].describe())
print(f"Skewness: {df['Prior_Visits_12m'].skew()}")

# %%
df["Prior_Visits_12m"] = df["Prior_Visits_12m"].fillna(df["Prior_Visits_12m"].median())

# %%
df.info()

# %% [markdown]
# ## Feature Engineering

# %%
print(df["Number_of_Claims_Per_Provider_Monthly"].describe())

# %%
# Flag providers with more than 90 monthly claims
df["High_Volume_Claims_Flag"] = (df["Number_of_Claims_Per_Provider_Monthly"] > 90).astype(int)

# %%
# Extract components from date column
df["Month"] = df["Claim_Submission_Date"].dt.month_name()
df["Quarter"] = "Q" + df["Claim_Submission_Date"].dt.quarter.astype(str)
df["Year"] = df["Claim_Submission_Date"].dt.year
df

# %% [markdown]
# ## Analysis

# %%
fraud_rate = df["Is_Fraud"].mean()


# %%
# Function to return fraud rate, fraud claims and claim count for a column.
def fraud_by (column):
    return (
        df.groupby(column)["Is_Fraud"]
            .agg(
                Fraud_Rate = "mean"
                ,Fraud_Claims = "sum"
                ,Total_Claims = "count"
        )
        .sort_values("Fraud_Rate", ascending=False)
        .reset_index()
    )


# %%
# Fraud by provider specialty
fraud_by_provider_specialty = fraud_by("Provider_Specialty")

fraud_by_provider_specialty

# %%
# Fraud by insurance type
fraud_by_insurance_type = fraud_by("Insurance_Type")

fraud_by_insurance_type

# %%
# Fraud by visit type
fraud_by_visit_type = fraud_by("Visit_Type")

fraud_by_visit_type 

# %%
# Fraud by high volume providers
fraud_by_hv = fraud_by("High_Volume_Claims_Flag")
display(fraud_by_hv)

label = "volume monthly claims providers fraud pct"
high_rate = fraud_by_hv['Fraud_Rate'].iloc[0]
low_rate = fraud_by_hv['Fraud_Rate'].iloc[1]

print(f"High {label}: {high_rate:.2%}")
print(f"Low {label}: {low_rate:.2%}")

# %% [markdown]
# ## Visualisations

# %%
# Fraud rate by provider specialty
plt.figure()

plt.barh(
    fraud_by_provider_specialty["Provider_Specialty"],
    fraud_by_provider_specialty["Fraud_Rate"] * 100,
    color="#2A9D8F"
)

plt.axvline(
    x = fraud_rate * 100,
    linestyle="--",
    label=f"Overall Avg Fraud Rate ({fraud_rate:.1%})",
    color="#6C757D"
)

plt.legend()

plt.title("Fraud Rate by Provider Specialty")

plt.xlabel("Fraud Rate (%)")
plt.ylabel("Provider Specialty")

plt.tight_layout()

plt.savefig("charts/fraud_rate_by_specialty.png")

# %%
# Fraud rate by insurance type
plt.figure()

plt.barh(
    fraud_by_insurance_type["Insurance_Type"],
    fraud_by_insurance_type["Fraud_Rate"] * 100,
    color="#2A9D8F"
)

plt.axvline(
    x=fraud_rate * 100,
    linestyle="--",
    label=f"Overall Avg Fraud Rate ({fraud_rate:.1%})",
    color="#6C757D"
)

plt.legend()

plt.title("Fraud Rate by Insurance Type")

plt.xlabel("Fraud Rate (%)")
plt.ylabel("Insurance Type")

plt.tight_layout()

plt.savefig("charts/fraud_rate_by_insurance_type.png")

# %%
