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
