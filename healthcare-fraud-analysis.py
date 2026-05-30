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
df.head()

# %%
df.info()
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")

# %%
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
print(df["Prior_Visits_12m"].describe().round(3))
skewness = df["Prior_Visits_12m"].skew()
print(f"Skewness: {skewness:.2f}")

# %%
df["Prior_Visits_12m"] = df["Prior_Visits_12m"].fillna(
    df["Prior_Visits_12m"].median()
)

# %%
df.info()

# %% [markdown]
# ## Feature Engineering

# %%
print(df["Number_of_Claims_Per_Provider_Monthly"].describe().round(3))

# %%
# Flag providers with more than 90 monthly claims
df["High_Volume_Claims_Flag"] = (
    df["Number_of_Claims_Per_Provider_Monthly"]
        .gt(90)
        .astype(int)
)

# %%
# Extract components from date column
df["Month"] = df["Claim_Submission_Date"].dt.month_name()
df["Quarter"] = "Q" + df["Claim_Submission_Date"].dt.quarter.astype(str)
df["Year"] = df["Claim_Submission_Date"].dt.year

# %%
# Measure the gap between what was claimed and what was approved 
df["Approved_Amt_Gap"] = df["Claim_Amount"] - df["Approved_Amount"]
df["Approved_Rate"] = (df["Approved_Amount"] / df["Claim_Amount"]).round(2)

# %%
# Fraud or legit
df["Fraud_Label"] = df["Is_Fraud"].map({1: "Fraud", 0: "Legitimate"})
df.head()

# %% [markdown]
# ## Analysis

# %%
fraud_rate = df["Is_Fraud"].mean()
avg_claims_count = df["Claim_Amount"].mean()


# %%
# Function to return fraud rate, fraud claims and claim count for a column.
def fraud_by (column):
    return (
        df.groupby(column)["Is_Fraud"]
            .agg(
                Fraud_Rate = "mean",
                Fraud_Claims = "sum",
                Total_Claims = "count"
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

# %%
# Claims averages - fraud vs legitimate
claims_avgs = (
    df.groupby("Fraud_Label")[
        ["Claim_Amount",
         "Approved_Amount",
         "Approved_Amt_Gap",
         "Days_Between_Service_and_Claim"]
    ]
    .mean()
    .round(2)
    .reset_index()
    .rename(columns={
        "Claim_Amount": "Avg_Claim_Amount",
        "Approved_Amount": "Avg_Approved_Amount",
        "Approved_Amt_Gap": "Avg_Approved_Amt_Gap",
        "Days_Between_Service_and_Claim": "Avg_Days_Between_Service_and_Claim"
    })
)

claims_avgs


# %% [markdown]
# ## Visualisations

# %%
# Function for 'fraud rate by' chart
def fraud_rate_by_chart(fraud_by_df, category_col, title):
    plt.figure()
    plt.barh(
        fraud_by_df[category_col],
        fraud_by_df["Fraud_Rate"] * 100,
        color="#2A9D8F"
    )
    plt.axvline(
        x=fraud_rate * 100,
        linestyle="--",
        label=f"Overall Avg Fraud Rate ({fraud_rate:.2%})",
        color="#6C757D"
    )
    plt.legend()
    plt.title(title, fontweight="bold")
    plt.xlabel("Fraud Rate (%)")
    plt.ylabel(category_col.replace('_', ' '))
    plt.tight_layout()
    plt.savefig(f"charts/fraud_rate_by_{category_col.lower()}.png")
    plt.show()


# %%
# Fraud rate by provider specialty chart
fraud_rate_by_chart(
    fraud_by_provider_specialty,
    "Provider_Specialty", 
    "Fraud Rate by Provider Specialty"
)

# %%
# Fraud rate by insurance type chart
fraud_rate_by_chart(
    fraud_by_insurance_type,
    "Insurance_Type", 
    "Fraud Rate by Insurance Type"
)

# %%
# Claims averages - fraud vs legitimate chart
plt.figure()

bar_colors = ["#2A9D8F", "#4C78A8"] 
plt.bar(
    claims_avgs["Fraud_Label"],
    claims_avgs["Avg_Claim_Amount"],
    color=bar_colors,
    width=0.5
)

plt.axhline(
    y=avg_claims_count,
    linestyle="--",
    label=f"Overall Avg Claim Amount ({avg_claims_count:.0f})",
    color="#6C757D"
)

plt.legend()
plt.title("Average Claim Amount: Fraud vs Legitimate", fontweight="bold")
plt.ylabel("Average Claim Amount ($)")
plt.tight_layout()
plt.savefig("charts/claim_amounts.png")
plt.show()
