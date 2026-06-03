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
df = fraud_df.copy()
print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
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
# Drop unnecessary ID and code columns
df.drop(columns=
        ["Provider_ID",
         "Claim_ID",
         "Diagnosis_Code",
         "Procedure_Code"],
        inplace=True)

# %%
df.head()

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
avg_claim_count = df["Claim_Amount"].mean()


# %%
# Function to return fraud rate, fraud claims and claim count for a column.
def fraud_by(column):
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

display(claims_avgs)
fraud_legit_cf1 = (
    claims_avgs.loc[0, "Avg_Approved_Amt_Gap"] / claims_avgs.loc[1, "Avg_Approved_Amt_Gap"]
) 

fraud_legit_cf2 = (
    (claims_avgs.loc[0, "Avg_Claim_Amount"] - claims_avgs.loc[1, "Avg_Claim_Amount"]) /
    claims_avgs.loc[1, "Avg_Claim_Amount"]
) 

print(f"Fraudulent claims have an average approved amount gap {fraud_legit_cf1:.2f} times greater than legitimate claims.")
print(f"Fraudulent claims have an average amount that is {fraud_legit_cf2:.2%} higher than legitimate claims.")

# %%
# Fraud by state
fraud_by_state = (
    fraud_by("Patient_State")
    .sort_values("Fraud_Rate", ascending=False)
)

fraud_by_state

# %%
# Fraud quarterly
fraud_quarterly = (
    fraud_by(["Year", "Quarter"])
    .sort_values(["Year", "Quarter"], ascending=True)
    .reset_index(drop=True)
)

fraud_quarterly


# %% [markdown]
# ## Visualisations

# %%
# Function for 'fraud rate by' chart
def fraud_rate_by_chart(fraud_by_df, category_col, title):
    
    plt.figure(figsize=(12, 5))
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

    plt.gca().invert_yaxis()

    plt.legend()
    plt.title(title, fontweight="bold")
    plt.xlabel("Fraud Rate (%)")
    plt.ylabel(category_col.replace('_', ' '))
    plt.tight_layout()
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
plt.figure(figsize=(12, 5))

bar_colours = ["#2A9D8F", "#4C78A8"] 
plt.bar(
    claims_avgs["Fraud_Label"],
    claims_avgs["Avg_Claim_Amount"],
    color=bar_colours,
    width=0.5
)

plt.axhline(
    y=avg_claim_count,
    linestyle="--",
    label=f"Overall Avg Claim Amount ({avg_claim_count:.0f})",
    color="#6C757D"
)

plt.legend()
plt.title("Average Claim Amount: Fraud vs Legitimate", fontweight="bold")
plt.ylabel("Average Claim Amount ($)")
plt.tight_layout()
plt.show()

# %%
# Days to submit - fraud vs legitimate chart
legit_days = df[df["Fraud_Label"] == "Legitimate"]["Days_Between_Service_and_Claim"]
fraud_days = df[df["Fraud_Label"] == "Fraud"]["Days_Between_Service_and_Claim"]

plt.figure(figsize=(12, 5))

plt.boxplot(
    [legit_days, fraud_days],
    tick_labels=["Legitimate", "Fraud"],
    vert=False,
    patch_artist=True,
    boxprops=dict(
        facecolor="#2A9D8F"
    ),
    medianprops=dict(
        color="black",
        linewidth=2
    )
)

plt.xlabel("Days")
plt.title("Days Between Service and Claim Submission", fontweight="bold")
plt.tight_layout()
plt.show()

# %%
# Fraud quarterly chart
fraud_quarterly["Period"] = (
    fraud_quarterly["Year"].astype(str) 
    + " " 
    + fraud_quarterly["Quarter"]
)

plt.figure(figsize=(12, 5))

plt.plot(
    fraud_quarterly["Period"],
    fraud_quarterly["Fraud_Rate"] * 100,
    marker="o",
    color="#2A9D8F",
    linewidth=2,
    markersize=5
)
plt.axhline(
    fraud_rate * 100,
    color="#6C757D",
    linestyle="--",
    label=f"Overall Avg ({fraud_rate:.2%})"
)
plt.ylabel("Fraud Rate (%)")
plt.title("Fraud Rate by Quarter (2021–2025)", fontweight="bold")
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
