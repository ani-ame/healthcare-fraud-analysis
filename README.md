# Google Data Analytics Certificate Capstone
## Healthcare Fraud Detection

This project was completed as part of the Google Data Analytics Professional Certificate, sponsored by INCO Academy.

[View Full Analysis](./healthcare-fraud-analysis.ipynb) → 

## Project Workflow
- Ask - Define the business question
- Prepare - Understand and assess the data
- Process - Clean, transform, and engineer features
- Analyse - Calculate metrics and identify patterns
- Share - Visualise findings for stakeholders
- Act - Deliver business recommendations

## Goal
Conduct a deep dive into 10,000 healthcare insurance claims from one year of simulated historical data to identify the patterns that distinguish fraudulent claims from legitimate ones. The objective is to develop a data driven fraud profile and actionable recommendations that would enable a claims review team to prioritise investigations more effectively to know which claims deserve closer scrutiny and reduce fraudulent payouts.


## About the Data

Healthcare Fraud Detection dataset — [Kaggle](https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset/data). Included alongside this notebook, this dataset contains claims submitted between January 2021 and January 2025, spanning 8 U.S. states and 6 known provider specialties. Each claim record includes:

- **Patient information:** Age, gender, state, chronic condition flag, prior visits
- **Provider information:** Provider ID, specialty, monthly claim volume
- **Claim financials:** Claim amount, approved amount, insurance type, claim status
- **Timing:** Claim submission date, days between service and submission
- **`Is_Fraud`:** Whether the claim was confirmed as fraudulent (1) or legitimate (0)

## Key Findings

| Finding | Detail |
|---|---|
| Overall fraud rate | 8.3% of all claims (829 of 10,000) |
| Claim amount | Fraudulent claims average 85% more than legitimate claims |
| Claim amount gap| Fraudulent claims have an average approved amount gap that is 6.76× larger than legitimate claims|
| Submission timing | Fraud submitted in 3 days vs 15 days for legitimate claims |
| Highest risk specialty | General Practice — 9.7% fraud rate |
| Highest risk insurance | Medicare — 8.5% fraud rate |
| Top fraud states | Pennsylvania, Florida and Illinois |
| High volume providers | Providers submitting 90+ claims/month show elevated fraud rates |

## Business Recommendations

1. **Auto flag high volume claims providers** — any provider submitting abnormally high claims/month should trigger secondary review automatically
2. **Prioritise General Practice and Medicare claims** — they show the highest risk for fraud
3. **Introduce a submission timing rule** — claims filed within 3 days of service should require more supporting documentation before approval

## Tools

| Tool | Purpose |
|---|---|
| Python (pandas, matplotlib) | Data cleaning, feature engineering, analysis and charts |
| Jupyter Notebook | Documented analysis |
| GitHub | Code repository and version control |

---