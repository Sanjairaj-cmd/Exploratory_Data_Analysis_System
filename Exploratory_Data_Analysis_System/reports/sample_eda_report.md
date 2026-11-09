# 📊 Exploratory Data Analysis & Insights Report
**Dataset Name:** `Employee_Analytics` | **Generated On:** `2026-10-08 17:52:32`

---
## 1. Executive Summary
- The dataset contains **50 rows** and **10 columns**.
- It consists of **5 numerical** features and **5 categorical** features.

## 2. Dataset Metadata & Dimensions
- **Total Records (Rows):** 50
- **Total Attributes (Columns):** 10
- **Duplicate Rows:** 0 (0.0%)
- **Memory Usage:** 0.01 MB

## 3. Data Cleaning & Quality Audit
### Missing Values Breakdown
| Column            |   Missing_Count |   Missing_Percentage | Data_Type   |
|:------------------|----------------:|---------------------:|:------------|
| Performance_Score |               1 |                    2 | float64     |
| Education         |               1 |                    2 | str         |

## 4. Statistical Summary
### Numerical Attributes
| Column             |   Count |      Mean |       Std |     Min |      25% |   50% (Median) |      75% |    Max |      IQR |   Skewness |   Kurtosis |   Null Count |
|:-------------------|--------:|----------:|----------:|--------:|---------:|---------------:|---------:|-------:|---------:|-----------:|-----------:|-------------:|
| Age                |      50 |    37.04  |    10.063 |    22   |    29    |           34.5 |    44.75 |     60 |    15.75 |      0.59  |     -0.675 |            0 |
| Years_Experience   |      50 |    10.04  |     8.068 |     1   |     4    |            7.5 |    15    |     32 |    11    |      0.987 |      0.199 |            0 |
| Salary             |      50 | 86300     | 48785.8   | 38000   | 56750    |        76500   | 93250    | 320000 | 36500    |      3.091 |     12.098 |            0 |
| Performance_Score  |      49 |     3.927 |     0.631 |     2.5 |     3.6  |            3.9 |     4.4  |      5 |     0.8  |     -0.294 |     -0.533 |            1 |
| Projects_Completed |      50 |     8.38  |     5.635 |     1   |     4.25 |            7   |    10.75 |     28 |     6.5  |      1.523 |      2.824 |            0 |

### Categorical Attributes
| Column        |   Total Count |   Unique Values | Top Value   |   Top Frequency |   Top Percentage (%) |   Null Count |
|:--------------|--------------:|----------------:|:------------|----------------:|---------------------:|-------------:|
| Employee_ID   |            50 |              50 | EMP001      |               1 |                    2 |            0 |
| Department    |            50 |               6 | Engineering |              15 |                   30 |            0 |
| Education     |            50 |               3 | Bachelor    |              24 |                   48 |            1 |
| Remote_Status |            50 |               3 | Hybrid      |              17 |                   34 |            0 |
| Overtime      |            50 |               2 | No          |              26 |                   52 |            0 |

## 5. Correlation Analysis
### Top Correlated Feature Pairs (|r| ≥ 0.50)
| Feature 1         | Feature 2          |   Correlation | Strength        |
|:------------------|:-------------------|--------------:|:----------------|
| Age               | Years_Experience   |         0.99  | Strong Positive |
| Salary            | Projects_Completed |         0.948 | Strong Positive |
| Years_Experience  | Projects_Completed |         0.924 | Strong Positive |
| Age               | Projects_Completed |         0.893 | Strong Positive |
| Performance_Score | Projects_Completed |         0.872 | Strong Positive |
| Years_Experience  | Salary             |         0.828 | Strong Positive |
| Age               | Performance_Score  |         0.816 | Strong Positive |
| Years_Experience  | Performance_Score  |         0.799 | Strong Positive |
| Age               | Salary             |         0.77  | Strong Positive |
| Salary            | Performance_Score  |         0.742 | Strong Positive |

## 6. Outlier Analysis (IQR Method)
| Column             |   Outlier Count |   Outlier Percentage (%) |   Lower Bound |   Upper Bound |
|:-------------------|----------------:|-------------------------:|--------------:|--------------:|
| Salary             |               2 |                        4 |      2000     |    148000     |
| Projects_Completed |               2 |                        4 |        -5.5   |        20.5   |
| Years_Experience   |               1 |                        2 |       -12.5   |        31.5   |
| Age                |               0 |                        0 |         5.375 |        68.375 |
| Performance_Score  |               0 |                        0 |         2.4   |         5.6   |

## 7. Automated Data Insights
### Missing & Duplicate Alerts
- ℹ️ **Moderate Missing Data**: Columns with minor missing values: `Performance_Score` (2.0%), `Education` (2.0%). Impute using median (numerical) or mode (categorical).
- ✅ **No Duplicates**: The dataset has zero duplicate rows.

### Distribution & Correlation Alerts
- 📈 **High Skewness Alert**: Features with significant skewness (|skew| > 1.0): `Salary` (skew: 3.09), `Projects_Completed` (skew: 1.52). Consider log or Box-Cox transformations for linear models.
- 🔗 **Strong Feature Relationships**: High correlations detected between: `Age` & `Years_Experience` (r = 0.99); `Salary` & `Projects_Completed` (r = 0.948); `Years_Experience` & `Projects_Completed` (r = 0.924). Watch for potential multicollinearity.
- 🎯 **Outliers Detected**: Significant outliers (>3% of data) found in: `Salary` (2 outliers, 4.0%), `Projects_Completed` (2 outliers, 4.0%). Consider IQR capping or winsorization.

## 8. Actionable Data Science Recommendations
1. Apply missing value imputation: Median for skewed numerical variables, Mode for categorical variables.
2. Apply log1p or power transformation on highly skewed features: Salary, Projects_Completed.
3. Perform IQR capping / winsorization on outlier-heavy features: Salary, Projects_Completed.
4. Normalize or standardize numerical features using `StandardScaler` or `RobustScaler` before modeling.