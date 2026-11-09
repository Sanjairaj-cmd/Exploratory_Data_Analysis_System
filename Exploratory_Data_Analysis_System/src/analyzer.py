import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def get_numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates extended statistical summary for all numerical columns.
    Includes mean, median, std, min, max, 25%, 75%, IQR, skewness, kurtosis.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return pd.DataFrame()
        
    stats = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        q25 = series.quantile(0.25)
        q75 = series.quantile(0.75)
        iqr = q75 - q25
        
        stats.append({
            "Column": col,
            "Count": len(series),
            "Mean": round(series.mean(), 3),
            "Std": round(series.std(), 3),
            "Min": round(series.min(), 3),
            "25%": round(q25, 3),
            "50% (Median)": round(series.median(), 3),
            "75%": round(q75, 3),
            "Max": round(series.max(), 3),
            "IQR": round(iqr, 3),
            "Skewness": round(series.skew(), 3),
            "Kurtosis": round(series.kurt(), 3),
            "Null Count": df[col].isnull().sum()
        })
        
    return pd.DataFrame(stats)

def get_categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates summary for categorical columns: Unique count, Top value, Top value frequency, Nulls.
    """
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
    if len(cat_cols) == 0:
        return pd.DataFrame()
        
    stats = []
    for col in cat_cols:
        series = df[col].dropna()
        if len(series) == 0:
            top_val = "N/A"
            top_freq = 0
        else:
            top_val = series.mode().iloc[0] if not series.mode().empty else "N/A"
            top_freq = int((series == top_val).sum())
            
        stats.append({
            "Column": col,
            "Total Count": len(df[col]),
            "Unique Values": df[col].nunique(),
            "Top Value": str(top_val),
            "Top Frequency": top_freq,
            "Top Percentage (%)": round((top_freq / len(df)) * 100, 2) if len(df) > 0 else 0,
            "Null Count": df[col].isnull().sum()
        })
        
    return pd.DataFrame(stats)

def get_univariate_metrics(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """
    Calculates detailed metrics for a single column.
    """
    if col not in df.columns:
        return {}
        
    series = df[col]
    is_num = pd.api.types.is_numeric_dtype(series)
    
    metrics = {
        "column_name": col,
        "is_numeric": is_num,
        "total_count": len(series),
        "missing_count": series.isnull().sum(),
        "missing_percentage": round((series.isnull().sum() / len(series)) * 100, 2),
        "unique_count": series.nunique()
    }
    
    if is_num:
        clean_series = series.dropna()
        mode_res = clean_series.mode()
        mode_val = round(mode_res.iloc[0], 3) if not mode_res.empty else np.nan
        q25 = clean_series.quantile(0.25)
        q75 = clean_series.quantile(0.75)
        
        metrics.update({
            "mean": round(clean_series.mean(), 3),
            "median": round(clean_series.median(), 3),
            "mode": mode_val,
            "std": round(clean_series.std(), 3),
            "var": round(clean_series.var(), 3),
            "min": round(clean_series.min(), 3),
            "max": round(clean_series.max(), 3),
            "q25": round(q25, 3),
            "q75": round(q75, 3),
            "iqr": round(q75 - q25, 3),
            "skewness": round(clean_series.skew(), 3),
            "kurtosis": round(clean_series.kurt(), 3)
        })
    else:
        val_counts = series.value_counts().head(10).to_dict()
        metrics["value_counts"] = val_counts
        
    return metrics

def get_bivariate_analysis(df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
    """
    Analyzes bivariate relationship depending on variable types.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return {}
        
    is_num1 = pd.api.types.is_numeric_dtype(df[col1])
    is_num2 = pd.api.types.is_numeric_dtype(df[col2])
    
    result = {"col1": col1, "col2": col2}
    
    if is_num1 and is_num2:
        result["type"] = "num_num"
        clean = df[[col1, col2]].dropna()
        pearson_corr = round(clean[col1].corr(clean[col2], method="pearson"), 4)
        spearman_corr = round(clean[col1].corr(clean[col2], method="spearman"), 4)
        result["pearson"] = pearson_corr
        result["spearman"] = spearman_corr
        
    elif is_num1 != is_num2:
        result["type"] = "num_cat"
        num_col = col1 if is_num1 else col2
        cat_col = col2 if is_num1 else col1
        
        grouped = df.groupby(cat_col)[num_col].agg(["count", "mean", "std", "median", "min", "max"]).reset_index()
        grouped["mean"] = grouped["mean"].round(3)
        grouped["std"] = grouped["std"].round(3)
        grouped["median"] = grouped["median"].round(3)
        
        result["num_col"] = num_col
        result["cat_col"] = cat_col
        result["grouped_stats"] = grouped
        
    else:
        result["type"] = "cat_cat"
        crosstab = pd.crosstab(df[col1], df[col2], margins=True)
        result["crosstab"] = crosstab
        
    return result

def compute_correlation_matrix(df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """
    Computes correlation matrix for numerical features.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or num_df.shape[1] < 2:
        return pd.DataFrame()
    return num_df.corr(method=method).round(3)

def get_top_correlations(corr_matrix: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Returns dataframe of highly correlated feature pairs (above threshold).
    """
    if corr_matrix.empty:
        return pd.DataFrame(columns=["Feature 1", "Feature 2", "Correlation", "Strength"])
        
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) >= threshold and not np.isnan(val):
                strength = "Strong Positive" if val > 0.7 else ("Moderate Positive" if val > 0 else ("Strong Negative" if val < -0.7 else "Moderate Negative"))
                pairs.append({
                    "Feature 1": cols[i],
                    "Feature 2": cols[j],
                    "Correlation": val,
                    "Abs Correlation": abs(val),
                    "Strength": strength
                })
                
    pairs_df = pd.DataFrame(pairs)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values(by="Abs Correlation", ascending=False).drop(columns=["Abs Correlation"]).reset_index(drop=True)
    return pairs_df
