import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def detect_outliers_iqr(df: pd.DataFrame, col: str, factor: float = 1.5) -> Dict[str, Any]:
    """
    Detects outliers in a numerical column using Interquartile Range (IQR).
    """
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return {"outlier_count": 0, "outlier_percentage": 0.0, "lower_bound": np.nan, "upper_bound": np.nan, "mask": pd.Series(False, index=df.index)}
        
    series = df[col].dropna()
    q25 = series.quantile(0.25)
    q75 = series.quantile(0.75)
    iqr = q75 - q25
    
    lower_bound = q25 - (factor * iqr)
    upper_bound = q75 + (factor * iqr)
    
    mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    count = int(mask.sum())
    percent = round((count / len(df)) * 100, 2) if len(df) > 0 else 0.0
    
    return {
        "column": col,
        "method": "IQR",
        "factor": factor,
        "q25": round(q25, 3),
        "q75": round(q75, 3),
        "iqr": round(iqr, 3),
        "lower_bound": round(lower_bound, 3),
        "upper_bound": round(upper_bound, 3),
        "outlier_count": count,
        "outlier_percentage": percent,
        "mask": mask
    }

def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> Dict[str, Any]:
    """
    Detects outliers in a numerical column using Z-Score.
    """
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return {"outlier_count": 0, "outlier_percentage": 0.0, "lower_bound": np.nan, "upper_bound": np.nan, "mask": pd.Series(False, index=df.index)}
        
    series = df[col].dropna()
    mean = series.mean()
    std = series.std()
    
    if std == 0 or np.isnan(std):
        return {"outlier_count": 0, "outlier_percentage": 0.0, "lower_bound": mean, "upper_bound": mean, "mask": pd.Series(False, index=df.index)}
        
    z_scores = (df[col] - mean) / std
    mask = z_scores.abs() > threshold
    count = int(mask.sum())
    percent = round((count / len(df)) * 100, 2) if len(df) > 0 else 0.0
    
    lower_bound = mean - (threshold * std)
    upper_bound = mean + (threshold * std)
    
    return {
        "column": col,
        "method": "Z-Score",
        "threshold": threshold,
        "mean": round(mean, 3),
        "std": round(std, 3),
        "lower_bound": round(lower_bound, 3),
        "upper_bound": round(upper_bound, 3),
        "outlier_count": count,
        "outlier_percentage": percent,
        "mask": mask
    }

def detect_all_outliers(df: pd.DataFrame, method: str = "iqr", factor: float = 1.5, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Scans all numerical columns in DataFrame for outliers.
    """
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) == 0:
        return pd.DataFrame(columns=["Column", "Outlier Count", "Outlier Percentage (%)", "Lower Bound", "Upper Bound"])
        
    summary = []
    for col in num_cols:
        if method.lower() == "iqr":
            res = detect_outliers_iqr(df, col, factor=factor)
        else:
            res = detect_outliers_zscore(df, col, threshold=z_threshold)
            
        summary.append({
            "Column": col,
            "Outlier Count": res["outlier_count"],
            "Outlier Percentage (%)": res["outlier_percentage"],
            "Lower Bound": res["lower_bound"],
            "Upper Bound": res["upper_bound"]
        })
        
    res_df = pd.DataFrame(summary)
    res_df = res_df.sort_values(by="Outlier Count", ascending=False).reset_index(drop=True)
    return res_df

def handle_outliers(df: pd.DataFrame, col: str, action: str = "cap", method: str = "iqr", factor: float = 1.5, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Handles outliers for a specific column.
    action: 'cap' (winsorize) or 'remove' (drop rows with outliers)
    """
    df_clean = df.copy()
    if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[col]):
        return df_clean
        
    if method.lower() == "iqr":
        res = detect_outliers_iqr(df_clean, col, factor=factor)
    else:
        res = detect_outliers_zscore(df_clean, col, threshold=z_threshold)
        
    lower_bound = res["lower_bound"]
    upper_bound = res["upper_bound"]
    mask = res["mask"]
    
    if action == "remove":
        df_clean = df_clean[~mask].reset_index(drop=True)
    elif action == "cap":
        df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
    return df_clean
