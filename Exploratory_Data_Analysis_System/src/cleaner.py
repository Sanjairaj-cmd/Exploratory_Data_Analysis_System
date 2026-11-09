import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple

def get_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame listing missing value count and percentage per column.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Column", "Missing_Count", "Missing_Percentage", "Data_Type"])
        
    missing_count = df.isnull().sum()
    missing_percent = (missing_count / len(df)) * 100
    
    summary = pd.DataFrame({
        "Column": df.columns,
        "Missing_Count": missing_count.values,
        "Missing_Percentage": missing_percent.round(2).values,
        "Data_Type": df.dtypes.astype(str).values
    })
    
    # Filter columns with missing values first or sort descending by missing percentage
    summary = summary.sort_values(by="Missing_Percentage", ascending=False).reset_index(drop=True)
    return summary

def get_duplicate_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns total count and percentage of duplicate rows.
    """
    if df is None or df.empty:
        return {"duplicate_count": 0, "duplicate_percent": 0.0}
        
    dup_count = df.duplicated().sum()
    dup_percent = round((dup_count / len(df)) * 100, 2)
    return {
        "duplicate_count": int(dup_count),
        "duplicate_percent": float(dup_percent)
    }

def handle_missing_values(df: pd.DataFrame, strategy_map: Dict[str, str], custom_vals: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Handles missing values based on strategy_map.
    Strategies: 'mean', 'median', 'mode', 'drop_rows', 'drop_col', 'ffill', 'bfill', 'constant'
    """
    df_clean = df.copy()
    if custom_vals is None:
        custom_vals = {}
        
    cols_to_drop = []
    
    for col, strategy in strategy_map.items():
        if col not in df_clean.columns:
            continue
            
        if strategy == "mean" and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        elif strategy == "mode":
            mode_val = df_clean[col].mode()
            if not mode_val.empty:
                df_clean[col] = df_clean[col].fillna(mode_val[0])
        elif strategy == "drop_rows":
            df_clean = df_clean.dropna(subset=[col])
        elif strategy == "drop_col":
            cols_to_drop.append(col)
        elif strategy == "ffill":
            df_clean[col] = df_clean[col].ffill()
        elif strategy == "bfill":
            df_clean[col] = df_clean[col].bfill()
        elif strategy == "constant" and col in custom_vals:
            df_clean[col] = df_clean[col].fillna(custom_vals[col])
            
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        
    return df_clean

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from DataFrame.
    """
    if df is None:
        return df
    return df.drop_duplicates().reset_index(drop=True)

def convert_column_dtypes(df: pd.DataFrame, type_map: Dict[str, str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Converts columns to specified data types safely.
    type_map: {'col_name': 'numeric' | 'category' | 'datetime' | 'string'}
    """
    df_converted = df.copy()
    errors = []
    
    for col, target_type in type_map.items():
        if col not in df_converted.columns:
            continue
        try:
            if target_type == "numeric":
                df_converted[col] = pd.to_numeric(df_converted[col], errors="coerce")
            elif target_type == "category":
                df_converted[col] = df_converted[col].astype("category")
            elif target_type == "datetime":
                df_converted[col] = pd.to_datetime(df_converted[col], errors="coerce")
            elif target_type == "string":
                df_converted[col] = df_converted[col].astype(str)
        except Exception as e:
            errors.append(f"Failed to convert '{col}' to {target_type}: {str(e)}")
            
    return df_converted, errors
