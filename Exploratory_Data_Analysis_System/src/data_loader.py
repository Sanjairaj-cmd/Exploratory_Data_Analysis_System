import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any

def load_data(file_input) -> Tuple[pd.DataFrame, str]:
    """
    Loads a CSV file into a pandas DataFrame.
    
    Parameters:
        file_input: File path (str) or file-like object (UploadedFile).
        
    Returns:
        Tuple[pd.DataFrame, str]: (DataFrame, error_message or success_message)
    """
    try:
        if isinstance(file_input, str):
            df = pd.read_csv(file_input)
        else:
            # Handle uploaded buffer or file-like object
            df = pd.read_csv(file_input)
            
        if df.empty:
            return df, "Error: The uploaded CSV file is empty."
            
        return df, "Success"
    except Exception as e:
        # Fallback encoding attempt if standard utf-8 fails
        try:
            if isinstance(file_input, str):
                df = pd.read_csv(file_input, encoding="latin1")
            else:
                file_input.seek(0)
                df = pd.read_csv(file_input, encoding="latin1")
            return df, "Success (loaded with latin1 encoding)"
        except Exception as inner_e:
            return pd.DataFrame(), f"Failed to load CSV: {str(e)} | Secondary error: {str(inner_e)}"

def get_dataset_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes dataset schema, row/column counts, memory usage, and column classifications.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        
    Returns:
        Dict[str, Any]: Dictionary containing schema details and column classification.
    """
    if df is None or df.empty:
        return {
            "rows": 0,
            "cols": 0,
            "memory_mb": 0.0,
            "numerical_cols": [],
            "categorical_cols": [],
            "datetime_cols": [],
            "boolean_cols": [],
            "all_cols": []
        }
        
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
    
    # Check if any object columns can be datetime
    possible_dt = []
    for col in cat_cols:
        if "date" in col.lower() or "time" in col.lower():
            possible_dt.append(col)
            
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)
    
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "memory_mb": memory_mb,
        "numerical_cols": num_cols,
        "categorical_cols": cat_cols,
        "datetime_cols": dt_cols,
        "boolean_cols": bool_cols,
        "possible_dt_cols": possible_dt,
        "all_cols": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
