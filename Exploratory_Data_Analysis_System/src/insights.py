import pandas as pd
import numpy as np
from typing import List, Dict, Any
from src.cleaner import get_missing_summary, get_duplicate_summary
from src.analyzer import compute_correlation_matrix, get_top_correlations
from src.outliers import detect_all_outliers

def generate_automated_insights(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyzes the DataFrame and generates dynamic natural-language insights and recommendations.
    
    Returns:
        Dict with keys:
            - 'overview': Summary overview lines
            - 'missing_insights': Missing value insights
            - 'duplicate_insights': Duplicate data insights
            - 'skewness_insights': Skewed feature warnings
            - 'correlation_insights': Key correlation insights
            - 'outlier_insights': High outlier feature alerts
            - 'recommendations': Step-by-step actionable ML/preprocessing steps
    """
    if df is None or df.empty:
        return {
            "overview": ["Dataset is empty."],
            "missing_insights": [],
            "duplicate_insights": [],
            "skewness_insights": [],
            "correlation_insights": [],
            "outlier_insights": [],
            "recommendations": []
        }
        
    n_rows, n_cols = df.shape
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Overview
    overview = [
        f"The dataset contains **{n_rows:,} rows** and **{n_cols} columns**.",
        f"It consists of **{len(num_cols)} numerical** features and **{len(cat_cols)} categorical** features."
    ]
    
    # Missing values
    missing_df = get_missing_summary(df)
    missing_cols = missing_df[missing_df["Missing_Count"] > 0]
    missing_insights = []
    
    if missing_cols.empty:
        missing_insights.append("✅ **Clean Dataset**: No missing values were detected across any columns.")
    else:
        severe = missing_cols[missing_cols["Missing_Percentage"] > 20]
        moderate = missing_cols[(missing_cols["Missing_Percentage"] <= 20) & (missing_cols["Missing_Percentage"] > 0)]
        
        if not severe.empty:
            severe_str = ", ".join([f"`{row['Column']}` ({row['Missing_Percentage']}%)" for _, row in severe.iterrows()])
            missing_insights.append(f"⚠️ **High Missing Data Warning**: Columns with >20% missing values: {severe_str}. Consider dropping or careful imputation.")
            
        if not moderate.empty:
            mod_str = ", ".join([f"`{row['Column']}` ({row['Missing_Percentage']}%)" for _, row in moderate.iterrows()])
            missing_insights.append(f"ℹ️ **Moderate Missing Data**: Columns with minor missing values: {mod_str}. Impute using median (numerical) or mode (categorical).")
            
    # Duplicates
    dup_info = get_duplicate_summary(df)
    dup_insights = []
    if dup_info["duplicate_count"] > 0:
        dup_insights.append(f"⚠️ **Duplicates Detected**: Found **{dup_info['duplicate_count']:,} duplicate rows** ({dup_info['duplicate_percent']}%). Deduplication is recommended before model training.")
    else:
        dup_insights.append("✅ **No Duplicates**: The dataset has zero duplicate rows.")
        
    # Skewness
    skewness_insights = []
    high_skew = []
    for col in num_cols:
        series = df[col].dropna()
        if len(series) > 0:
            skew_val = series.skew()
            if abs(skew_val) > 1.0:
                direction = "Right-skewed (positive)" if skew_val > 0 else "Left-skewed (negative)"
                high_skew.append((col, skew_val, direction))
                
    if high_skew:
        high_skew.sort(key=lambda x: abs(x[1]), reverse=True)
        skew_details = ", ".join([f"`{col}` (skew: {val:.2f})" for col, val, _ in high_skew[:4]])
        skewness_insights.append(f"📈 **High Skewness Alert**: Features with significant skewness (|skew| > 1.0): {skew_details}. Consider log or Box-Cox transformations for linear models.")
    else:
        skewness_insights.append("✅ **Balanced Distributions**: Numerical features display fairly symmetric distributions (|skew| ≤ 1.0).")
        
    # Correlation
    corr_insights = []
    if len(num_cols) >= 2:
        corr_matrix = compute_correlation_matrix(df)
        top_corr = get_top_correlations(corr_matrix, threshold=0.6)
        if not top_corr.empty:
            corr_list = []
            for _, row in top_corr.head(3).iterrows():
                corr_list.append(f"`{row['Feature 1']}` & `{row['Feature 2']}` (r = {row['Correlation']})")
            corr_str = "; ".join(corr_list)
            corr_insights.append(f"🔗 **Strong Feature Relationships**: High correlations detected between: {corr_str}. Watch for potential multicollinearity.")
        else:
            corr_insights.append("ℹ️ **No Extreme Correlations**: No pairs of numerical features exhibited correlation above |r| = 0.60.")
    else:
        corr_insights.append("ℹ️ Insufficient numerical features for correlation analysis.")
        
    # Outliers
    outlier_insights = []
    outlier_df = detect_all_outliers(df, method="iqr", factor=1.5)
    high_outliers = outlier_df[outlier_df["Outlier Percentage (%)"] > 3.0]
    
    if not high_outliers.empty:
        out_list = [f"`{row['Column']}` ({row['Outlier Count']} outliers, {row['Outlier Percentage (%)']}%)" for _, row in high_outliers.head(3).iterrows()]
        out_str = ", ".join(out_list)
        outlier_insights.append(f"🎯 **Outliers Detected**: Significant outliers (>3% of data) found in: {out_str}. Consider IQR capping or winsorization.")
    else:
        outlier_insights.append("✅ **Low Outlier Presence**: Numerical features have minor or no severe outliers.")
        
    # Actionable Preprocessing Recommendations
    recommendations = []
    if missing_cols["Missing_Count"].sum() > 0:
        recommendations.append("Apply missing value imputation: Median for skewed numerical variables, Mode for categorical variables.")
    if dup_info["duplicate_count"] > 0:
        recommendations.append("Execute duplicate removal to avoid overfitting or inflated metric statistics.")
    if high_skew:
        recommendations.append(f"Apply log1p or power transformation on highly skewed features: {', '.join([col for col, _, _ in high_skew[:3]])}.")
    if not high_outliers.empty:
        recommendations.append(f"Perform IQR capping / winsorization on outlier-heavy features: {', '.join([row['Column'] for _, row in high_outliers.head(2).iterrows()])}.")
    if len(num_cols) > 0:
        recommendations.append("Normalize or standardize numerical features using `StandardScaler` or `RobustScaler` before modeling.")
        
    if not recommendations:
        recommendations.append("Dataset is clean and ready for machine learning feature selection and baseline modeling!")
        
    return {
        "overview": overview,
        "missing_insights": missing_insights,
        "duplicate_insights": dup_insights,
        "skewness_insights": skewness_insights,
        "correlation_insights": corr_insights,
        "outlier_insights": outlier_insights,
        "recommendations": recommendations
    }
