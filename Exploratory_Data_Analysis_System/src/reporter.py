import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any
from src.cleaner import get_missing_summary, get_duplicate_summary
from src.analyzer import get_numerical_summary, get_categorical_summary, compute_correlation_matrix, get_top_correlations
from src.outliers import detect_all_outliers

def df_to_markdown(dframe: pd.DataFrame) -> str:
    """Safely converts DataFrame to markdown table with fallback."""
    if dframe.empty:
        return "*No data available*"
    try:
        return dframe.to_markdown(index=False)
    except Exception:
        # Fallback string representation if tabulate is unavailable
        headers = "| " + " | ".join(dframe.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(dframe.columns)) + " |"
        rows = []
        for _, row in dframe.iterrows():
            rows.append("| " + " | ".join(str(val) for val in row.values) + " |")
        return "\n".join([headers, sep] + rows)

def generate_markdown_report(df: pd.DataFrame, dataset_name: str = "Dataset", insights: Dict[str, Any] = None) -> str:
    """
    Generates a comprehensive Markdown report summarizing the EDA results.
    """
    if insights is None:
        from src.insights import generate_automated_insights
        insights = generate_automated_insights(df)
        
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_rows, n_cols = df.shape
    
    num_summary = get_numerical_summary(df)
    cat_summary = get_categorical_summary(df)
    missing_summary = get_missing_summary(df)
    dup_summary = get_duplicate_summary(df)
    outlier_summary = detect_all_outliers(df)
    
    md = []
    md.append(f"# 📊 Exploratory Data Analysis & Insights Report")
    md.append(f"**Dataset Name:** `{dataset_name}` | **Generated On:** `{now_str}`\n")
    md.append("---")
    
    # Executive Summary
    md.append("## 1. Executive Summary")
    for item in insights.get("overview", []):
        md.append(f"- {item}")
    md.append("")
    
    # Dataset Metadata
    md.append("## 2. Dataset Metadata & Dimensions")
    md.append(f"- **Total Records (Rows):** {n_rows:,}")
    md.append(f"- **Total Attributes (Columns):** {n_cols}")
    md.append(f"- **Duplicate Rows:** {dup_summary['duplicate_count']} ({dup_summary['duplicate_percent']}%)")
    md.append(f"- **Memory Usage:** {round(df.memory_usage(deep=True).sum() / (1024*1024), 2)} MB\n")
    
    # Data Quality & Missing Values
    md.append("## 3. Data Cleaning & Quality Audit")
    if missing_summary["Missing_Count"].sum() > 0:
        md.append("### Missing Values Breakdown")
        md.append(df_to_markdown(missing_summary[missing_summary["Missing_Count"] > 0]))
    else:
        md.append("✅ **No missing values found across all attributes.**")
    md.append("")
    
    # Statistical Overview
    md.append("## 4. Statistical Summary")
    if not num_summary.empty:
        md.append("### Numerical Attributes")
        md.append(df_to_markdown(num_summary))
        md.append("")
        
    if not cat_summary.empty:
        md.append("### Categorical Attributes")
        md.append(df_to_markdown(cat_summary))
        md.append("")
        
    # Correlations
    md.append("## 5. Correlation Analysis")
    corr_matrix = compute_correlation_matrix(df)
    top_corr = get_top_correlations(corr_matrix, threshold=0.5)
    if not top_corr.empty:
        md.append("### Top Correlated Feature Pairs (|r| ≥ 0.50)")
        md.append(df_to_markdown(top_corr))
    else:
        md.append("No strong correlations (|r| ≥ 0.50) identified between numerical attributes.")
    md.append("")
    
    # Outliers
    md.append("## 6. Outlier Analysis (IQR Method)")
    if not outlier_summary.empty:
        md.append(df_to_markdown(outlier_summary))
    md.append("")
    
    # Key Automated Insights
    md.append("## 7. Automated Data Insights")
    md.append("### Missing & Duplicate Alerts")
    for line in insights.get("missing_insights", []) + insights.get("duplicate_insights", []):
        md.append(f"- {line}")
    md.append("\n### Distribution & Correlation Alerts")
    for line in insights.get("skewness_insights", []) + insights.get("correlation_insights", []) + insights.get("outlier_insights", []):
        md.append(f"- {line}")
    md.append("")
    
    # Recommendations
    md.append("## 8. Actionable Data Science Recommendations")
    for i, rec in enumerate(insights.get("recommendations", []), 1):
        md.append(f"{i}. {rec}")
        
    return "\n".join(md)

def generate_html_report(df: pd.DataFrame, dataset_name: str = "Dataset", insights: Dict[str, Any] = None) -> str:
    """
    Generates a standalone, beautiful HTML report document.
    """
    if insights is None:
        from src.insights import generate_automated_insights
        insights = generate_automated_insights(df)
        
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_rows, n_cols = df.shape
    
    num_summary = get_numerical_summary(df)
    cat_summary = get_categorical_summary(df)
    missing_summary = get_missing_summary(df)
    dup_summary = get_duplicate_summary(df)
    outlier_summary = detect_all_outliers(df)
    
    def df_to_html(dframe):
        if dframe.empty:
            return "<p><em>No data available</em></p>"
        return dframe.to_html(classes="report-table", index=False, border=0)
        
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>EDA Summary Report - {dataset_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f4f6f9;
            color: #2c3e50;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1100px;
            margin: 30px auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }}
        .header {{
            border-bottom: 3px solid #1e88e5;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            margin: 0;
            color: #0d47a1;
            font-size: 28px;
        }}
        .meta {{
            color: #666;
            font-size: 14px;
            margin-top: 8px;
        }}
        .badge {{
            background: #e3f2fd;
            color: #1565c0;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: bold;
        }}
        h2 {{
            color: #1565c0;
            border-left: 4px solid #1976d2;
            padding-left: 10px;
            margin-top: 35px;
            font-size: 20px;
        }}
        h3 {{
            color: #37474f;
            font-size: 16px;
            margin-top: 20px;
        }}
        .kpi-container {{
            display: flex;
            gap: 20px;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }}
        .kpi-card {{
            flex: 1;
            min-width: 200px;
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .kpi-value {{
            font-size: 26px;
            font-weight: bold;
            color: #1e88e5;
        }}
        .kpi-label {{
            font-size: 13px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .report-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }}
        .report-table th {{
            background-color: #f0f4f8;
            color: #102a43;
            text-align: left;
            padding: 10px 12px;
            border-bottom: 2px solid #cbd5e1;
        }}
        .report-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
        }}
        .report-table tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        .insight-list {{
            background: #f1f8e9;
            border-left: 4px solid #7cb342;
            padding: 15px 20px;
            border-radius: 4px;
            margin: 15px 0;
        }}
        .recommendation-list {{
            background: #fff8e1;
            border-left: 4px solid #ffb300;
            padding: 15px 20px;
            border-radius: 4px;
            margin: 15px 0;
        }}
        ul, ol {{
            margin: 0;
            padding-left: 20px;
        }}
        li {{
            margin-bottom: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Exploratory Data Analysis & Insights System</h1>
            <div class="meta">
                Dataset: <span class="badge">{dataset_name}</span> | 
                Generated on: <strong>{now_str}</strong>
            </div>
        </div>

        <div class="kpi-container">
            <div class="kpi-card">
                <div class="kpi-value">{n_rows:,}</div>
                <div class="kpi-label">Total Rows</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{n_cols}</div>
                <div class="kpi-label">Total Columns</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{dup_summary['duplicate_count']}</div>
                <div class="kpi-label">Duplicate Rows ({dup_summary['duplicate_percent']}%)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{missing_summary[missing_summary['Missing_Count']>0].shape[0]}</div>
                <div class="kpi-label">Columns with Missing Data</div>
            </div>
        </div>

        <h2>1. Automated Key Insights</h2>
        <div class="insight-list">
            <ul>
                {"".join([f"<li>{item}</li>" for item in insights.get('missing_insights', []) + insights.get('duplicate_insights', []) + insights.get('skewness_insights', []) + insights.get('correlation_insights', []) + insights.get('outlier_insights', [])])}
            </ul>
        </div>

        <h2>2. Data Quality & Missing Summary</h2>
        {df_to_html(missing_summary)}

        <h2>3. Statistical Summary - Numerical Features</h2>
        {df_to_html(num_summary)}

        <h2>4. Statistical Summary - Categorical Features</h2>
        {df_to_html(cat_summary)}

        <h2>5. Outlier Detection Summary (IQR Method)</h2>
        {df_to_html(outlier_summary)}

        <h2>6. Data Preprocessing & Modeling Recommendations</h2>
        <div class="recommendation-list">
            <ol>
                {"".join([f"<li>{rec}</li>" for rec in insights.get('recommendations', [])])}
            </ol>
        </div>
    </div>
</body>
</html>
"""
    return html
