import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional

# Set global matplotlib style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

def plot_missing_values(df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Plots a bar chart showing missing value percentage for columns with nulls.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if len(missing) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "🎉 No missing values found in dataset!", 
                ha="center", va="center", fontsize=14, color="#2e7d32", fontweight="bold")
        ax.axis("off")
        return fig
        
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct}).sort_values(by="Percentage (%)", ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, max(3, len(missing_df) * 0.4)))
    bars = ax.barh(missing_df.index, missing_df["Percentage (%)"], color="#e53935", alpha=0.85)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                va="center", ha="left", fontsize=10, fontweight="bold", color="#333333")
                
    ax.set_xlabel("Missing Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Missing Values Breakdown per Column", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, max(100, missing_pct.max() + 10))
    plt.tight_layout()
    return fig

def plot_univariate_num(df: pd.DataFrame, col: str, bins: int = 30) -> Optional[plt.Figure]:
    """
    Generates side-by-side Histogram + KDE and Boxplot for a numerical column.
    """
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return None
        
    fig, (ax_box, ax_hist) = plt.subplots(
        nrows=2, 
        ncols=1, 
        sharex=True, 
        gridspec_kw={"height_ratios": (0.25, 0.75)}, 
        figsize=(9, 5)
    )
    
    clean_series = df[col].dropna()
    mean_val = clean_series.mean()
    median_val = clean_series.median()
    
    # Boxplot
    sns.boxplot(x=clean_series, ax=ax_box, color="#42a5f5", flierprops=dict(markerfacecolor="#e53935", marker="o"))
    ax_box.set(xlabel="")
    ax_box.set_title(f"Distribution & Outlier Analysis for '{col}'", fontsize=13, fontweight="bold")
    
    # Histogram + KDE
    sns.histplot(clean_series, kde=True, bins=bins, ax=ax_hist, color="#1e88e5", edgecolor="white", alpha=0.6)
    
    # Add vertical lines for mean and median
    ax_hist.axvline(mean_val, color="#d32f2f", linestyle="--", linewidth=1.8, label=f"Mean: {mean_val:.2f}")
    ax_hist.axvline(median_val, color="#388e3c", linestyle="-.", linewidth=1.8, label=f"Median: {median_val:.2f}")
    
    ax_hist.set_xlabel(col, fontsize=11, fontweight="bold")
    ax_hist.set_ylabel("Frequency", fontsize=11, fontweight="bold")
    ax_hist.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc")
    
    plt.tight_layout()
    return fig

def plot_univariate_cat(df: pd.DataFrame, col: str, max_cats: int = 10) -> Optional[plt.Figure]:
    """
    Generates a horizontal countplot with percentages for a categorical column.
    """
    if col not in df.columns:
        return None
        
    counts = df[col].value_counts().head(max_cats)
    total = len(df[col].dropna())
    
    fig, ax = plt.subplots(figsize=(8, max(3, len(counts) * 0.45)))
    bars = ax.barh(counts.index.astype(str)[::-1], counts.values[::-1], color="#26a69a", alpha=0.85)
    
    for bar in bars:
        width = bar.get_width()
        pct = (width / total) * 100 if total > 0 else 0
        ax.text(width + (counts.max() * 0.01), bar.get_y() + bar.get_height()/2, 
                f"{width:,} ({pct:.1f}%)", va="center", ha="left", fontsize=10, color="#333333")
                
    ax.set_xlabel("Count", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {len(counts)} Categories for '{col}'", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(0, counts.max() * 1.25)
    plt.tight_layout()
    return fig

def plot_bivariate_num_num(df: pd.DataFrame, col1: str, col2: str, hue: Optional[str] = None) -> Optional[plt.Figure]:
    """
    Generates a scatter plot with regression line for two numerical columns.
    """
    if col1 not in df.columns or col2 not in df.columns:
        return None
        
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if hue and hue in df.columns:
        sns.scatterplot(data=df, x=col1, y=col2, hue=hue, ax=ax, alpha=0.7, palette="tab10")
    else:
        sns.scatterplot(data=df, x=col1, y=col2, ax=ax, color="#5c6bc0", alpha=0.6, s=50)
        sns.regplot(data=df, x=col1, y=col2, ax=ax, scatter=False, color="#e53935", line_kws={"linewidth": 2, "label": "Trendline"})
        ax.legend(loc="best")
        
    ax.set_xlabel(col1, fontsize=11, fontweight="bold")
    ax.set_ylabel(col2, fontsize=11, fontweight="bold")
    
    clean = df[[col1, col2]].dropna()
    corr_val = clean[col1].corr(clean[col2]) if not clean.empty else 0
    ax.set_title(f"Bivariate Scatter: '{col1}' vs '{col2}' (r = {corr_val:.3f})", fontsize=13, fontweight="bold")
    
    plt.tight_layout()
    return fig

def plot_bivariate_num_cat(df: pd.DataFrame, num_col: str, cat_col: str, plot_type: str = "box") -> Optional[plt.Figure]:
    """
    Generates a Boxplot or Violinplot for Numerical vs Categorical variables.
    """
    if num_col not in df.columns or cat_col not in df.columns:
        return None
        
    fig, ax = plt.subplots(figsize=(9, 5))
    
    top_cats = df[cat_col].value_counts().head(10).index
    filtered_df = df[df[cat_col].isin(top_cats)]
    
    if plot_type == "violin":
        sns.violinplot(data=filtered_df, x=cat_col, y=num_col, ax=ax, palette="Set2", inner="quartile")
    else:
        sns.boxplot(data=filtered_df, x=cat_col, y=num_col, ax=ax, palette="Set2")
        
    ax.set_xlabel(cat_col, fontsize=11, fontweight="bold")
    ax.set_ylabel(num_col, fontsize=11, fontweight="bold")
    ax.set_title(f"'{num_col}' Distribution across '{cat_col}' Categories", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(corr_df: pd.DataFrame) -> Optional[plt.Figure]:
    """
    Generates correlation matrix heatmap.
    """
    if corr_df.empty:
        return None
        
    fig, ax = plt.subplots(figsize=(min(12, max(6, len(corr_df) * 0.8)), min(10, max(5, len(corr_df) * 0.7))))
    
    sns.heatmap(
        corr_df, 
        annot=True, 
        fmt=".2f", 
        cmap="coolwarm", 
        vmin=-1, 
        vmax=1, 
        linewidths=0.5, 
        ax=ax, 
        cbar_kws={"shrink": 0.8}
    )
    
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    return fig

def plot_outliers_annotated(df: pd.DataFrame, col: str, lower_bound: float, upper_bound: float, method_label: str = "IQR") -> Optional[plt.Figure]:
    """
    Generates annotated box plot showing outlier threshold boundaries.
    """
    if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
        return None
        
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=df[col], ax=ax, color="#ab47bc", flierprops=dict(markerfacecolor="#e53935", marker="D", markersize=6))
    
    ax.axvline(lower_bound, color="#e53935", linestyle="--", linewidth=1.5, label=f"Lower Threshold ({lower_bound:.2f})")
    ax.axvline(upper_bound, color="#e53935", linestyle="--", linewidth=1.5, label=f"Upper Threshold ({upper_bound:.2f})")
    
    ax.set_xlabel(col, fontsize=11, fontweight="bold")
    ax.set_title(f"Outlier Detection Thresholds ({method_label}) for '{col}'", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white")
    plt.tight_layout()
    return fig
