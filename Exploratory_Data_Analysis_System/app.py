import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src folder to python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_data, get_dataset_info
from src.cleaner import get_missing_summary, get_duplicate_summary, handle_missing_values, remove_duplicates, convert_column_dtypes
from src.analyzer import get_numerical_summary, get_categorical_summary, get_univariate_metrics, get_bivariate_analysis, compute_correlation_matrix, get_top_correlations
from src.outliers import detect_all_outliers, detect_outliers_iqr, detect_outliers_zscore, handle_outliers
from src.visualizer import (plot_missing_values, plot_univariate_num, plot_univariate_cat, 
                            plot_bivariate_num_num, plot_bivariate_num_cat, plot_correlation_heatmap, plot_outliers_annotated)
from src.insights import generate_automated_insights
from src.reporter import generate_html_report, generate_markdown_report

# Page Config
st.set_page_config(
    page_title="EDA & Data Insights System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Data Science Dashboard
st.markdown("""
<style>
    /* Main container background & font */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Header title styling */
    .title-text {
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
        font-weight: 800;
        color: #1a237e;
        font-size: 2.2rem;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        font-family: 'Segoe UI', Roboto, sans-serif;
        color: #546e7a;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Card Styling */
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 18px 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.04), 0 1px 3px rgba(0,0,0,0.08);
        text-align: center;
        border-left: 4px solid #3f51b5;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e88e5;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #78909c;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Insight Alert Box */
    .insight-card {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    .warning-card {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }
    .danger-card {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 0.95rem;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #ffffff;
        border-radius: 6px 6px 0 0;
        padding-left: 16px;
        padding-right: 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = None
if "df" not in st.session_state:
    st.session_state["df"] = None
if "dataset_name" not in st.session_state:
    st.session_state["dataset_name"] = "Dataset"

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("Navigation")

nav_choice = st.sidebar.radio(
    "Select Module:",
    [
        "📌 Dataset Upload & Overview",
        "🧹 Data Cleaning & Quality",
        "📊 Statistical Summary",
        "📈 Univariate Analysis",
        "🔀 Bivariate Analysis",
        "🔗 Correlation Analysis",
        "🎯 Outlier Detection",
        "💡 Automated Insights",
        "📄 Report & Data Export"
    ]
)

# Dataset Loading Section in Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Input")
data_source = st.sidebar.radio("Data Source:", ["Upload CSV", "Sample Dataset"])

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload a CSV dataset", type=["csv"])
    if uploaded_file is not None:
        df_loaded, msg = load_data(uploaded_file)
        if msg.startswith("Success"):
            st.session_state["df_raw"] = df_loaded.copy()
            if st.session_state["df"] is None or st.sidebar.button("Reload Original File"):
                st.session_state["df"] = df_loaded.copy()
            st.session_state["dataset_name"] = uploaded_file.name
        else:
            st.sidebar.error(msg)
else:
    sample_option = st.sidebar.selectbox("Choose Sample Dataset:", ["Employee Analytics", "Housing Prices"])
    sample_path = "data/sample_employee_data.csv" if sample_option == "Employee Analytics" else "data/housing_prices.csv"
    if os.path.exists(sample_path):
        df_loaded, msg = load_data(sample_path)
        if msg.startswith("Success"):
            if st.session_state["df_raw"] is None or st.session_state["dataset_name"] != sample_option:
                st.session_state["df_raw"] = df_loaded.copy()
                st.session_state["df"] = df_loaded.copy()
                st.session_state["dataset_name"] = sample_option
    else:
        st.sidebar.error(f"Sample dataset not found at {sample_path}")

# Check if data is available
df = st.session_state["df"]

if df is None:
    st.markdown("<h1 class='title-text'>Exploratory Data Analysis & Insights System</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>An automated, intelligent EDA engine for data cleaning, statistical diagnostics, and visual insights generation.</p>", unsafe_allow_html=True)
    
    st.info("👈 **Please upload a CSV file or select a sample dataset from the sidebar to get started!**")
    
    st.markdown("### ✨ Key System Features:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        #### 1. Data Cleaning
        - Missing value detection & imputation
        - Duplicate row removal
        - Data type conversion & validation
        """)
    with col2:
        st.markdown("""
        #### 2. Visual Analytics
        - Univariate distributions (Histograms, Box plots)
        - Bivariate scatter, bar, & violin plots
        - Correlation heatmaps & top pairs
        """)
    with col3:
        st.markdown("""
        #### 3. Automated Intelligence
        - IQR & Z-score outlier detection
        - Natural language insights generator
        - Downloadable HTML & Markdown reports
        """)
    st.stop()

# Header Information
info = get_dataset_info(df)
st.markdown(f"<h1 class='title-text'>Exploratory Data Analysis - {st.session_state['dataset_name']}</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Automated diagnostics, data quality auditing, and interactive visual insights.</p>", unsafe_allow_html=True)

# Dataset Status indicator in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Dataset:** `{st.session_state['dataset_name']}`")
st.sidebar.markdown(f"**Rows:** {info['rows']:,} | **Cols:** {info['cols']}")
if st.sidebar.button("🔄 Reset Cleaned Data to Raw"):
    st.session_state["df"] = st.session_state["df_raw"].copy()
    st.rerun()

# ==========================================
# MODULE 1: DATASET OVERVIEW
# ==========================================
if nav_choice == "📌 Dataset Upload & Overview":
    st.subheader("📌 Dataset Metrics & Schema Overview")
    
    # KPI Metric Cards
    col1, col2, col3, col4, col5 = st.columns(5)
    dup_info = get_duplicate_summary(df)
    missing_summary = get_missing_summary(df)
    missing_cols_count = missing_summary[missing_summary["Missing_Count"] > 0].shape[0]
    
    with col1:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{info['rows']:,}</div><div class='metric-lbl'>Total Rows</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{info['cols']}</div><div class='metric-lbl'>Total Columns</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(info['numerical_cols'])}</div><div class='metric-lbl'>Numerical Features</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{len(info['categorical_cols'])}</div><div class='metric-lbl'>Categorical Features</div></div>", unsafe_allow_html=True)
    with col5:
        st.markdown(f"<div class='metric-card'><div class='metric-val'>{dup_info['duplicate_count']}</div><div class='metric-lbl'>Duplicate Rows</div></div>", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for Overview details
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "🏷️ Column Types & Memory", "🔢 Unique Values", "📊 Summary Statistics"])
    
    with tab1:
        st.write("##### First 10 Records (Head)")
        st.dataframe(df.head(10), use_container_width=True)
        st.write("##### Last 10 Records (Tail)")
        st.dataframe(df.tail(10), use_container_width=True)
        
    with tab2:
        col_type_df = pd.DataFrame({
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values,
            "Null Percentage (%)": ((df.isnull().sum() / len(df)) * 100).round(2).values
        })
        st.dataframe(col_type_df, use_container_width=True)
        st.caption(f"Estimated Dataset Memory Usage: **{info['memory_mb']} MB**")
        
    with tab3:
        unique_df = pd.DataFrame({
            "Column Name": df.columns,
            "Unique Values": [df[col].nunique() for col in df.columns],
            "Sample Unique Values": [str(df[col].dropna().unique()[:5].tolist()) for col in df.columns]
        })
        st.dataframe(unique_df, use_container_width=True)
        
    with tab4:
        st.write("##### Numerical Feature Summary")
        num_sum = get_numerical_summary(df)
        if not num_sum.empty:
            st.dataframe(num_sum, use_container_width=True)
        else:
            st.info("No numerical features found.")
            
        st.write("##### Categorical Feature Summary")
        cat_sum = get_categorical_summary(df)
        if not cat_sum.empty:
            st.dataframe(cat_sum, use_container_width=True)
        else:
            st.info("No categorical features found.")

# ==========================================
# MODULE 2: DATA CLEANING & QUALITY
# ==========================================
elif nav_choice == "🧹 Data Cleaning & Quality":
    st.subheader("🧹 Interactive Data Cleaning Engine")
    
    tab_clean1, tab_clean2, tab_clean3 = st.tabs(["❓ Missing Values Handling", "👯 Duplicate Removal", "⚙️ Type Conversion"])
    
    # 1. Missing Values Tab
    with tab_clean1:
        st.write("##### Missing Values Breakdown")
        missing_df = get_missing_summary(df)
        missing_cols = missing_df[missing_df["Missing_Count"] > 0]
        
        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            if not missing_cols.empty:
                st.dataframe(missing_cols, use_container_width=True)
            else:
                st.success("🎉 No missing values detected in the dataset!")
                
        with col_m2:
            fig_missing = plot_missing_values(df)
            if fig_missing:
                st.pyplot(fig_missing)
                
        if not missing_cols.empty:
            st.markdown("---")
            st.write("##### Impute / Treat Missing Values")
            
            target_col = st.selectbox("Select Column to Treat:", missing_cols["Column"].tolist())
            col_dtype = df[target_col].dtype
            
            is_num = pd.api.types.is_numeric_dtype(col_dtype)
            if is_num:
                strategy_opts = ["Mean", "Median", "Mode", "Drop Rows with Missing", "Drop Entire Column", "Forward Fill (ffill)", "Backward Fill (bfill)"]
            else:
                strategy_opts = ["Mode", "Drop Rows with Missing", "Drop Entire Column", "Forward Fill (ffill)", "Backward Fill (bfill)", "Constant Value"]
                
            strategy = st.selectbox("Select Imputation Strategy:", strategy_opts)
            custom_val = None
            if strategy == "Constant Value":
                custom_val = st.text_input("Enter Constant Fill Value:", value="Unknown")
                
            if st.button(f"Apply {strategy} Imputation on '{target_col}'"):
                strat_key = strategy.lower().replace(" (ffill)", "").replace(" (bfill)", "").replace(" drop rows with missing", "_rows").replace(" drop entire column", "_col")
                if "mean" in strat_key: strat_map = {target_col: "mean"}
                elif "median" in strat_key: strat_map = {target_col: "median"}
                elif "mode" in strat_key: strat_map = {target_col: "mode"}
                elif "rows" in strat_key: strat_map = {target_col: "drop_rows"}
                elif "col" in strat_key: strat_map = {target_col: "drop_col"}
                elif "ffill" in strat_key: strat_map = {target_col: "ffill"}
                elif "bfill" in strat_key: strat_map = {target_col: "bfill"}
                else: strat_map = {target_col: "constant"}
                
                custom_map = {target_col: custom_val} if custom_val else None
                st.session_state["df"] = handle_missing_values(st.session_state["df"], strat_map, custom_map)
                st.success(f"Successfully applied '{strategy}' on '{target_col}'!")
                st.rerun()

    # 2. Duplicate Removal Tab
    with tab_clean2:
        dup_info = get_duplicate_summary(df)
        st.write(f"##### Duplicate Rows Audit")
        st.info(f"Detected **{dup_info['duplicate_count']:,} duplicate rows** ({dup_info['duplicate_percent']}%) in the dataset.")
        
        if dup_info["duplicate_count"] > 0:
            st.write("##### Sample Duplicate Rows")
            st.dataframe(df[df.duplicated(keep=False)].head(10), use_container_width=True)
            
            if st.button("🧹 Remove All Duplicate Rows"):
                st.session_state["df"] = remove_duplicates(st.session_state["df"])
                st.success("Successfully removed duplicate records!")
                st.rerun()
        else:
            st.success("Dataset is free of duplicate records!")

    # 3. Data Type Conversion Tab
    with tab_clean3:
        st.write("##### Convert Column Data Types")
        c1, c2 = st.columns(2)
        with c1:
            col_to_convert = st.selectbox("Select Column:", df.columns)
        with c2:
            target_type = st.selectbox("Target Data Type:", ["numeric", "category", "string", "datetime"])
            
        if st.button(f"Convert '{col_to_convert}' to {target_type}"):
            df_conv, errs = convert_column_dtypes(st.session_state["df"], {col_to_convert: target_type})
            if errs:
                st.error("\n".join(errs))
            else:
                st.session_state["df"] = df_conv
                st.success(f"Successfully converted '{col_to_convert}' to {target_type}!")
                st.rerun()

# ==========================================
# MODULE 3: STATISTICAL SUMMARY
# ==========================================
elif nav_choice == "📊 Statistical Summary":
    st.subheader("📊 Complete Statistical Diagnostics")
    
    st.write("##### Numerical Summary Matrix")
    num_summary = get_numerical_summary(df)
    if not num_summary.empty:
        st.dataframe(num_summary, use_container_width=True)
    else:
        st.info("No numerical attributes present in dataset.")
        
    st.write("##### Categorical Summary Matrix")
    cat_summary = get_categorical_summary(df)
    if not cat_summary.empty:
        st.dataframe(cat_summary, use_container_width=True)
    else:
        st.info("No categorical attributes present in dataset.")

# ==========================================
# MODULE 4: UNIVARIATE ANALYSIS
# ==========================================
elif nav_choice == "📈 Univariate Analysis":
    st.subheader("📈 Univariate Distribution & Metric Explorer")
    
    selected_col = st.selectbox("Select Column to Analyze:", df.columns)
    
    metrics = get_univariate_metrics(df, selected_col)
    
    col_u1, col_u2 = st.columns([1, 2])
    
    with col_u1:
        st.markdown(f"#### Stats for `{selected_col}`")
        st.markdown(f"- **Data Type:** `{df[selected_col].dtype}`")
        st.markdown(f"- **Total Records:** `{metrics['total_count']:,}`")
        st.markdown(f"- **Missing Count:** `{metrics['missing_count']} ({metrics['missing_percentage']}%)`")
        st.markdown(f"- **Unique Count:** `{metrics['unique_count']:,}`")
        
        if metrics["is_numeric"]:
            st.markdown("---")
            st.markdown(f"- **Mean:** `{metrics.get('mean')}`")
            st.markdown(f"- **Median:** `{metrics.get('median')}`")
            st.markdown(f"- **Mode:** `{metrics.get('mode')}`")
            st.markdown(f"- **Std Dev:** `{metrics.get('std')}`")
            st.markdown(f"- **Min:** `{metrics.get('min')}` | **Max:** `{metrics.get('max')}`")
            st.markdown(f"- **IQR:** `{metrics.get('iqr')}` (`Q25: {metrics.get('q25')}`, `Q75: {metrics.get('q75')}`)")
            st.markdown(f"- **Skewness:** `{metrics.get('skewness')}`")
            st.markdown(f"- **Kurtosis:** `{metrics.get('kurtosis')}`")
        else:
            st.markdown("---")
            st.markdown("**Top Value Counts:**")
            st.write(pd.DataFrame(list(metrics.get("value_counts", {}).items()), columns=["Category", "Count"]))
            
    with col_u2:
        if metrics["is_numeric"]:
            bin_count = st.slider("Histogram Bins:", 10, 100, 30)
            fig = plot_univariate_num(df, selected_col, bins=bin_count)
            if fig:
                st.pyplot(fig)
        else:
            max_cats = st.slider("Max Categories to Display:", 5, 20, 10)
            fig = plot_univariate_cat(df, selected_col, max_cats=max_cats)
            if fig:
                st.pyplot(fig)

# ==========================================
# MODULE 5: BIVARIATE ANALYSIS
# ==========================================
elif nav_choice == "🔀 Bivariate Analysis":
    st.subheader("🔀 Bivariate Feature Relationship Analysis")
    
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        var1 = st.selectbox("Select Feature 1 (X-Axis):", df.columns, index=0)
    with col_b2:
        default_idx = 1 if len(df.columns) > 1 else 0
        var2 = st.selectbox("Select Feature 2 (Y-Axis):", df.columns, index=default_idx)
        
    if var1 == var2:
        st.warning("Please select two distinct variables for bivariate relationship analysis.")
    else:
        bivariate_res = get_bivariate_analysis(df, var1, var2)
        b_type = bivariate_res.get("type")
        
        if b_type == "num_num":
            st.info(f"**Pearson Correlation (r):** `{bivariate_res['pearson']}` | **Spearman Correlation (ρ):** `{bivariate_res['spearman']}`")
            hue_col = st.selectbox("Optional Color Grouping (Hue):", [None] + info["categorical_cols"])
            fig = plot_bivariate_num_num(df, var1, var2, hue=hue_col)
            if fig:
                st.pyplot(fig)
                
        elif b_type == "num_cat":
            num_var = bivariate_res["num_col"]
            cat_var = bivariate_res["cat_col"]
            st.write(f"##### Grouped Statistics (`{num_var}` across `{cat_var}` categories)")
            st.dataframe(bivariate_res["grouped_stats"], use_container_width=True)
            
            p_type = st.radio("Plot Type:", ["Boxplot", "Violin Plot"], horizontal=True)
            fig = plot_bivariate_num_cat(df, num_var, cat_var, plot_type="violin" if p_type == "Violin Plot" else "box")
            if fig:
                st.pyplot(fig)
                
        elif b_type == "cat_cat":
            st.write("##### Contingency Table / Crosstab")
            st.dataframe(bivariate_res["crosstab"], use_container_width=True)

# ==========================================
# MODULE 6: CORRELATION ANALYSIS
# ==========================================
elif nav_choice == "🔗 Correlation Analysis":
    st.subheader("🔗 Feature Correlation Analysis & Heatmap")
    
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        st.warning("At least two numerical columns are required for correlation analysis.")
    else:
        corr_method = st.radio("Correlation Method:", ["pearson", "spearman"], horizontal=True)
        corr_matrix = compute_correlation_matrix(df, method=corr_method)
        
        tab_c1, tab_c2 = st.tabs(["🔥 Heatmap", "🔍 Top Correlated Feature Pairs"])
        
        with tab_c1:
            fig_corr = plot_correlation_heatmap(corr_matrix)
            if fig_corr:
                st.pyplot(fig_corr)
                
        with tab_c2:
            thresh = st.slider("Filter Absolute Correlation Threshold (|r| ≥):", 0.1, 0.95, 0.5, step=0.05)
            top_pairs = get_top_correlations(corr_matrix, threshold=thresh)
            if not top_pairs.empty:
                st.dataframe(top_pairs, use_container_width=True)
            else:
                st.info(f"No feature pairs found with correlation |r| ≥ {thresh}")

# ==========================================
# MODULE 7: OUTLIER DETECTION
# ==========================================
elif nav_choice == "🎯 Outlier Detection":
    st.subheader("🎯 Outlier Detection & Treatment Engine")
    
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("No numerical features available for outlier detection.")
    else:
        out_method = st.radio("Detection Algorithm:", ["IQR (Interquartile Range)", "Z-Score"], horizontal=True)
        
        if "IQR" in out_method:
            iqr_factor = st.slider("IQR Multiplier Factor:", 1.0, 3.0, 1.5, step=0.1)
            outlier_df = detect_all_outliers(df, method="iqr", factor=iqr_factor)
        else:
            z_thresh = st.slider("Z-Score Threshold:", 2.0, 4.5, 3.0, step=0.1)
            outlier_df = detect_all_outliers(df, method="zscore", z_threshold=z_thresh)
            
        st.write("##### Outlier Summary Across Numerical Attributes")
        st.dataframe(outlier_df, use_container_width=True)
        
        st.markdown("---")
        st.write("##### Single Column Outlier Inspector & Action")
        
        col_out = st.selectbox("Select Column to Inspect & Treat:", num_cols)
        
        if "IQR" in out_method:
            res = detect_outliers_iqr(df, col_out, factor=iqr_factor)
        else:
            res = detect_outliers_zscore(df, col_out, threshold=z_thresh)
            
        col_o1, col_o2 = st.columns([1, 2])
        with col_o1:
            st.metric("Outlier Count", f"{res['outlier_count']:,}")
            st.metric("Outlier Percentage", f"{res['outlier_percentage']}%")
            st.write(f"**Lower Bound:** `{res['lower_bound']}`")
            st.write(f"**Upper Bound:** `{res['upper_bound']}`")
            
            action = st.selectbox("Select Outlier Action:", ["Keep (Do Nothing)", "Cap Outliers (Winsorization)", "Remove Outlier Rows"])
            if st.button(f"Apply '{action}' on '{col_out}'"):
                if action != "Keep (Do Nothing)":
                    act_key = "cap" if "Cap" in action else "remove"
                    method_key = "iqr" if "IQR" in out_method else "zscore"
                    st.session_state["df"] = handle_outliers(
                        st.session_state["df"], 
                        col_out, 
                        action=act_key, 
                        method=method_key, 
                        factor=iqr_factor if "IQR" in out_method else 1.5, 
                        z_threshold=z_thresh if "Z-Score" in out_method else 3.0
                    )
                    st.success(f"Applied '{action}' to '{col_out}'!")
                    st.rerun()
                    
        with col_o2:
            fig_out = plot_outliers_annotated(df, col_out, res['lower_bound'], res['upper_bound'], method_label="IQR" if "IQR" in out_method else "Z-Score")
            if fig_out:
                st.pyplot(fig_out)

# ==========================================
# MODULE 8: AUTOMATED INSIGHTS
# ==========================================
elif nav_choice == "💡 Automated Insights":
    st.subheader("💡 Dynamic Automated Insights & ML Recommendations")
    
    insights = generate_automated_insights(df)
    
    st.markdown("### 📋 Executive Dataset Overview")
    for line in insights["overview"]:
        st.markdown(f"- {line}")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_i1, col_i2 = st.columns(2)
    
    with col_i1:
        st.markdown("### ⚠️ Data Quality & Missing Alerts")
        for line in insights["missing_insights"] + insights["duplicate_insights"]:
            st.markdown(f"<div class='warning-card'>{line}</div>", unsafe_allow_html=True)
            
        st.markdown("### 🎯 Outlier & Distribution Alerts")
        for line in insights["skewness_insights"] + insights["outlier_insights"]:
            st.markdown(f"<div class='danger-card'>{line}</div>", unsafe_allow_html=True)
            
    with col_i2:
        st.markdown("### 🔗 Correlation & Pattern Alerts")
        for line in insights["correlation_insights"]:
            st.markdown(f"<div class='insight-card'>{line}</div>", unsafe_allow_html=True)
            
        st.markdown("### 🚀 Actionable Preprocessing Recommendations")
        for i, rec in enumerate(insights["recommendations"], 1):
            st.markdown(f"**{i}.** {rec}")

# ==========================================
# MODULE 9: REPORT & DATA EXPORT
# ==========================================
elif nav_choice == "📄 Report & Data Export":
    st.subheader("📄 Automated EDA Summary Report & Cleaned Data Export")
    
    st.markdown("Generate and download a full exploratory data analysis report or export the cleaned CSV dataset.")
    
    insights = generate_automated_insights(df)
    
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.markdown("### 💾 Export Cleaned Dataset")
        st.write("Download the current DataFrame containing all data cleaning, missing value imputation, duplicate removal, and outlier capping modifications.")
        
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Cleaned Dataset (CSV)",
            data=csv_data,
            file_name=f"cleaned_{st.session_state['dataset_name']}",
            mime="text/csv"
        )
        
    with col_exp2:
        st.markdown("### 📄 Export Analysis Summary Report")
        st.write("Download a comprehensive audit report formatted in responsive HTML or GitHub Markdown.")
        
        html_report = generate_html_report(df, dataset_name=st.session_state["dataset_name"], insights=insights)
        md_report = generate_markdown_report(df, dataset_name=st.session_state["dataset_name"], insights=insights)
        
        st.download_button(
            label="🌐 Download Full EDA Report (HTML)",
            data=html_report,
            file_name=f"EDA_Report_{st.session_state['dataset_name']}.html",
            mime="text/html"
        )
        
        st.download_button(
            label="📝 Download EDA Report (Markdown)",
            data=md_report,
            file_name=f"EDA_Report_{st.session_state['dataset_name']}.md",
            mime="text/markdown"
        )

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #999; font-size: 0.85rem;'>Exploratory Data Analysis and Data Insights System • Built with Python, Streamlit & Scikit-Learn</p>", unsafe_allow_html=True)
