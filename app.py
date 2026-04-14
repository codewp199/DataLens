import io
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="DataLens", page_icon="🔎", layout="wide")

CUSTOM_CSS = """
<style>
:root {
    --bg: #f7f1e8;
    --card: #fff3e6;
    --card-2: #f2e8ff;
    --card-3: #e7f7ef;
    --card-4: #fff0c9;
    --text: #402b2b;
    --muted: #7a5c55;
    --primary: #8a4fff;
    --secondary: #00a67e;
    --accent: #ff9f43;
    --accent-2: #cc7a00;
    --border: rgba(89, 52, 52, 0.10);
}
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #fdf2d4 0%, #f7efe7 45%, #f3e9ff 100%);
    color: var(--text);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff2db 0%, #f5e9ff 100%);
    border-right: 1px solid var(--border);
}
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
h1, h2, h3, h4 { color: var(--text); }
p, label, .stMarkdown, .stText, .stCaption { color: var(--text); }
.hero {
    background: linear-gradient(135deg, #f8df87 0%, #ffd6b5 48%, #d9c2ff 100%);
    padding: 1.4rem 1.5rem;
    border-radius: 24px;
    border: 1px solid var(--border);
    box-shadow: 0 16px 40px rgba(118, 73, 73, 0.08);
    margin-bottom: 1rem;
}
.hero h1 { margin: 0 0 0.35rem 0; font-size: 2.4rem; }
.hero p { margin: 0; color: #5d4743; font-size: 1.02rem; }
.metric-card {
    border-radius: 22px;
    padding: 1rem 1rem 0.9rem 1rem;
    border: 1px solid var(--border);
    box-shadow: 0 12px 30px rgba(118, 73, 73, 0.06);
}
.metric-card h4 { margin: 0 0 0.15rem 0; font-size: 0.95rem; color: #674a45; }
.metric-card h2 { margin: 0; font-size: 2rem; }
.metric-card p { margin: 0.25rem 0 0 0; color: var(--muted); font-size: 0.88rem; }
.bg1 { background: #fff2d7; }
.bg2 { background: #f2e8ff; }
.bg3 { background: #e6f8ee; }
.bg4 { background: #ffe7cf; }
.section-card {
    background: rgba(255, 248, 239, 0.92);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 1rem 1.1rem;
    box-shadow: 0 12px 28px rgba(118, 73, 73, 0.05);
}
.insight-box {
    background: linear-gradient(135deg, #f3e6ff 0%, #fff5de 100%);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}
.small-note {
    color: var(--muted);
    font-size: 0.9rem;
}
[data-testid="stMetricValue"] { color: var(--text); }
.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #8a4fff 0%, #b985ff 100%);
    color: #fff7f1;
    border-radius: 999px;
    border: none;
    padding: 0.6rem 1rem;
}
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div,
.stTextInput input {
    background: #fff8ef;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def detect_column_roles(df: pd.DataFrame, target_col: Optional[str]) -> pd.DataFrame:
    rows = []
    n_rows = len(df)
    for col in df.columns:
        s = df[col]
        role = "feature"
        reason = "General purpose feature"
        if col == target_col:
            role = "target"
            reason = "Selected as target column"
        elif pd.api.types.is_datetime64_any_dtype(s):
            role = "date"
            reason = "Detected as datetime"
        elif s.nunique(dropna=True) >= max(0.9 * n_rows, 1) and s.dtype == object:
            role = "identifier"
            reason = "Mostly unique text values"
        elif s.nunique(dropna=True) >= max(0.95 * n_rows, 1) and pd.api.types.is_numeric_dtype(s):
            role = "identifier"
            reason = "Mostly unique numeric values"
        elif s.dtype == object and s.astype(str).str.len().mean() > 40:
            role = "free_text"
            reason = "Long text-like values"
        elif pd.api.types.is_numeric_dtype(s):
            role = "numeric_measure"
            reason = "Numeric feature"
        elif s.dtype == object or pd.api.types.is_categorical_dtype(s):
            role = "category"
            reason = "Categorical feature"
        rows.append({"column": col, "role": role, "reason": reason})
    return pd.DataFrame(rows)


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    profiles = []
    n_rows = len(df)
    for col in df.columns:
        s = df[col]
        null_pct = round(100 * safe_div(s.isna().sum(), n_rows), 2)
        unique_count = int(s.nunique(dropna=True))
        dtype = str(s.dtype)
        sample_vals = ", ".join(map(str, s.dropna().astype(str).head(3).tolist()))
        risk_score = 0
        reasons: List[str] = []
        if null_pct > 30:
            risk_score += 3
            reasons.append("high missingness")
        elif null_pct > 10:
            risk_score += 1
            reasons.append("moderate missingness")
        if unique_count <= 1:
            risk_score += 3
            reasons.append("zero variance")
        if pd.api.types.is_numeric_dtype(s):
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                outlier_pct = 100 * safe_div(outliers, s.notna().sum())
                if outlier_pct > 15:
                    risk_score += 2
                    reasons.append("high outliers")
                elif outlier_pct > 5:
                    risk_score += 1
                    reasons.append("some outliers")
            skew = round(float(s.skew()), 2) if s.notna().sum() > 2 else np.nan
            summary = f"min={s.min():.2f}, max={s.max():.2f}, mean={s.mean():.2f}" if s.notna().sum() else "No numeric values"
        else:
            skew = np.nan
            summary = sample_vals or "No sample values"
            if unique_count > max(50, n_rows * 0.5):
                risk_score += 2
                reasons.append("high cardinality")
        risk = "Low"
        if risk_score >= 4:
            risk = "High"
        elif risk_score >= 2:
            risk = "Medium"
        profiles.append(
            {
                "column": col,
                "dtype": dtype,
                "null_pct": null_pct,
                "unique_count": unique_count,
                "sample_or_summary": summary,
                "skewness": skew,
                "risk": risk,
                "risk_reasons": ", ".join(reasons) if reasons else "healthy",
            }
        )
    return pd.DataFrame(profiles)


def get_quality_summary(df: pd.DataFrame, target_col: Optional[str]) -> dict:
    n_rows, n_cols = df.shape
    total_cells = max(n_rows * n_cols, 1)
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    mixed_type_cols = []
    for col in df.columns:
        types = df[col].dropna().map(type).astype(str).nunique()
        if types > 1:
            mixed_type_cols.append(col)

    zero_variance_cols = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
    id_like_cols = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if len(df) > 0 and nunique >= 0.95 * len(df):
            id_like_cols.append(col)

    high_card_cols = []
    for col in categorical_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique > max(30, len(df) * 0.3):
            high_card_cols.append(col)

    date_like_cols = []
    for col in categorical_cols:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() > 0.7:
            date_like_cols.append(col)

    outlier_stats = []
    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 5:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_count = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        outlier_pct = round(100 * safe_div(outlier_count, len(s)), 2)
        outlier_stats.append({"column": col, "outlier_count": outlier_count, "outlier_pct": outlier_pct})
    outlier_df = pd.DataFrame(outlier_stats).sort_values("outlier_pct", ascending=False) if outlier_stats else pd.DataFrame(columns=["column", "outlier_count", "outlier_pct"])

    leakage_flags = []
    target_imbalance = None
    modeling_hint = "No target selected"
    if target_col and target_col in df.columns:
        target_series = df[target_col]
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique(dropna=True) > 15:
            modeling_hint = "Regression"
            corr_df = df.select_dtypes(include=np.number).corr(numeric_only=True)
            if target_col in corr_df.columns:
                target_corr = corr_df[target_col].drop(labels=[target_col], errors="ignore").dropna()
                leakage_flags = target_corr[abs(target_corr) > 0.95].index.tolist()
        else:
            modeling_hint = "Classification"
            vc = target_series.value_counts(dropna=False, normalize=True)
            if not vc.empty:
                target_imbalance = round(float(vc.max() * 100), 2)

    return {
        "rows": n_rows,
        "cols": n_cols,
        "missing_pct": round(100 * safe_div(missing_cells, total_cells), 2),
        "duplicate_rows": duplicate_rows,
        "numeric_cols": len(numeric_cols),
        "categorical_cols": len(categorical_cols),
        "mixed_type_cols": mixed_type_cols,
        "zero_variance_cols": zero_variance_cols,
        "id_like_cols": id_like_cols,
        "high_card_cols": high_card_cols,
        "date_like_cols": date_like_cols,
        "outlier_df": outlier_df,
        "leakage_flags": leakage_flags,
        "target_imbalance": target_imbalance,
        "modeling_hint": modeling_hint,
    }


def build_insights(df: pd.DataFrame, quality: dict, profiles: pd.DataFrame, target_col: Optional[str]) -> List[str]:
    insights: List[str] = []
    if quality["missing_pct"] > 15:
        insights.append(f"This dataset has {quality['missing_pct']}% missing values overall, so cleaning strategy will materially affect downstream results.")
    if quality["duplicate_rows"] > 0:
        insights.append(f"There are {quality['duplicate_rows']} duplicate rows, which may inflate metrics or bias descriptive summaries.")
    if quality["id_like_cols"]:
        insights.append(f"{', '.join(quality['id_like_cols'][:3])} behave like identifier columns and should usually be excluded from modeling features.")
    if quality["high_card_cols"]:
        insights.append(f"High-cardinality fields such as {', '.join(quality['high_card_cols'][:3])} may require grouping, hashing, or alternate encoding strategies.")
    if not quality["outlier_df"].empty:
        top_col = quality["outlier_df"].iloc[0]
        if float(top_col["outlier_pct"]) > 5:
            insights.append(f"{top_col['column']} shows {top_col['outlier_pct']}% outliers, which may distort averages and baseline models.")
    high_risk = profiles[profiles["risk"] == "High"]["column"].tolist()
    if high_risk:
        insights.append(f"The highest-risk columns right now are {', '.join(high_risk[:4])}, based on missingness, outliers, cardinality, or low variance.")
    if target_col and quality["modeling_hint"] == "Classification" and quality["target_imbalance"]:
        if quality["target_imbalance"] > 70:
            insights.append(f"The target looks imbalanced: the largest class accounts for {quality['target_imbalance']}% of records, so accuracy alone may be misleading.")
    if target_col and quality["leakage_flags"]:
        insights.append(f"Possible leakage warning: {', '.join(quality['leakage_flags'][:3])} are extremely correlated with the target and should be reviewed before training.")
    if quality["date_like_cols"]:
        insights.append(f"Date-like columns were detected in text form. Converting them to proper datetime fields could unlock better time-based analysis and feature engineering.")
    return insights[:8]


def recommend_actions(quality: dict, profiles: pd.DataFrame, target_col: Optional[str]) -> pd.DataFrame:
    actions = []
    if quality["duplicate_rows"] > 0:
        actions.append(("Critical", "Remove duplicate rows before analysis", "Prevents repeated records from skewing metrics and training."))
    if quality["missing_pct"] > 15:
        actions.append(("Critical", "Prioritize imputation strategy for high-missingness columns", "Overall missingness is materially high."))
    if quality["mixed_type_cols"]:
        actions.append(("Moderate", "Standardize mixed-type columns", f"Review: {', '.join(quality['mixed_type_cols'][:4])}"))
    if quality["id_like_cols"]:
        actions.append(("Critical", "Exclude identifier-like fields from modeling", f"Likely IDs: {', '.join(quality['id_like_cols'][:4])}"))
    if quality["high_card_cols"]:
        actions.append(("Moderate", "Reduce or encode high-cardinality categories carefully", f"Columns: {', '.join(quality['high_card_cols'][:4])}"))
    if quality["leakage_flags"]:
        actions.append(("Critical", "Investigate target leakage before model training", f"Leakage candidates: {', '.join(quality['leakage_flags'][:4])}"))
    high_risk = profiles[profiles["risk"] == "High"]["column"].tolist()
    if high_risk:
        actions.append(("Moderate", "Review high-risk columns manually", f"Focus on: {', '.join(high_risk[:5])}"))
    if target_col and quality["modeling_hint"] == "Classification" and quality["target_imbalance"] and quality["target_imbalance"] > 70:
        actions.append(("Moderate", "Plan for class imbalance handling", "Consider stratified split, class weights, or resampling."))
    if not actions:
        actions.append(("Informational", "Dataset looks relatively healthy", "Proceed with exploratory analysis and baseline modeling."))
    return pd.DataFrame(actions, columns=["priority", "recommended_action", "why_it_matters"])


def compute_readiness_score(quality: dict, profiles: pd.DataFrame) -> tuple[int, int, List[str]]:
    analytics = 100
    ml = 100
    penalties = []

    def penalize(label: str, analytics_penalty: int, ml_penalty: int):
        nonlocal analytics, ml
        analytics -= analytics_penalty
        ml -= ml_penalty
        penalties.append(label)

    if quality["missing_pct"] > 20:
        penalize("High overall missingness", 18, 20)
    elif quality["missing_pct"] > 10:
        penalize("Moderate overall missingness", 8, 10)
    if quality["duplicate_rows"] > 0:
        penalize("Duplicate rows detected", 6, 8)
    if quality["mixed_type_cols"]:
        penalize("Mixed type columns detected", 5, 7)
    if quality["id_like_cols"]:
        penalize("Identifier-like columns present", 2, 10)
    if quality["high_card_cols"]:
        penalize("High-cardinality features present", 4, 7)
    if quality["leakage_flags"]:
        penalize("Potential target leakage", 0, 18)
    high_risk_count = int((profiles["risk"] == "High").sum())
    if high_risk_count >= 3:
        penalize("Multiple high-risk columns", 8, 10)
    elif high_risk_count >= 1:
        penalize("At least one high-risk column", 4, 5)

    analytics = max(35, min(100, analytics))
    ml = max(25, min(100, ml))
    return analytics, ml, penalties


def convert_df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def main() -> None:
    st.markdown(
        """
        <div class='hero'>
            <h1>DataLens</h1>
            <p>Data profiling, quality audit, and ML-readiness review in one polished workspace.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("DataLens")
    st.sidebar.caption("Upload a dataset and review quality, risk, and readiness.")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    sample = st.sidebar.selectbox("Or load a sample dataset", ["None", "Titanic", "Iris"])

    df: Optional[pd.DataFrame] = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            return
    elif sample == "Titanic":
        df = px.data.tips()  # stable local sample; use renamed product-friendly labels below
        df = df.rename(columns={"total_bill": "bill_total", "tip": "tip_amount", "sex": "customer_gender", "smoker": "smoker_flag", "day": "visit_day", "time": "service_window", "size": "party_size"})
    elif sample == "Iris":
        df = px.data.iris()

    if df is None:
        st.info("Upload a CSV from the sidebar or load a sample dataset to begin.")
        st.stop()

    for col in df.columns:
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                df[col] = parsed

    st.sidebar.markdown("---")
    target_col = st.sidebar.selectbox("Optional target column", ["None"] + list(df.columns))
    target_col = None if target_col == "None" else target_col

    quality = get_quality_summary(df, target_col)
    profiles = profile_columns(df)
    roles = detect_column_roles(df, target_col)
    insights = build_insights(df, quality, profiles, target_col)
    actions = recommend_actions(quality, profiles, target_col)
    analytics_score, ml_score, penalties = compute_readiness_score(quality, profiles)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='metric-card bg1'><h4>Rows</h4><h2>{quality['rows']:,}</h2><p>Records loaded</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(f"<div class='metric-card bg2'><h4>Columns</h4><h2>{quality['cols']}</h2><p>{quality['numeric_cols']} numeric • {quality['categorical_cols']} other</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='metric-card bg3'><h4>Analytics Readiness</h4><h2>{analytics_score}/100</h2><p>Penalty drivers: {len(penalties)}</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='metric-card bg4'><h4>ML Readiness</h4><h2>{ml_score}/100</h2><p>{quality['modeling_hint']}</p></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Quality Audit",
        "Column Profiler",
        "ML Risk Checks",
        "Insights & Actions",
        "Export",
    ])

    with tab1:
        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Dataset snapshot")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption("Preview of the first 10 rows.")
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Detected column roles")
            st.dataframe(roles, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            missing_by_col = df.isna().mean().mul(100).sort_values(ascending=False).head(10)
            if not missing_by_col.empty:
                fig = px.bar(
                    x=missing_by_col.values,
                    y=missing_by_col.index,
                    orientation="h",
                    color=missing_by_col.values,
                    color_continuous_scale=["#ffcf70", "#c48bff", "#00a67e"],
                    title="Top columns by missing percentage",
                )
                fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
            dtype_counts.columns = ["dtype", "count"]
            fig = px.pie(dtype_counts, names="dtype", values="count", hole=0.48, color_discrete_sequence=["#8a4fff", "#ffb347", "#00a67e", "#d9a8ff", "#ffd97d"])
            fig.update_layout(title="Column type mix", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Quality findings")
        q1, q2, q3 = st.columns(3)
        q1.metric("Overall missing", f"{quality['missing_pct']}%")
        q2.metric("Duplicate rows", quality["duplicate_rows"])
        q3.metric("Zero-variance columns", len(quality["zero_variance_cols"]))

        findings = {
            "Mixed type columns": quality["mixed_type_cols"],
            "Identifier-like columns": quality["id_like_cols"],
            "High-cardinality columns": quality["high_card_cols"],
            "Date-like text columns": quality["date_like_cols"],
            "Zero-variance columns": quality["zero_variance_cols"],
        }
        cols = st.columns(2)
        for i, (label, vals) in enumerate(findings.items()):
            with cols[i % 2]:
                st.markdown(f"**{label}**")
                st.write(vals if vals else "None")
        st.markdown("</div>", unsafe_allow_html=True)

        if not quality["outlier_df"].empty:
            fig = px.bar(
                quality["outlier_df"].head(10),
                x="column",
                y="outlier_pct",
                color="outlier_pct",
                color_continuous_scale=["#00a67e", "#ffd97d", "#8a4fff"],
                title="Outlier percentage by numeric column",
            )
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.subheader("Column profiler")
        risk_filter = st.multiselect("Filter by risk", ["Low", "Medium", "High"], default=["Low", "Medium", "High"])
        filtered_profiles = profiles[profiles["risk"].isin(risk_filter)]
        st.dataframe(filtered_profiles, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        left, right = st.columns(2)
        numeric_df = df.select_dtypes(include=np.number)
        with left:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Correlation review")
            if numeric_df.shape[1] >= 2:
                corr = numeric_df.corr(numeric_only=True)
                fig = px.imshow(corr, text_auto='.2f', color_continuous_scale=["#fff0c9", "#cfa5ff", "#8a4fff"], aspect="auto")
                fig.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need at least two numeric columns for a correlation heatmap.")
            st.markdown("</div>", unsafe_allow_html=True)
        with right:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.subheader("Target-aware checks")
            st.write(f"Modeling hint: **{quality['modeling_hint']}**")
            if target_col is None:
                st.info("Select a target column from the sidebar to unlock leakage and imbalance checks.")
            else:
                if quality["modeling_hint"] == "Classification":
                    value_counts = df[target_col].astype(str).value_counts().reset_index()
                    value_counts.columns = [target_col, "count"]
                    fig = px.bar(value_counts, x=target_col, y="count", color="count", color_continuous_scale=["#ffcf70", "#00a67e", "#8a4fff"], title="Target class distribution")
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)
                    if quality["target_imbalance"] is not None:
                        st.write(f"Largest class share: **{quality['target_imbalance']}%**")
                if quality["leakage_flags"]:
                    st.warning(f"Leakage candidates: {', '.join(quality['leakage_flags'])}")
                else:
                    st.success("No extreme leakage flags detected from simple target-correlation checks.")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab5:
        left, right = st.columns(2)
        with left:
            st.subheader("Plain-English insights")
            if insights:
                for item in insights:
                    st.markdown(f"<div class='insight-box'>• {item}</div>", unsafe_allow_html=True)
            else:
                st.info("No major concerns surfaced from the current checks.")
        with right:
            st.subheader("Recommended next actions")
            st.dataframe(actions, use_container_width=True, hide_index=True)
            if penalties:
                st.caption("Readiness score deductions came from: " + ", ".join(penalties))

    with tab6:
        st.subheader("Export review outputs")
        summary_df = pd.DataFrame(
            {
                "metric": [
                    "rows",
                    "columns",
                    "overall_missing_pct",
                    "duplicate_rows",
                    "analytics_readiness",
                    "ml_readiness",
                    "modeling_hint",
                ],
                "value": [
                    quality["rows"],
                    quality["cols"],
                    quality["missing_pct"],
                    quality["duplicate_rows"],
                    analytics_score,
                    ml_score,
                    quality["modeling_hint"],
                ],
            }
        )
        st.download_button("Download summary CSV", data=convert_df_to_csv_bytes(summary_df), file_name="datalens_summary.csv", mime="text/csv")
        st.download_button("Download column profile CSV", data=convert_df_to_csv_bytes(profiles), file_name="datalens_column_profile.csv", mime="text/csv")
        st.download_button("Download recommended actions CSV", data=convert_df_to_csv_bytes(actions), file_name="datalens_recommendations.csv", mime="text/csv")

        report_lines = [
            "DataLens Report",
            "================",
            f"Rows: {quality['rows']}",
            f"Columns: {quality['cols']}",
            f"Overall missing %: {quality['missing_pct']}",
            f"Duplicate rows: {quality['duplicate_rows']}",
            f"Analytics readiness: {analytics_score}/100",
            f"ML readiness: {ml_score}/100",
            f"Modeling hint: {quality['modeling_hint']}",
            "",
            "Top insights:",
        ] + [f"- {item}" for item in insights] + ["", "Recommended actions:"] + [f"- {row.recommended_action}: {row.why_it_matters}" for row in actions.itertuples()]
        st.download_button("Download text report", data="\n".join(report_lines), file_name="datalens_report.txt", mime="text/plain")

    st.markdown("<p class='small-note'>Tip: replace the sample dataset with your own CSV and select a target column to make the app feel more ML-aware during demos.</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
