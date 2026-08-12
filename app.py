import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Toronto Ferry Analytics",
    page_icon="⛴️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "ferry_processed_ml.csv"
KPI_FILE = BASE_DIR / "final_kpis.csv"
CLUSTER_FILE = BASE_DIR / "cluster_profile.csv"
K_SELECTION_FILE = BASE_DIR / "k_selection_scores.csv"

MODEL_DIR = BASE_DIR / "models"
CHART_DIR = BASE_DIR / "charts"

KMEANS_FILE = MODEL_DIR / "kmeans_model.pkl"
ISOLATION_FILE = MODEL_DIR / "isolation_forest.pkl"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
CONFIG_FILE = MODEL_DIR / "model_config.json"


# ============================================================
# CIVIC / GOVERNMENT-STYLE THEME
# ============================================================
# Palette modeled after municipal information portals:
# deep navy header, restrained blues/greys, a single gold accent.
# No gradients, no glassmorphism, no neon.

NAVY = "#14315C"
NAVY_DARK = "#0D2440"
GOLD = "#C98A1F"
BODY_BG = "#F5F6F8"
CARD_BG = "#FFFFFF"
BORDER = "#D7DCE3"
TEXT_DARK = "#1B2733"
TEXT_MUTED = "#5B6B7C"
STATUS_HIGH = "#9E4B3A"
STATUS_LOW = "#6B7686"
STATUS_NORMAL = "#3A6B8A"

PLOTLY_TEMPLATE_COLORS = [NAVY, GOLD, "#3A6B8A", "#6B7686", "#9E4B3A", "#5F8B6E"]

st.markdown(
    f"""
    <style>

    .stApp {{
        background-color: {BODY_BG};
    }}

    .main {{
        padding-top: 0rem;
    }}

    .block-container {{
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}

    /* ---------- Government header bar ---------- */
    .gov-header {{
        background-color: {NAVY};
        margin: -1rem -4rem 0 -4rem;
        padding: 18px 4rem 18px 4rem;
        border-bottom: 4px solid {GOLD};
    }}

    .gov-header-inner {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }}

    .gov-title {{
        color: #FFFFFF;
        font-size: 22px;
        font-weight: 700;
        letter-spacing: 0.5px;
        margin: 0;
    }}

    .gov-subtitle {{
        color: #C9D3DE;
        font-size: 13px;
        margin-top: 2px;
        font-weight: 400;
    }}

    .gov-nav {{
        margin-top: 10px;
        display: flex;
        gap: 26px;
        flex-wrap: wrap;
    }}

    .gov-nav a {{
        color: #E7ECF2;
        font-size: 14px;
        font-weight: 600;
        text-decoration: none;
        padding-bottom: 4px;
    }}

    .gov-nav a:hover {{
        color: {GOLD};
        border-bottom: 2px solid {GOLD};
    }}

    /* ---------- Breadcrumb ---------- */
    .breadcrumb {{
        background-color: #EBEEF2;
        margin: 0 -4rem 1.5rem -4rem;
        padding: 10px 4rem;
        font-size: 13px;
        color: {TEXT_MUTED};
        border-bottom: 1px solid {BORDER};
    }}

    .breadcrumb a {{
        color: #2C5C86;
        text-decoration: none;
    }}

    .breadcrumb a:hover {{
        text-decoration: underline;
    }}

    /* ---------- Page title block ---------- */
    .page-title {{
        font-size: 32px;
        font-weight: 800;
        color: {TEXT_DARK};
        margin-bottom: 4px;
    }}

    .page-lede {{
        font-size: 15px;
        color: {TEXT_MUTED};
        max-width: 780px;
        line-height: 1.55;
        margin-bottom: 1.4rem;
    }}

    /* ---------- Section titles ---------- */
    .section-title {{
        color: {TEXT_DARK};
        font-size: 21px;
        font-weight: 700;
        margin-top: 2.2rem;
        margin-bottom: 0.6rem;
        padding-bottom: 6px;
        border-bottom: 2px solid {NAVY};
    }}

    .section-subnote {{
        color: {TEXT_MUTED};
        font-size: 13.5px;
        margin-bottom: 0.8rem;
    }}

    /* ---------- In This Analysis box ---------- */
    .side-box {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 4px;
        overflow: hidden;
        margin-bottom: 18px;
    }}

    .side-box-header {{
        background-color: {NAVY};
        color: #FFFFFF;
        font-size: 13.5px;
        font-weight: 700;
        letter-spacing: 0.3px;
        padding: 10px 14px;
        text-transform: uppercase;
    }}

    .side-box-body a {{
        display: block;
        padding: 9px 14px;
        font-size: 14px;
        color: #2C5C86;
        text-decoration: none;
        border-bottom: 1px solid #EEF1F4;
    }}

    .side-box-body a:hover {{
        background-color: #F0F4F8;
        color: {NAVY_DARK};
    }}

    .side-box-body a:last-child {{
        border-bottom: none;
    }}

    /* ---------- Info / methodology panels ---------- */
    .info-panel {{
        background-color: #EEF3F8;
        border-left: 4px solid {NAVY};
        padding: 14px 16px;
        border-radius: 3px;
        margin: 10px 0;
        font-size: 14px;
        color: {TEXT_DARK};
        line-height: 1.5;
    }}

    .recommend-panel {{
        background-color: #FFFFFF;
        border: 1px solid {BORDER};
        border-left: 4px solid {GOLD};
        padding: 14px 16px;
        border-radius: 3px;
        margin: 10px 0;
    }}

    .recommend-title {{
        font-weight: 700;
        font-size: 14.5px;
        color: {TEXT_DARK};
        margin-bottom: 4px;
    }}

    .recommend-body {{
        font-size: 13.5px;
        color: {TEXT_MUTED};
        line-height: 1.5;
    }}

    /* ---------- Restrained KPI blocks ---------- */
    .kpi-block {{
        background-color: {CARD_BG};
        border: 1px solid {BORDER};
        border-top: 3px solid {NAVY};
        border-radius: 3px;
        padding: 14px 16px;
        height: 100%;
    }}

    .kpi-label {{
        font-size: 12.5px;
        color: {TEXT_MUTED};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }}

    .kpi-value {{
        font-size: 24px;
        color: {TEXT_DARK};
        font-weight: 800;
        margin-top: 4px;
    }}

    /* ---------- Threshold-based alert states ---------- */
    .kpi-block-alert {{
        border-top: 3px solid {STATUS_HIGH};
    }}

    .kpi-value-alert {{
        color: {STATUS_HIGH};
    }}

    .kpi-flag {{
        display: inline-block;
        font-size: 11px;
        font-weight: 700;
        color: {STATUS_HIGH};
        margin-top: 3px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }}

    .alert-banner {{
        background-color: #FBEAE5;
        border-left: 4px solid {STATUS_HIGH};
        padding: 12px 16px;
        border-radius: 3px;
        margin: 6px 0 18px 0;
        font-size: 13.5px;
        font-weight: 600;
        color: {TEXT_DARK};
        line-height: 1.5;
    }}

    .alert-banner-ok {{
        background-color: #EAF2ED;
        border-left: 4px solid #4E7A5D;
    }}

    /* ---------- Data table style ---------- */
    .gov-table-note {{
        font-size: 12.5px;
        color: {TEXT_MUTED};
        margin-bottom: 6px;
    }}

    /* ---------- Footer ---------- */
    .gov-footer {{
        margin: 3rem -4rem -2rem -4rem;
        padding: 20px 4rem;
        background-color: {NAVY_DARK};
        color: #C9D3DE;
        font-size: 12.5px;
        text-align: left;
    }}

    /* Tame default Streamlit chrome (keep header visible so the
       sidebar collapse/expand arrow stays usable) */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{background-color: transparent;}}

    div[data-testid="stMetricValue"] {{
        color: {TEXT_DARK};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        st.error(f"Dataset not found: {DATA_FILE}")
        st.stop()

    data = pd.read_csv(DATA_FILE)

    if "Timestamp" in data.columns:
        data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")

    return data


@st.cache_data
def load_optional_csv(path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


@st.cache_resource
def load_models():
    models = {}
    try:
        if KMEANS_FILE.exists():
            models["kmeans"] = joblib.load(KMEANS_FILE)
        if ISOLATION_FILE.exists():
            models["isolation_forest"] = joblib.load(ISOLATION_FILE)
        if SCALER_FILE.exists():
            models["scaler"] = joblib.load(SCALER_FILE)
    except Exception as e:
        st.warning(f"Some ML models could not be loaded: {e}")
    return models


@st.cache_data
def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def format_number(value):
    if pd.isna(value):
        return "N/A"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def find_column(data, possible_names):
    for name in possible_names:
        if name in data.columns:
            return name
    return None


def section_anchor(anchor_id):
    """Invisible anchor target so the 'In This Analysis' box can jump here."""
    st.markdown(f'<div id="{anchor_id}"></div>', unsafe_allow_html=True)


def section_title(text, anchor_id=None, subnote=None):
    if anchor_id:
        section_anchor(anchor_id)
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)
    if subnote:
        st.markdown(f'<div class="section-subnote">{subnote}</div>', unsafe_allow_html=True)


def style_fig(fig):
    """Apply a restrained civic-portal look to a Plotly figure."""
    fig.update_layout(
        colorway=PLOTLY_TEMPLATE_COLORS,
        font=dict(family="Arial, Helvetica, sans-serif", size=13, color=TEXT_DARK),
        title_font=dict(size=15, color=TEXT_DARK),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEF1F4", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEF1F4", zeroline=False)
    return fig


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()

kpi_data = load_optional_csv(KPI_FILE)
cluster_profile = load_optional_csv(CLUSTER_FILE)
k_selection = load_optional_csv(K_SELECTION_FILE)

models = load_models()
model_config = load_config()


# ============================================================
# DERIVED COLUMNS
# ============================================================

if "Total_Activity_Load" not in df.columns:
    if "Sales Count" in df.columns and "Redemption Count" in df.columns:
        df["Total_Activity_Load"] = df["Sales Count"] + df["Redemption Count"]

if "Operational_Status" not in df.columns:
    if "Total_Activity_Load" in df.columns:
        high_threshold = df["Total_Activity_Load"].quantile(0.95)
        low_threshold = df["Total_Activity_Load"].quantile(0.05)

        df["Operational_Status"] = np.select(
            [
                df["Total_Activity_Load"] >= high_threshold,
                df["Total_Activity_Load"] <= low_threshold
            ],
            ["High Pressure", "Low Activity"],
            default="Normal"
        )

# Redemption Pressure Ratio: how much redemption activity is occurring
# relative to sales in the same interval.
if "Redemption_Pressure_Ratio" not in df.columns:
    if "Sales Count" in df.columns and "Redemption Count" in df.columns:
        df["Redemption_Pressure_Ratio"] = (
            df["Redemption Count"] / (df["Sales Count"] + 1)
        )

# Operational Load Index (OLI): activity load normalized to a 0-100 scale
# against the full dataset's observed range, so it stays comparable
# regardless of how the person filters the view.
if "Operational_Load_Index" not in df.columns:
    if "Total_Activity_Load" in df.columns:
        _oli_min = df["Total_Activity_Load"].min()
        _oli_max = df["Total_Activity_Load"].max()
        _oli_range = (_oli_max - _oli_min) if (_oli_max - _oli_min) != 0 else 1
        df["Operational_Load_Index"] = (
            (df["Total_Activity_Load"] - _oli_min) / _oli_range
        ) * 100


# ============================================================
# HEADER (civic-portal style)
# ============================================================

st.markdown(
    f"""
    <div class="gov-header">
        <div class="gov-header-inner">
            <div>
                <div class="gov-title">⛴ TORONTO FERRY ANALYTICS</div>
                <div class="gov-subtitle">Capacity Utilization &amp; Operational Efficiency — Analytical Portal</div>
            </div>
        </div>
        <div class="gov-nav">
            <a href="#overview">Overview</a>
            <a href="#capacity">Capacity &amp; Utilization</a>
            <a href="#temporal">Time &amp; Seasonal Analysis</a>
            <a href="#efficiency">Operational Efficiency</a>
            <a href="#ml">Machine Learning</a>
            <a href="#recommendations">Recommendations</a>
            <a href="#methodology">Data &amp; Methodology</a>
        </div>
    </div>
    <div class="breadcrumb">
        <a href="#overview">Home</a> &nbsp;/&nbsp;
        <a href="#overview">Transportation</a> &nbsp;/&nbsp;
        <a href="#overview">Ferry Operations</a> &nbsp;/&nbsp;
        Capacity &amp; Efficiency
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR — FILTERS
# ============================================================

st.sidebar.markdown("### Filters")
st.sidebar.caption("Refine the dataset used across this analysis.")
st.sidebar.divider()

filtered_df = df.copy()

# Date filter
if "Timestamp" in filtered_df.columns:
    min_date = filtered_df["Timestamp"].min().date()
    max_date = filtered_df["Timestamp"].max().date()

    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df["Timestamp"].dt.date >= start_date) &
            (filtered_df["Timestamp"].dt.date <= end_date)
        ]

# Season filter
if "Season" in filtered_df.columns:
    seasons = sorted(filtered_df["Season"].dropna().unique().tolist())
    selected_seasons = st.sidebar.multiselect("Season", seasons, default=seasons)
    if selected_seasons:
        filtered_df = filtered_df[filtered_df["Season"].isin(selected_seasons)]

# Weekend filter
if "Is_Weekend" in filtered_df.columns:
    weekend_options = st.sidebar.multiselect(
        "Day Type", ["Weekday", "Weekend"], default=["Weekday", "Weekend"]
    )
    if len(weekend_options) == 1:
        if weekend_options[0] == "Weekend":
            filtered_df = filtered_df[filtered_df["Is_Weekend"] == 1]
        else:
            filtered_df = filtered_df[filtered_df["Is_Weekend"] == 0]

# Operational status filter
if "Operational_Status" in filtered_df.columns:
    statuses = sorted(filtered_df["Operational_Status"].dropna().unique().tolist())
    selected_status = st.sidebar.multiselect("Operational Status", statuses, default=statuses)
    if selected_status:
        filtered_df = filtered_df[filtered_df["Operational_Status"].isin(selected_status)]

st.sidebar.divider()
st.sidebar.caption(f"Records in view: {len(filtered_df):,}")


# ============================================================
# PAGE TITLE + "IN THIS ANALYSIS" LAYOUT
# ============================================================

main_col, nav_col = st.columns([3.2, 1], gap="large")

with main_col:
    section_anchor("overview")
    st.markdown('<div class="page-title">Ferry Capacity &amp; Operational Efficiency</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="page-lede">
        Data-driven analysis of ferry ticket activity, capacity utilization, operational
        pressure, idle periods and temporal demand patterns, 
        supported by machine-learning segmentation and anomaly detection across the ferry network.
        
        This analytical portal provides a data-driven view of Toronto ferry operations, 
        focusing on passenger activity, capacity utilization and operational efficiency. 
        
        Historical ticket activity is analysed across time, seasons and operating periods to identify demand patterns,
        high-pressure intervals and under-utilized periods. 
        
        Machine-learning techniques are additionally used to segment operational activity and detect unusual patterns, 
        supporting more informed capacity planning and resource allocation.
        </div>
        """,
        unsafe_allow_html=True
    )

with nav_col:
    st.markdown(
        """
        <div class="side-box">
            <div class="side-box-header">In This Analysis</div>
            <div class="side-box-body">
                <a href="#overview">Overview</a>
                <a href="#capacity">Capacity &amp; Utilization</a>
                <a href="#temporal">Time &amp; Seasonal Patterns</a>
                <a href="#efficiency">Operational Efficiency</a>
                <a href="#ml">Machine Learning</a>
                <a href="#recommendations">Recommendations</a>
                <a href="#methodology">Data &amp; Methodology</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# KEY FINDINGS — OFFICIAL KPIs
# ============================================================

section_title(
    "Key Findings",
    subnote="The five performance indicators defined in the project specification, calculated for the current filter selection."
)

total_records = len(filtered_df)

sales_col = find_column(filtered_df, ["Sales Count", "Sales_Count"])
redemption_col = find_column(filtered_df, ["Redemption Count", "Redemption_Count"])
load_col = find_column(filtered_df, ["Total_Activity_Load"])

total_sales = filtered_df[sales_col].sum() if sales_col else 0
total_redemptions = filtered_df[redemption_col].sum() if redemption_col else 0
average_load = filtered_df[load_col].mean() if load_col else 0
max_load = filtered_df[load_col].max() if load_col else 0

# Reference peak is the 95th percentile of load across the full (unfiltered)
# dataset — not the single highest interval — so one outlier spike doesn't
# distort the ratio for everyone else. Stays comparable across filters.
reference_peak_load = df[load_col].quantile(0.95) if load_col else 0

if load_col:
    high_pressure_count = filtered_df["Operational_Status"].eq("High Pressure").sum()
    low_activity_count = filtered_df["Operational_Status"].eq("Low Activity").sum()
else:
    high_pressure_count = 0
    low_activity_count = 0

anomaly_count = 0
if "Anomaly_Label" in filtered_df.columns:
    anomaly_count = filtered_df["Anomaly_Label"].eq("Anomaly").sum()
elif "Anomaly_Prediction" in filtered_df.columns:
    anomaly_count = (filtered_df["Anomaly_Prediction"] == -1).sum()

# ---- 1. Capacity Utilization Ratio ----
# Average activity load relative to a typical high-demand reference point
# (95th percentile system-wide), so a single extreme spike doesn't distort it.
capacity_utilization_ratio = (
    (average_load / reference_peak_load * 100) if reference_peak_load else 0
)
# Ratio can exceed 100% if the current filter's average load is itself
# higher than the system-wide 95th percentile — that's expected, not a bug.

# ---- 2. Congestion Pressure Index ----
# Share of intervals in view flagged High Pressure (top 5% of system-wide load).
congestion_pressure_index = (
    (high_pressure_count / total_records * 100) if total_records else 0
)

# ---- 3. Idle Capacity Percentage ----
# Share of intervals in view flagged Low Activity (bottom 5% of system-wide load).
idle_capacity_percentage = (
    (low_activity_count / total_records * 100) if total_records else 0
)

# ---- 4. Peak Strain Duration ----
# Longest consecutive run of High Pressure intervals in the current selection.
peak_strain_intervals = 0
peak_strain_label = "N/A"

if "Timestamp" in filtered_df.columns and "Operational_Status" in filtered_df.columns and total_records > 0:
    ordered = filtered_df.sort_values("Timestamp")
    is_high = (ordered["Operational_Status"] == "High Pressure").astype(int)

    if is_high.any():
        run_id = (is_high != is_high.shift()).cumsum()
        run_sums = is_high.groupby(run_id).sum()
        peak_strain_intervals = int(run_sums.max())

        time_diffs = ordered["Timestamp"].diff().dropna()
        if not time_diffs.empty and peak_strain_intervals > 0:
            median_interval = time_diffs.median()
            total_minutes = (median_interval * peak_strain_intervals).total_seconds() / 60
            if total_minutes >= 60:
                peak_strain_label = f"{total_minutes / 60:.1f} hrs"
            else:
                peak_strain_label = f"{total_minutes:.0f} min"
        else:
            peak_strain_label = f"{peak_strain_intervals} intervals"
    else:
        peak_strain_intervals = 0
        peak_strain_label = "0 min"

# ---- 5. Operational Variability Score ----
# Coefficient of variation of activity load — how stable vs. erratic demand is.
operational_variability_score = 0
if load_col and average_load:
    std_load = filtered_df[load_col].std()
    operational_variability_score = (std_load / average_load * 100) if average_load else 0

# ---- Threshold-based visual alerts ----
CONGESTION_ALERT_THRESHOLD = 15.0   # % of intervals under high pressure
IDLE_ALERT_THRESHOLD = 20.0         # % of intervals idle
PEAK_STRAIN_ALERT_INTERVALS = 20    # consecutive high-pressure intervals

congestion_breached = congestion_pressure_index >= CONGESTION_ALERT_THRESHOLD
idle_breached = idle_capacity_percentage >= IDLE_ALERT_THRESHOLD
strain_breached = peak_strain_intervals >= PEAK_STRAIN_ALERT_INTERVALS

alerts = []
if congestion_breached:
    alerts.append(
        f"Congestion Pressure Index is {congestion_pressure_index:.1f}%, at or above the "
        f"{CONGESTION_ALERT_THRESHOLD:.0f}% threshold — sustained high-pressure operation in the current selection."
    )
if idle_breached:
    alerts.append(
        f"Idle Capacity is {idle_capacity_percentage:.1f}%, at or above the "
        f"{IDLE_ALERT_THRESHOLD:.0f}% threshold — a significant share of intervals show under-utilization."
    )
if strain_breached:
    alerts.append(
        f"Peak Strain Duration reached {peak_strain_intervals} consecutive high-pressure intervals "
        f"({peak_strain_label}), at or above the {PEAK_STRAIN_ALERT_INTERVALS}-interval threshold."
    )

if alerts:
    for a in alerts:
        st.markdown(f'<div class="alert-banner">⚠ {a}</div>', unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="alert-banner alert-banner-ok">✓ No threshold breaches detected in the current filter selection.</div>',
        unsafe_allow_html=True
    )

# ---- Official KPI cards ----
kpi_cols = st.columns(5)
kpi_definitions = [
    ("Capacity Utilization Ratio", f"{capacity_utilization_ratio:.1f}%", False),
    ("Congestion Pressure Index", f"{congestion_pressure_index:.1f}%", congestion_breached),
    ("Idle Capacity Percentage", f"{idle_capacity_percentage:.1f}%", idle_breached),
    ("Peak Strain Duration", peak_strain_label, strain_breached),
    ("Operational Variability Score", f"{operational_variability_score:.1f}%", False),
]

for col, (label, value, is_alert) in zip(kpi_cols, kpi_definitions):
    block_class = "kpi-block kpi-block-alert" if is_alert else "kpi-block"
    value_class = "kpi-value kpi-value-alert" if is_alert else "kpi-value"
    flag_html = '<div class="kpi-flag">Above threshold</div>' if is_alert else ""
    with col:
        st.markdown(
            f"""
            <div class="{block_class}">
                <div class="kpi-label">{label}</div>
                <div class="{value_class}">{value}</div>
                {flag_html}
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown(
    '<div class="section-subnote" style="margin-top:1.1rem;">Volume summary for the current filter selection.</div>',
    unsafe_allow_html=True
)

vol_cols = st.columns(4)
vol_items = [
    ("Analysed Intervals", f"{total_records:,}"),
    ("Total Sales", format_number(total_sales)),
    ("Total Redemptions", format_number(total_redemptions)),
    ("Detected Anomalies", f"{anomaly_count:,}"),
]

for col, (label, value) in zip(vol_cols, vol_items):
    with col:
        st.markdown(
            f"""
            <div class="kpi-block">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# CAPACITY & UTILIZATION
# ============================================================

section_title(
    "Capacity &amp; Utilization",
    anchor_id="capacity",
    subnote="Operational activity load over time, engineered pressure indicators, and the relationship between ticket sales and redemptions."
)

GRANULARITY_OPTIONS = {"15-Minute": "15min", "Hourly": "h", "Daily": "D"}

if "Timestamp" in filtered_df.columns and load_col:
    granularity_label = st.radio(
        "Time Granularity",
        list(GRANULARITY_OPTIONS.keys()),
        index=2,
        horizontal=True,
        key="capacity_granularity"
    )
    granularity_freq = GRANULARITY_OPTIONS[granularity_label]

    resampled = (
        filtered_df.set_index("Timestamp").resample(granularity_freq)[load_col].sum().reset_index()
    )

    fig = px.line(
        resampled, x="Timestamp", y=load_col,
        title=f"Operational Activity Load ({granularity_label})"
    )
    fig.update_layout(xaxis_title="Time", yaxis_title="Total Activity Load", hovermode="x unified")
    st.plotly_chart(style_fig(fig), use_container_width=True)

# ---- Operational Load Index (OLI) and Redemption Pressure Ratio ----
oli_col = find_column(filtered_df, ["Operational_Load_Index"])
rpr_col = find_column(filtered_df, ["Redemption_Pressure_Ratio"])

if oli_col or rpr_col:
    st.markdown(
        '<div class="section-subnote">Engineered capacity indicators: normalized operational pressure and redemption-to-sales ratio.</div>',
        unsafe_allow_html=True
    )

    oli_c1, oli_c2 = st.columns(2)

    if oli_col:
        with oli_c1:
            avg_oli = filtered_df[oli_col].mean()
            st.markdown(
                f"""
                <div class="kpi-block">
                    <div class="kpi-label">Average Operational Load Index</div>
                    <div class="kpi-value">{avg_oli:.1f} / 100</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    if rpr_col:
        with oli_c2:
            avg_rpr = filtered_df[rpr_col].mean()
            st.markdown(
                f"""
                <div class="kpi-block">
                    <div class="kpi-label">Average Redemption Pressure Ratio</div>
                    <div class="kpi-value">{avg_rpr:.2f}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    if oli_col and "Timestamp" in filtered_df.columns:
        oli_daily = (
            filtered_df.set_index("Timestamp").resample("D")[oli_col].mean().reset_index()
        )
        fig = px.line(
            oli_daily, x="Timestamp", y=oli_col,
            title="Operational Load Index Over Time (0 = lowest observed load, 100 = peak)"
        )
        fig.update_layout(xaxis_title="Date", yaxis_title="Operational Load Index")
        fig.add_hline(y=80, line_dash="dot", line_color=STATUS_HIGH, annotation_text="High-pressure zone")
        fig.add_hline(y=20, line_dash="dot", line_color=STATUS_LOW, annotation_text="Idle zone")
        st.plotly_chart(style_fig(fig), use_container_width=True)

if sales_col and redemption_col:
    sample_size = min(len(filtered_df), 15000)
    scatter_data = filtered_df.sample(sample_size, random_state=42)

    fig = px.scatter(
        scatter_data, x=sales_col, y=redemption_col, opacity=0.5,
        title="Sales vs. Redemption Relationship"
    )
    fig.update_layout(xaxis_title="Sales Count", yaxis_title="Redemption Count")
    st.plotly_chart(style_fig(fig), use_container_width=True)


# ============================================================
# TIME & SEASONAL ANALYSIS
# ============================================================

section_title(
    "Time &amp; Seasonal Analysis",
    anchor_id="temporal",
    subnote="Demand patterns by hour, month, year, day of week and season."
)

tab1, tab2, tab3 = st.tabs(["Hourly", "Monthly", "Yearly"])

with tab1:
    if "Hour" in filtered_df.columns:
        hourly = (
            filtered_df.groupby("Hour").agg(
                Average_Sales=(sales_col, "mean") if sales_col else (load_col, "mean"),
                Average_Redemptions=(redemption_col, "mean") if redemption_col else (load_col, "mean")
            ).reset_index()
        )
    elif "Timestamp" in filtered_df.columns:
        temp = filtered_df.copy()
        temp["Hour"] = temp["Timestamp"].dt.hour
        hourly = (
            temp.groupby("Hour").agg(
                Average_Sales=(sales_col, "mean") if sales_col else (load_col, "mean"),
                Average_Redemptions=(redemption_col, "mean") if redemption_col else (load_col, "mean")
            ).reset_index()
        )
    else:
        hourly = pd.DataFrame()

    if not hourly.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly["Hour"], y=hourly["Average_Sales"], mode="lines+markers", name="Sales"))
        fig.add_trace(go.Scatter(x=hourly["Hour"], y=hourly["Average_Redemptions"], mode="lines+markers", name="Redemptions"))
        fig.update_layout(title="Average Ticket Activity by Hour", xaxis_title="Hour of Day", yaxis_title="Average Tickets per Interval")
        st.plotly_chart(style_fig(fig), use_container_width=True)

with tab2:
    if "Month" in filtered_df.columns:
        monthly = (
            filtered_df.groupby("Month").agg(
                Average_Load=(load_col, "mean") if load_col else (sales_col, "mean")
            ).reset_index()
        )
    elif "Timestamp" in filtered_df.columns:
        temp = filtered_df.copy()
        temp["Month"] = temp["Timestamp"].dt.month_name()
        month_order = ["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"]
        monthly = (
            temp.groupby("Month").agg(
                Average_Load=(load_col, "mean") if load_col else (sales_col, "mean")
            ).reindex(month_order).reset_index()
        )
    else:
        monthly = pd.DataFrame()

    if not monthly.empty:
        fig = px.line(monthly, x="Month", y="Average_Load", markers=True, title="Monthly Ferry Activity Pattern")
        st.plotly_chart(style_fig(fig), use_container_width=True)

with tab3:
    if "Timestamp" in filtered_df.columns:
        temp = filtered_df.copy()
        temp["Year"] = temp["Timestamp"].dt.year
        yearly = (
            temp.groupby("Year").agg(
                Total_Sales=(sales_col, "sum") if sales_col else (load_col, "sum"),
                Total_Redemptions=(redemption_col, "sum") if redemption_col else (load_col, "sum")
            ).reset_index()
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["Total_Sales"], mode="lines+markers", name="Sales"))
        fig.add_trace(go.Scatter(x=yearly["Year"], y=yearly["Total_Redemptions"], mode="lines+markers", name="Redemptions"))
        fig.update_layout(title="Yearly Ferry Ticket Activity", xaxis_title="Year", yaxis_title="Total Tickets")
        st.plotly_chart(style_fig(fig), use_container_width=True)

st.markdown('<div class="section-subnote" style="margin-top:1.2rem;">Average operational load by day of week.</div>', unsafe_allow_html=True)

if "Day_Name" in filtered_df.columns:
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = filtered_df.groupby("Day_Name")[load_col].mean().reindex(weekday_order).reset_index()
else:
    temp = filtered_df.copy()
    temp["Day_Name"] = temp["Timestamp"].dt.day_name()
    weekday = temp.groupby("Day_Name")[load_col].mean().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).reset_index()

fig = px.bar(weekday, x="Day_Name", y=load_col, title="Average Operational Load by Day")
fig.update_layout(xaxis_title="Day", yaxis_title="Average Activity Load")
st.plotly_chart(style_fig(fig), use_container_width=True)

if "Season" in filtered_df.columns:
    season = (
        filtered_df.groupby("Season").agg(
            Average_Load=(load_col, "mean"),
            Maximum_Load=(load_col, "max"),
            Average_Sales=(sales_col, "mean") if sales_col else (load_col, "mean")
        ).reset_index()
    )

    fig = px.bar(season, x="Season", y="Average_Load", title="Average Activity Load by Season", text_auto=".2f")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    st.markdown('<div class="gov-table-note">Table: seasonal summary statistics</div>', unsafe_allow_html=True)
    st.dataframe(season, use_container_width=True, hide_index=True)


# ============================================================
# OPERATIONAL EFFICIENCY
# ============================================================

section_title(
    "Operational Efficiency",
    anchor_id="efficiency",
    subnote="Distribution of high-pressure, normal and low-activity intervals, sustained idle periods, and load intensity by day and hour."
)

if "Operational_Status" in filtered_df.columns:
    status_counts = filtered_df["Operational_Status"].value_counts().reset_index()
    status_counts.columns = ["Operational_Status", "Count"]

    fig = px.pie(
        status_counts, names="Operational_Status", values="Count", hole=0.45,
        title="Operational Status Distribution",
        color="Operational_Status",
        color_discrete_map={
            "High Pressure": STATUS_HIGH,
            "Normal": STATUS_NORMAL,
            "Low Activity": STATUS_LOW,
        }
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="kpi-block"><div class="kpi-label">High Pressure Intervals</div><div class="kpi-value">{high_pressure_count:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="kpi-block"><div class="kpi-label">Low Activity Intervals</div><div class="kpi-value">{low_activity_count:,}</div></div>', unsafe_allow_html=True)
    with c3:
        pressure_percentage = (high_pressure_count / len(filtered_df) * 100) if len(filtered_df) > 0 else 0
        st.markdown(f'<div class="kpi-block"><div class="kpi-label">High Pressure %</div><div class="kpi-value">{pressure_percentage:.2f}%</div></div>', unsafe_allow_html=True)

# ---- Idle Capacity Indicator: sustained low-activity runs ----
# A single low-activity interval is noise; a run of several consecutive
# ones is a genuine idle-capacity signal worth flagging operationally.
SUSTAINED_IDLE_MIN_RUN = 3

sustained_idle_intervals = 0
sustained_idle_label = "N/A"
sustained_idle_episode_count = 0

if "Timestamp" in filtered_df.columns and "Operational_Status" in filtered_df.columns and total_records > 0:
    ordered_idle = filtered_df.sort_values("Timestamp")
    is_idle = (ordered_idle["Operational_Status"] == "Low Activity").astype(int)

    if is_idle.any():
        idle_run_id = (is_idle != is_idle.shift()).cumsum()
        idle_run_sums = is_idle.groupby(idle_run_id).sum()
        idle_run_sums = idle_run_sums[idle_run_sums > 0]

        sustained_idle_intervals = int(idle_run_sums.max()) if not idle_run_sums.empty else 0
        sustained_idle_episode_count = int((idle_run_sums >= SUSTAINED_IDLE_MIN_RUN).sum())

        idle_time_diffs = ordered_idle["Timestamp"].diff().dropna()
        if not idle_time_diffs.empty and sustained_idle_intervals > 0:
            median_interval_idle = idle_time_diffs.median()
            idle_minutes = (median_interval_idle * sustained_idle_intervals).total_seconds() / 60
            if idle_minutes >= 60:
                sustained_idle_label = f"{idle_minutes / 60:.1f} hrs"
            else:
                sustained_idle_label = f"{idle_minutes:.0f} min"
        else:
            sustained_idle_label = f"{sustained_idle_intervals} intervals"
    else:
        sustained_idle_label = "0 min"

st.markdown(
    f'<div class="section-subnote" style="margin-top:1.4rem;">Idle Capacity Indicator — low activity sustained over {SUSTAINED_IDLE_MIN_RUN}+ consecutive intervals.</div>',
    unsafe_allow_html=True
)

idle_c1, idle_c2 = st.columns(2)
with idle_c1:
    st.markdown(
        f'<div class="kpi-block"><div class="kpi-label">Longest Sustained Idle Period</div><div class="kpi-value">{sustained_idle_label}</div></div>',
        unsafe_allow_html=True
    )
with idle_c2:
    st.markdown(
        f'<div class="kpi-block"><div class="kpi-label">Sustained Idle Episodes (≥{SUSTAINED_IDLE_MIN_RUN} intervals)</div><div class="kpi-value">{sustained_idle_episode_count:,}</div></div>',
        unsafe_allow_html=True
    )

# ---- Heatmaps: overall load, congestion intensity, idle intensity ----
st.markdown(
    '<div class="section-subnote" style="margin-top:1.4rem;">Operational intensity by day and hour.</div>',
    unsafe_allow_html=True
)

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

heat_tab1, heat_tab2, heat_tab3 = st.tabs(["Overall Load", "Congestion Intensity", "Idle Intensity"])

if "Timestamp" in filtered_df.columns:
    heatmap_base = filtered_df.copy()
    heatmap_base["Day"] = heatmap_base["Timestamp"].dt.day_name()
    heatmap_base["Hour"] = heatmap_base["Timestamp"].dt.hour

    with heat_tab1:
        heatmap = heatmap_base.pivot_table(
            index="Day", columns="Hour", values=load_col, aggfunc="mean"
        ).reindex(day_order)

        fig = px.imshow(
            heatmap, aspect="auto", title="Average Operational Activity by Day and Hour",
            labels={"x": "Hour of Day", "y": "Day of Week", "color": "Activity Load"},
            color_continuous_scale=[[0, "#F5F6F8"], [0.5, "#3A6B8A"], [1, NAVY_DARK]]
        )
        st.plotly_chart(style_fig(fig), use_container_width=True)

    with heat_tab2:
        if "Operational_Status" in heatmap_base.columns:
            heatmap_base["Is_High"] = (heatmap_base["Operational_Status"] == "High Pressure").astype(int)
            congestion_heatmap = (
                heatmap_base.pivot_table(index="Day", columns="Hour", values="Is_High", aggfunc="mean")
                .reindex(day_order) * 100
            )

            fig = px.imshow(
                congestion_heatmap, aspect="auto",
                title="Congestion Intensity — % of Intervals Flagged High Pressure, by Day and Hour",
                labels={"x": "Hour of Day", "y": "Day of Week", "color": "% High Pressure"},
                color_continuous_scale=[[0, "#F5F6F8"], [0.5, GOLD], [1, STATUS_HIGH]]
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with heat_tab3:
        if "Operational_Status" in heatmap_base.columns:
            heatmap_base["Is_Low"] = (heatmap_base["Operational_Status"] == "Low Activity").astype(int)
            idle_heatmap = (
                heatmap_base.pivot_table(index="Day", columns="Hour", values="Is_Low", aggfunc="mean")
                .reindex(day_order) * 100
            )

            fig = px.imshow(
                idle_heatmap, aspect="auto",
                title="Idle Intensity — % of Intervals Flagged Low Activity, by Day and Hour",
                labels={"x": "Hour of Day", "y": "Day of Week", "color": "% Low Activity"},
                color_continuous_scale=[[0, "#F5F6F8"], [0.5, "#8FA3B3"], [1, STATUS_LOW]]
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)

# ---- Comparative Efficiency Analysis: high-cost, low-utilization windows ----
st.markdown(
    """
    <div class="info-panel" style="margin-top:1.4rem;">
    <strong>High-cost, low-utilization windows.</strong> Ferry service runs on a fixed
    schedule, so every operating slot is staffed and resourced at comparable cost
    regardless of demand. The windows below are recurring day/hour slots that are
    disproportionately idle relative to how often they run — strong candidates for
    schedule or staffing review.
    </div>
    """,
    unsafe_allow_html=True
)

if "Timestamp" in filtered_df.columns and "Operational_Status" in filtered_df.columns and load_col:
    window_data = filtered_df.copy()
    window_data["Day"] = window_data["Timestamp"].dt.day_name()
    window_data["Hour"] = window_data["Timestamp"].dt.hour
    window_data["Is_Low"] = (window_data["Operational_Status"] == "Low Activity").astype(int)

    window_summary = (
        window_data.groupby(["Day", "Hour"]).agg(
            Interval_Count=("Is_Low", "count"),
            Low_Activity_Count=("Is_Low", "sum"),
            Average_Load=(load_col, "mean")
        ).reset_index()
    )
    window_summary["Idle_Rate"] = (
        window_summary["Low_Activity_Count"] / window_summary["Interval_Count"] * 100
    )

    MIN_WINDOW_SAMPLE = 5
    qualifying_windows = window_summary[window_summary["Interval_Count"] >= MIN_WINDOW_SAMPLE]
    top_windows = qualifying_windows.sort_values("Idle_Rate", ascending=False).head(10)

    if not top_windows.empty:
        display_windows = top_windows[["Day", "Hour", "Average_Load", "Idle_Rate", "Interval_Count"]].copy()
        display_windows["Average_Load"] = display_windows["Average_Load"].round(2)
        display_windows["Idle_Rate"] = display_windows["Idle_Rate"].round(1).astype(str) + "%"
        display_windows.columns = ["Day", "Hour", "Average Activity Load", "Idle Rate", "Sample Size (Intervals)"]

        st.markdown(
            f'<div class="gov-table-note">Table: top recurring day/hour windows by idle rate (minimum {MIN_WINDOW_SAMPLE} observed intervals in the current selection)</div>',
            unsafe_allow_html=True
        )
        st.dataframe(display_windows, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div class="info-panel">Not enough data in the current filter selection to identify recurring low-utilization windows.</div>',
            unsafe_allow_html=True
        )


# ============================================================
# MACHINE LEARNING
# ============================================================

section_title(
    "Machine Learning",
    anchor_id="ml",
    subnote="Operational activity segmentation (K-Means) and anomaly detection (Isolation Forest)."
)

ml_tab1, ml_tab2, ml_tab3 = st.tabs(["Segmentation", "Anomaly Detection", "Model Information"])

with ml_tab1:
    st.markdown(
        """
        <div class="info-panel">
        K-Means clustering was used to identify recurring operational activity profiles
        based on ticket activity and derived operational features.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dataset uses Operational_Cluster (human-readable label) and/or
    # Cluster_ID (numeric) rather than a plain "Cluster" column — detect
    # whichever is actually present instead of assuming one name.
    cluster_col = find_column(filtered_df, ["Operational_Cluster", "Cluster", "Cluster_ID"])

    if cluster_col:
        cluster_counts = filtered_df[cluster_col].value_counts().sort_index().reset_index()
        cluster_counts.columns = [cluster_col, "Count"]

        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(cluster_counts, names=cluster_col, values="Count", title="Cluster Distribution")
            st.plotly_chart(style_fig(fig), use_container_width=True)

        with col2:
            cluster_summary = (
                filtered_df.groupby(cluster_col).agg(
                    Average_Sales=(sales_col, "mean") if sales_col else (load_col, "mean"),
                    Average_Redemptions=(redemption_col, "mean") if redemption_col else (load_col, "mean"),
                    Average_Load=(load_col, "mean"),
                    Maximum_Load=(load_col, "max")
                ).reset_index()
            )
            st.markdown('<div class="gov-table-note">Table: cluster profile summary</div>', unsafe_allow_html=True)
            st.dataframe(cluster_summary, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div class="info-panel">No cluster assignment column was found in the current dataset.</div>',
            unsafe_allow_html=True
        )

with ml_tab2:
    st.markdown(
        """
        <div class="info-panel">
        Isolation Forest was applied to flag operational intervals whose activity
        pattern deviates materially from the norm.
        </div>
        """,
        unsafe_allow_html=True
    )

    if "Anomaly_Label" in filtered_df.columns:
        anomaly_counts = filtered_df["Anomaly_Label"].value_counts().reset_index()
        anomaly_counts.columns = ["Anomaly_Label", "Count"]

        fig = px.pie(anomaly_counts, names="Anomaly_Label", values="Count", title="Anomaly Detection Results")
        st.plotly_chart(style_fig(fig), use_container_width=True)

        anomalies = filtered_df[filtered_df["Anomaly_Label"] == "Anomaly"]

        st.markdown('<div class="gov-table-note">Table: detected anomalous intervals (first 100 shown)</div>', unsafe_allow_html=True)
        st.dataframe(anomalies.head(100), use_container_width=True, hide_index=True)

with ml_tab3:
    model_status = pd.DataFrame({
        "Model": ["K-Means", "Isolation Forest", "Standard Scaler"],
        "Status": [
            "Loaded" if "kmeans" in models else "Not Found",
            "Loaded" if "isolation_forest" in models else "Not Found",
            "Loaded" if "scaler" in models else "Not Found"
        ]
    })
    st.dataframe(model_status, use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="info-panel">
    <strong>Responsible use of ML results:</strong> Clusters and anomaly flags describe
    statistical patterns in historical activity data. They are decision-support inputs,
    not operational directives, and should be interpreted alongside domain knowledge
    and current service conditions.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# RECOMMENDATIONS
# ============================================================

section_title(
    "Operational Recommendations",
    anchor_id="recommendations",
    subnote="Findings translated into practical operational and planning actions."
)

recommendations = []

if len(filtered_df) > 0:
    if high_pressure_count > 0:
        recommendations.append((
            "Peak-period deployment",
            "Increase ferry deployment or staffing during high-pressure intervals to reduce congestion and wait times."
        ))
    if low_activity_count > 0:
        recommendations.append((
            "Low-utilization periods",
            "Review sustained low-activity intervals for schedule optimization or reduced deployment."
        ))

if "Season" in filtered_df.columns:
    season_avg = filtered_df.groupby("Season")[load_col].mean().sort_values(ascending=False)
    if len(season_avg) > 0:
        peak_season = season_avg.index[0]
        recommendations.append((
            "Seasonal planning",
            f"{peak_season} shows the highest average operational demand and should receive additional capacity planning."
        ))

if "Timestamp" in filtered_df.columns:
    temp = filtered_df.copy()
    temp["Hour"] = temp["Timestamp"].dt.hour
    hourly_load = temp.groupby("Hour")[load_col].mean().sort_values(ascending=False)
    if len(hourly_load) > 0:
        peak_hour = hourly_load.index[0]
        recommendations.append((
            "Peak-hour readiness",
            f"Hour {peak_hour}:00 represents the strongest average activity period and should be prioritized for operational readiness."
        ))

for title, body in recommendations:
    st.markdown(
        f"""
        <div class="recommend-panel">
            <div class="recommend-title">{title}</div>
            <div class="recommend-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# DATA & METHODOLOGY
# ============================================================

section_title(
    "Data &amp; Methodology",
    anchor_id="methodology",
    subnote="Dataset composition, feature engineering and modelling approach."
)

st.markdown(
    """
    <div class="info-panel">
    <strong>Dataset:</strong> Ferry ticket sales and redemption records, aggregated into
    operational intervals with derived features for time-of-day, day type, season and
    activity load.<br><br>
    <strong>Feature engineering:</strong> Total activity load (sales + redemptions),
    operational status thresholds (5th/95th percentile), calendar features (hour, day,
    month, season, weekend flag), Redemption Pressure Ratio, and a normalized
    Operational Load Index (OLI).<br><br>
    <strong>Capacity Utilization Ratio:</strong> average activity load in the current
    view divided by the system-wide 95th-percentile load, expressed as a percentage.
    The 95th percentile (rather than the single highest interval) is used as the
    reference so one extreme spike does not distort the ratio.<br><br>
    <strong>Modelling:</strong> K-Means clustering for activity segmentation; Isolation
    Forest for anomaly detection on standardized features.<br><br>
    <strong>Limitations:</strong> Findings reflect historical patterns in the dataset
    provided and do not account for external disruptions (weather, service changes,
    special events) unless explicitly captured in the source data.
    </div>
    """,
    unsafe_allow_html=True
)

with st.expander("View underlying data"):
    st.dataframe(filtered_df.head(500), use_container_width=True, hide_index=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="gov-footer">
        Toronto Ferry Analytics — Capacity &amp; Operational Efficiency Portal<br>
        
    </div>
    """,
    unsafe_allow_html=True
)