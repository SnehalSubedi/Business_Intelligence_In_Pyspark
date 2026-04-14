"""
Real-Time Streaming Dashboard
==============================
Auto-refreshes every 5 seconds to show live streaming data
as Spark processes batches.

Launched automatically from notebook cell 11.3
Stopped automatically from notebook cell 11.4
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import json
import time
import glob

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Real-Time Streaming Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE = os.path.dirname(os.path.abspath(__file__))
STREAM_OUTPUT = os.path.join(BASE, "output", "streaming", "results")
STATUS_FILE = os.path.join(BASE, "output", "streaming", "stream_status.json")

COLORS = {
    "primary": "#1565C0",
    "green": "#2E7D32",
    "orange": "#E65100",
    "purple": "#6A1B9A",
    "red": "#C62828",
    "teal": "#00838F",
}

PALETTE = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A",
           "#C62828", "#00838F", "#F9A825", "#AD1457",
           "#283593", "#00695C"]


# -------------------------------------------------------------------
# AUTO-REFRESH (every 5 seconds)
# -------------------------------------------------------------------
try:
    from streamlit_autorefresh import st_autorefresh
    refresh_count = st_autorefresh(
        interval=5000, limit=0, key="streaming_refresh")
except ImportError:
    refresh_count = 0
    st.sidebar.info("Install streamlit-autorefresh for auto-refresh")


# -------------------------------------------------------------------
# READ STREAMING STATUS
# -------------------------------------------------------------------
def read_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE) as f:
            return json.load(f)
    return {"status": "waiting", "batches": 0, "batch_size": 0}


def read_stream_data():
    """Read all Parquet files from streaming output."""
    if not os.path.exists(STREAM_OUTPUT):
        return pd.DataFrame()
    parquet_files = glob.glob(
        os.path.join(STREAM_OUTPUT, "*.parquet"))
    if not parquet_files:
        return pd.DataFrame()
    try:
        df = pd.read_parquet(STREAM_OUTPUT)
        return df
    except Exception:
        return pd.DataFrame()


# -------------------------------------------------------------------
# METRIC CARD
# -------------------------------------------------------------------
def metric_card(label, value, color="#1565C0", subtitle=None):
    sub_html = ""
    if subtitle:
        sub_html = (f'<p style="margin:2px 0 0;font-size:11px;'
                    f'color:#888;">{subtitle}</p>')
    st.markdown(
        f"""<div style="background:white;padding:16px 18px;
        border-radius:10px;border-left:4px solid {color};
        box-shadow:0 2px 6px rgba(0,0,0,0.08);margin-bottom:6px;">
        <p style="margin:0;font-size:11px;color:#666;
        font-weight:600;text-transform:uppercase;
        letter-spacing:0.5px;">{label}</p>
        <p style="margin:4px 0 0;font-size:24px;color:#1a1a2e;
        font-weight:700;">{value}</p>{sub_html}</div>""",
        unsafe_allow_html=True,
    )


def format_currency(val):
    if val >= 1e9:
        return f"${val/1e9:,.2f}B"
    if val >= 1e6:
        return f"${val/1e6:,.1f}M"
    if val >= 1e3:
        return f"${val/1e3:,.0f}K"
    return f"${val:,.0f}"


# -------------------------------------------------------------------
# HEADER
# -------------------------------------------------------------------
status = read_status()
is_running = status.get("status") == "running"

# Status banner
if is_running:
    st.markdown(
        '<div style="background:linear-gradient(90deg,#1565C0,#0D47A1);'
        'padding:12px 24px;border-radius:8px;margin-bottom:16px;">'
        '<span style="color:white;font-size:16px;font-weight:700;">'
        'LIVE STREAMING DASHBOARD'
        '</span>'
        '<span style="color:#64B5F6;font-size:14px;margin-left:16px;">'
        'Auto-refreshing every 5 seconds | '
        f'Refresh #{refresh_count}'
        '</span></div>',
        unsafe_allow_html=True,
    )
elif status.get("status") == "stopped":
    st.markdown(
        '<div style="background:linear-gradient(90deg,#2E7D32,#1B5E20);'
        'padding:12px 24px;border-radius:8px;margin-bottom:16px;">'
        '<span style="color:white;font-size:16px;font-weight:700;">'
        'STREAMING COMPLETE'
        '</span>'
        '<span style="color:#A5D6A7;font-size:14px;margin-left:16px;">'
        'All batches processed successfully'
        '</span></div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div style="background:linear-gradient(90deg,#E65100,#BF360C);'
        'padding:12px 24px;border-radius:8px;margin-bottom:16px;">'
        '<span style="color:white;font-size:16px;font-weight:700;">'
        'WAITING FOR STREAM'
        '</span>'
        '<span style="color:#FFCC80;font-size:14px;margin-left:16px;">'
        'Waiting for streaming data...'
        '</span></div>',
        unsafe_allow_html=True,
    )

st.title("Real-Time Streaming Pipeline Dashboard")

# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
df = read_stream_data()

# Count parquet files as proxy for batches processed
parquet_files = glob.glob(
    os.path.join(STREAM_OUTPUT, "*.parquet")) if os.path.exists(
    STREAM_OUTPUT) else []
batches_processed = len(parquet_files)

total_batches = status.get("batches", 5)
batch_size = status.get("batch_size", 500)
start_time = status.get("start_time", "")
elapsed = ""
if start_time:
    try:
        elapsed_sec = time.time() - float(start_time)
        elapsed = f"{elapsed_sec:.0f}s"
    except (ValueError, TypeError):
        elapsed = "N/A"


# -------------------------------------------------------------------
# KPI ROW
# -------------------------------------------------------------------
total_rows = len(df) if len(df) > 0 else 0
total_revenue = df["line_total"].sum() if len(df) > 0 else 0
n_categories = df["category"].nunique() if len(df) > 0 else 0
n_countries = (df["shipping_country"].nunique()
               if len(df) > 0 else 0)
avg_order = (df["line_total"].mean()
             if len(df) > 0 else 0)
avg_discount = (df["discount_pct"].mean() * 100
                if len(df) > 0 else 0)

c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1:
    metric_card("Rows Processed", f"{total_rows:,}",
                color=COLORS["primary"],
                subtitle=f"of {total_batches * batch_size:,} expected")
with c2:
    metric_card("Batches Detected", f"{batches_processed}",
                color=COLORS["green"],
                subtitle=f"of {total_batches} total")
with c3:
    metric_card("Total Revenue", format_currency(total_revenue),
                color=COLORS["orange"],
                subtitle="Running total")
with c4:
    metric_card("Categories", f"{n_categories}",
                color=COLORS["purple"],
                subtitle=f"{n_countries} countries")
with c5:
    metric_card("Avg Order Value", format_currency(avg_order),
                color=COLORS["teal"],
                subtitle=f"Discount: {avg_discount:.1f}%")
with c6:
    status_label = status.get("status", "waiting").upper()
    metric_card("Stream Status", status_label,
                color=COLORS["green"] if is_running
                else COLORS["red"],
                subtitle=f"Elapsed: {elapsed}" if elapsed
                else "Trigger: 5s")

# -------------------------------------------------------------------
# PROGRESS BAR
# -------------------------------------------------------------------
progress = min(batches_processed / max(total_batches, 1), 1.0)
st.progress(progress,
            text=f"Streaming Progress: {batches_processed}/"
                 f"{total_batches} batches "
                 f"({progress*100:.0f}%)")

st.markdown("###")

# -------------------------------------------------------------------
# CHARTS (only if data exists)
# -------------------------------------------------------------------
if len(df) > 0:

    col1, col2 = st.columns(2)

    # --- Revenue by Category ---
    with col1:
        cat_rev = (df.groupby("category")["line_total"]
                   .sum().sort_values(ascending=True))
        fig = go.Figure(go.Bar(
            y=cat_rev.index, x=cat_rev.values,
            orientation="h",
            marker=dict(
                color=cat_rev.values,
                colorscale=[[0, "#BBDEFB"], [1, "#1565C0"]]),
            text=[format_currency(v) for v in cat_rev.values],
            textposition="outside", textfont=dict(size=10),
        ))
        fig.update_layout(
            title=dict(
                text="Revenue by Category (Live)",
                font=dict(size=15, color="#2c3e50")),
            plot_bgcolor="white", height=400,
            margin=dict(l=10, r=60, t=50, b=30),
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
            font=dict(family="Segoe UI, sans-serif", size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Orders by Country ---
    with col2:
        country_rev = (df.groupby("shipping_country")
                       ["line_total"].sum()
                       .sort_values(ascending=True).tail(10))
        fig = go.Figure(go.Bar(
            y=country_rev.index, x=country_rev.values,
            orientation="h",
            marker=dict(
                color=country_rev.values,
                colorscale=[[0, "#C8E6C9"], [1, "#2E7D32"]]),
            text=[format_currency(v) for v in country_rev.values],
            textposition="outside", textfont=dict(size=10),
        ))
        fig.update_layout(
            title=dict(
                text="Top 10 Countries by Revenue (Live)",
                font=dict(size=15, color="#2c3e50")),
            plot_bgcolor="white", height=400,
            margin=dict(l=10, r=60, t=50, b=30),
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
            font=dict(family="Segoe UI, sans-serif", size=11),
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    # --- Revenue Band Distribution ---
    with col3:
        if "revenue_band" in df.columns:
            band = (df["revenue_band"].value_counts()
                    .sort_index())
            fig = go.Figure(go.Bar(
                x=band.index, y=band.values,
                marker_color=PALETTE[:len(band)],
                text=[f"{v:,}" for v in band.values],
                textposition="outside",
            ))
            fig.update_layout(
                title=dict(
                    text="Revenue Band Distribution (Live)",
                    font=dict(size=15, color="#2c3e50")),
                plot_bgcolor="white", height=380,
                margin=dict(l=10, r=10, t=50, b=30),
                yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                font=dict(family="Segoe UI, sans-serif",
                          size=11),
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Payment Method ---
    with col4:
        if "payment_method" in df.columns:
            pay = (df.groupby("payment_method")
                   ["line_total"].sum()
                   .sort_values(ascending=False))
            fig = go.Figure(go.Bar(
                x=pay.index, y=pay.values,
                marker_color=PALETTE[:len(pay)],
                text=[format_currency(v) for v in pay.values],
                textposition="outside",
            ))
            fig.update_layout(
                title=dict(
                    text="Revenue by Payment Method (Live)",
                    font=dict(size=15, color="#2c3e50")),
                plot_bgcolor="white", height=380,
                margin=dict(l=10, r=10, t=50, b=30),
                yaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                font=dict(family="Segoe UI, sans-serif",
                          size=11),
            )
            st.plotly_chart(fig, use_container_width=True)

    col5, col6 = st.columns(2)

    # --- Customer Segment ---
    with col5:
        if "customer_segment" in df.columns:
            seg = (df.groupby("customer_segment")
                   ["line_total"].sum()
                   .sort_values(ascending=True))
            fig = go.Figure(go.Bar(
                y=seg.index, x=seg.values,
                orientation="h",
                marker=dict(
                    color=seg.values,
                    colorscale=[[0, "#E1BEE7"],
                                [1, "#6A1B9A"]]),
                text=[format_currency(v) for v in seg.values],
                textposition="outside",
            ))
            fig.update_layout(
                title=dict(
                    text="Revenue by Customer Segment (Live)",
                    font=dict(size=15, color="#2c3e50")),
                plot_bgcolor="white", height=380,
                margin=dict(l=10, r=60, t=50, b=30),
                xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
                font=dict(family="Segoe UI, sans-serif",
                          size=11),
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Discount vs Full Price ---
    with col6:
        if "is_discounted" in df.columns:
            disc = (df.groupby("is_discounted")
                    .agg(count=("order_item_id", "count"),
                         revenue=("line_total", "sum"))
                    .reset_index())
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Order Count", "Revenue"))
            fig.add_trace(go.Bar(
                x=disc["is_discounted"], y=disc["count"],
                marker_color=[COLORS["primary"],
                              COLORS["orange"]],
                showlegend=False,
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                x=disc["is_discounted"], y=disc["revenue"],
                marker_color=[COLORS["primary"],
                              COLORS["orange"]],
                showlegend=False,
            ), row=1, col=2)
            fig.update_layout(
                title=dict(
                    text="Discount vs Full Price (Live)",
                    font=dict(size=15, color="#2c3e50")),
                plot_bgcolor="white", height=380,
                margin=dict(l=10, r=10, t=60, b=30),
                font=dict(family="Segoe UI, sans-serif",
                          size=11),
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Live Data Table ---
    st.markdown("### Latest Streamed Records")
    display_cols = [c for c in [
        "order_id", "category", "subcategory",
        "line_total", "quantity", "payment_method",
        "shipping_country", "customer_segment",
        "revenue_band", "is_discounted"
    ] if c in df.columns]
    st.dataframe(
        df[display_cols].tail(20).sort_index(ascending=False),
        use_container_width=True, hide_index=True,
        height=300)

else:
    st.markdown("###")
    st.info(
        "Waiting for streaming data to arrive... "
        "Dashboard will update automatically when "
        "Spark processes the first batch.",
        icon="--"
    )
    st.markdown(
        '<div style="text-align:center;padding:80px 0;">'
        '<p style="font-size:48px;color:#ccc;">...</p>'
        '<p style="color:#999;font-size:16px;">'
        'Spark Structured Streaming is processing batches<br>'
        'Data will appear here as each batch completes'
        '</p></div>',
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# PIPELINE ARCHITECTURE (sidebar)
# -------------------------------------------------------------------
st.sidebar.markdown("### Streaming Pipeline")
st.sidebar.markdown(f"""
**Source**: JSON file stream
**Schema**: 21 columns
**Trigger**: Every 5 seconds
**Files/Trigger**: 1
**Output**: Parquet (append mode)
**Batches**: {total_batches}
**Batch Size**: {batch_size:,} rows

---
**Transformations**:
- revenue_band (4 tiers)
- is_discounted flag
- NULL filters

---
**Refresh**: #{refresh_count}
**Status**: {status.get('status', 'waiting')}
""")

# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<small style="color:#888;">'
    'Real-Time Streaming Dashboard | '
    'Auto-refreshes every 5 seconds | '
    'Built with Streamlit + Plotly | '
    f'Rows: {total_rows:,} | '
    f'Batches: {batches_processed}/{total_batches}'
    '</small>',
    unsafe_allow_html=True,
)
