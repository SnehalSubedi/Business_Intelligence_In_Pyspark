"""
E-Commerce Big Data Analytics Dashboard
========================================
Professional Business Intelligence Dashboard
Star Schema Data Warehouse | K-Means Clustering | ML Models | MBA

Run:  venv/bin/streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import numpy as np

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="E-Commerce BI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(BASE, "output", "parquet")

PALETTE = ["#0061F2", "#00BA88", "#F4A100", "#7C3AED",
           "#E02D1B", "#0097A7", "#E91E9C", "#FF6D00",
           "#2962FF", "#00BFA5"]

GRADIENT_BLUE = [[0, "#DBEAFE"], [1, "#1E40AF"]]
GRADIENT_GREEN = [[0, "#D1FAE5"], [1, "#065F46"]]
GRADIENT_ORANGE = [[0, "#FED7AA"], [1, "#C2410C"]]
GRADIENT_PURPLE = [[0, "#EDE9FE"], [1, "#5B21B6"]]

LAYOUT_DEFAULTS = dict(
    plot_bgcolor="white",
    font=dict(family="Inter, Segoe UI, sans-serif", size=12,
              color="#334155"),
    margin=dict(l=16, r=16, t=56, b=32),
    title_font=dict(size=15, color="#0F172A"),
)


# -------------------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    .block-container { padding-top: 1.5rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
    }
    [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
    [data-testid="stSidebar"] .stRadio label:hover { color: #fff !important; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px 22px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        border-left: 4px solid var(--accent);
        transition: transform 0.15s, box-shadow 0.15s;
        margin-bottom: 8px;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-label {
        margin: 0; font-size: 11px; color: #64748B;
        font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.8px; font-family: Inter, sans-serif;
    }
    .metric-value {
        margin: 6px 0 0; font-size: 28px; color: #0F172A;
        font-weight: 800; font-family: Inter, sans-serif;
        line-height: 1.1;
    }
    .metric-sub {
        margin: 4px 0 0; font-size: 12px; color: #94A3B8;
        font-family: Inter, sans-serif;
    }

    .section-header {
        font-size: 20px; font-weight: 700; color: #0F172A;
        margin: 24px 0 12px; padding-bottom: 8px;
        border-bottom: 2px solid #E2E8F0;
        font-family: Inter, sans-serif;
    }

    .insight-box {
        background: #F0F9FF; border-left: 3px solid #0061F2;
        border-radius: 8px; padding: 14px 18px; margin: 10px 0;
        font-size: 13px; color: #1E3A5F;
    }

    .page-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A5F 100%);
        padding: 28px 32px; border-radius: 14px;
        margin-bottom: 24px;
    }
    .page-header h1 {
        color: white !important; margin: 0 !important;
        font-size: 28px !important; font-weight: 800 !important;
    }
    .page-header p {
        color: #93C5FD !important; margin: 6px 0 0 !important;
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------
def metric_card(label, value, color="#0061F2", subtitle=None):
    sub = (f'<p class="metric-sub">{subtitle}</p>'
           if subtitle else "")
    st.markdown(
        f'<div class="metric-card" style="--accent:{color}">'
        f'<p class="metric-label">{label}</p>'
        f'<p class="metric-value">{value}</p>{sub}</div>',
        unsafe_allow_html=True)


def page_header(title, subtitle):
    st.markdown(
        f'<div class="page-header">'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True)


def section_header(text):
    st.markdown(f'<div class="section-header">{text}</div>',
                unsafe_allow_html=True)


def insight_box(text):
    st.markdown(f'<div class="insight-box">{text}</div>',
                unsafe_allow_html=True)


def fmt_currency(val):
    if abs(val) >= 1e9:
        return f"${val/1e9:,.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:,.1f}M"
    if abs(val) >= 1e3:
        return f"${val/1e3:,.0f}K"
    return f"${val:,.0f}"


def fmt_number(val):
    if abs(val) >= 1e6:
        return f"{val/1e6:,.1f}M"
    if abs(val) >= 1e3:
        return f"{val/1e3:,.0f}K"
    return f"{val:,.0f}"


def styled_bar_h(y, x, title, gradient=GRADIENT_BLUE,
                 height=420, text_fmt=None, **kwargs):
    text = ([fmt_currency(v) for v in x] if text_fmt == "$"
            else [fmt_number(v) for v in x] if text_fmt == "#"
            else [f"{v}" for v in x])
    fig = go.Figure(go.Bar(
        y=y, x=x, orientation="h",
        marker=dict(color=x, colorscale=gradient,
                    line=dict(color="white", width=0.5)),
        text=text, textposition="outside",
        textfont=dict(size=10), showlegend=False))
    fig.update_layout(title=title, height=height,
                      xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                      **LAYOUT_DEFAULTS, **kwargs)
    return fig


def styled_bar_v(x, y, title, colors=None, height=400,
                 text_fmt=None, **kwargs):
    text = ([fmt_currency(v) for v in y] if text_fmt == "$"
            else [fmt_number(v) for v in y] if text_fmt == "#"
            else [f"{v}" for v in y])
    fig = go.Figure(go.Bar(
        x=x, y=y,
        marker_color=colors or PALETTE[:len(x)],
        text=text, textposition="outside",
        textfont=dict(size=10), showlegend=False))
    fig.update_layout(title=title, height=height,
                      yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                      **LAYOUT_DEFAULTS, **kwargs)
    return fig


# -------------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------------
@st.cache_data(show_spinner="Loading data warehouse...")
def load_tables():
    tables = {}
    for name in ["FACT_SALES", "DIM_CUSTOMER", "DIM_PRODUCT",
                  "DIM_ORDER", "DIM_DATE", "DIM_REVIEW",
                  "DIM_SUPPLIER"]:
        path = os.path.join(PARQUET, name)
        if os.path.exists(path):
            tables[name] = pd.read_parquet(path)
    return tables


@st.cache_data(show_spinner="Joining star schema...")
def build_merged(tables):
    fact = tables["FACT_SALES"]
    m = fact.merge(
        tables["DIM_CUSTOMER"][["customer_sk", "customer_name",
            "country", "customer_segment", "age_group"]],
        on="customer_sk", how="left"
    ).merge(
        tables["DIM_PRODUCT"][["product_sk", "product_name",
            "category", "subcategory"]],
        on="product_sk", how="left"
    ).merge(
        tables["DIM_ORDER"][["order_sk", "order_id", "order_date",
            "order_status", "payment_method", "shipping_country"]],
        on="order_sk", how="left"
    ).merge(
        tables["DIM_DATE"][["date_sk", "full_date", "month_name",
            "month_number", "quarter", "year", "is_weekend"]],
        on="date_sk", how="left"
    ).merge(
        tables["DIM_SUPPLIER"][["supplier_sk", "supplier_name",
            "country"]].rename(columns={"country": "supplier_country"}),
        on="supplier_sk", how="left"
    )
    m["order_date"] = pd.to_datetime(m["order_date"], errors="coerce")
    return m


@st.cache_data
def load_clusters():
    data = {}
    for key, path in {
        "customer_profile": "output/clusters/customer_cluster_profile.csv",
        "product_profile": "output/clusters/product_cluster_profile.csv",
        "customer_detail": "output/clusters/customer_clusters",
        "product_detail": "output/clusters/product_clusters",
    }.items():
        fp = os.path.join(BASE, path)
        if os.path.exists(fp):
            if fp.endswith(".csv"):
                data[key] = pd.read_csv(fp)
            else:
                data[key] = pd.read_parquet(fp)
    return data


@st.cache_data
def load_ml():
    fp = os.path.join(BASE, "output/models/model_performance_summary.csv")
    return pd.read_csv(fp) if os.path.exists(fp) else pd.DataFrame()


@st.cache_data
def load_mba():
    data = {}
    base = os.path.join(BASE, "output/analytics")
    for key in ["mba_rules_category", "mba_rules_subcategory",
                "mba_freq_itemsets_category",
                "mba_freq_itemsets_subcategory"]:
        fp = os.path.join(base, key)
        if os.path.exists(fp):
            data[key] = pd.read_parquet(fp)
    csv_fp = os.path.join(base, "mba_summary.csv")
    if os.path.exists(csv_fp):
        data["summary"] = pd.read_csv(csv_fp)
    return data


# -------------------------------------------------------------------
# LOAD ALL DATA
# -------------------------------------------------------------------
tables = load_tables()
if not tables:
    st.error("No data found. Run the notebook first.")
    st.stop()

merged = build_merged(tables)
clusters = load_clusters()
ml_data = load_ml()
mba = load_mba()

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<p style="font-size:22px;font-weight:800;color:#fff !important;'
        'margin-bottom:4px;">Analytics Hub</p>'
        '<p style="font-size:12px;color:#64748B !important;'
        'margin-top:0;">E-Commerce BI Dashboard</p>',
        unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("Navigation", [
        "Executive Summary",
        "Revenue Deep Dive",
        "Customer Intelligence",
        "Product Performance",
        "Order & Payment",
        "Supplier Analysis",
        "Customer Segmentation (K-Means)",
        "Product Clustering (K-Means)",
        "Predictive Models (ML)",
        "Market Basket Analysis",
        "Star Schema Explorer",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown('<p style="font-size:13px;font-weight:600;'
                'color:#94A3B8 !important;">GLOBAL FILTERS</p>',
                unsafe_allow_html=True)

    years = sorted(merged["year"].dropna().unique())
    sel_years = st.multiselect("Year", years, default=years)

    categories = sorted(merged["category"].dropna().unique())
    sel_cats = st.multiselect("Category", categories,
                               default=categories)

    segments = sorted(merged["customer_segment"].dropna().unique())
    sel_segs = st.multiselect("Segment", segments, default=segments)

    mask = (merged["year"].isin(sel_years) &
            merged["category"].isin(sel_cats) &
            merged["customer_segment"].isin(sel_segs))
    df = merged[mask].copy()

    st.markdown("---")
    st.markdown(
        f'<p style="font-size:11px;color:#64748B !important;">'
        f'Showing {len(df):,} / {len(merged):,} records</p>',
        unsafe_allow_html=True)


# ===================================================================
# PAGE: EXECUTIVE SUMMARY
# ===================================================================
if page == "Executive Summary":
    page_header("Executive Summary",
                "Key performance indicators across the entire "
                "e-commerce data warehouse")

    # KPIs
    total_rev = df["line_total"].sum()
    total_orders = df["order_sk"].nunique()
    total_cust = df["customer_sk"].nunique()
    total_prod = df["product_sk"].nunique()
    avg_order = total_rev / total_orders if total_orders else 0
    avg_review = df["review_score"].mean()
    total_qty = df["quantity"].sum()
    avg_disc = df["discount_pct"].mean() * 100

    r1 = st.columns(4)
    with r1[0]:
        metric_card("Total Revenue", fmt_currency(total_rev),
                     "#0061F2", f"{total_orders:,} orders")
    with r1[1]:
        metric_card("Customers", f"{total_cust:,}",
                     "#00BA88", f"{total_prod:,} products")
    with r1[2]:
        metric_card("Avg Order Value", fmt_currency(avg_order),
                     "#F4A100", f"Discount: {avg_disc:.1f}%")
    with r1[3]:
        metric_card("Avg Review Score", f"{avg_review:.2f} / 5",
                     "#7C3AED", f"{total_qty:,} units sold")

    st.markdown("###")

    # Revenue trend + Top categories
    c1, c2 = st.columns([3, 2])
    with c1:
        monthly = (df.groupby(["year", "month_number"])
                   ["line_total"].sum().reset_index())
        monthly["period"] = (monthly["year"].astype(str) + "-" +
                             monthly["month_number"].astype(str)
                             .str.zfill(2))
        monthly = monthly.sort_values("period")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly["period"], y=monthly["line_total"],
            mode="lines+markers",
            line=dict(color="#0061F2", width=2.5),
            marker=dict(size=5, color="#0061F2"),
            fill="tozeroy",
            fillcolor="rgba(0,97,242,0.06)",
        ))
        fig.update_layout(title="Revenue Trend (Monthly)",
                          height=400,
                          xaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                          yaxis=dict(title="Revenue",
                                     gridcolor="rgba(0,0,0,0.05)"),
                          **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cat_rev = (df.groupby("category")["line_total"]
                   .sum().sort_values(ascending=True).tail(8))
        st.plotly_chart(
            styled_bar_h(cat_rev.index, cat_rev.values,
                         "Top Categories by Revenue",
                         text_fmt="$", height=400),
            use_container_width=True)

    # Row 3: Segment + Payment + Geography
    c3, c4, c5 = st.columns(3)
    with c3:
        seg = (df.groupby("customer_segment")["line_total"]
               .sum().sort_values(ascending=False))
        st.plotly_chart(
            styled_bar_v(seg.index, seg.values,
                         "Revenue by Segment",
                         text_fmt="$", height=360),
            use_container_width=True)
    with c4:
        pay = (df.groupby("payment_method")["line_total"]
               .sum().sort_values(ascending=False))
        st.plotly_chart(
            styled_bar_v(pay.index, pay.values,
                         "Revenue by Payment",
                         text_fmt="$", height=360),
            use_container_width=True)
    with c5:
        country_top = (df.groupby("country")["line_total"]
                       .sum().sort_values(ascending=True).tail(8))
        st.plotly_chart(
            styled_bar_h(country_top.index, country_top.values,
                         "Top Countries",
                         gradient=GRADIENT_GREEN,
                         text_fmt="$", height=360),
            use_container_width=True)

    # Data warehouse info
    section_header("Data Warehouse Schema")
    schema_cols = st.columns(7)
    infos = [
        ("FACT_SALES", "#0061F2"), ("DIM_CUSTOMER", "#00BA88"),
        ("DIM_PRODUCT", "#F4A100"), ("DIM_ORDER", "#7C3AED"),
        ("DIM_DATE", "#0097A7"), ("DIM_REVIEW", "#E02D1B"),
        ("DIM_SUPPLIER", "#E91E9C"),
    ]
    for col, (name, color) in zip(schema_cols, infos):
        with col:
            metric_card(name, f"{len(tables[name]):,}",
                        color, "rows")


# ===================================================================
# PAGE: REVENUE DEEP DIVE
# ===================================================================
elif page == "Revenue Deep Dive":
    page_header("Revenue Deep Dive",
                "Granular revenue analysis by category, time, "
                "discount, and channel")

    total = df["line_total"].sum()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Revenue", fmt_currency(total), "#0061F2")
    with c2:
        metric_card("Avg Unit Price",
                     fmt_currency(df["unit_price_at_sale"].mean()),
                     "#00BA88")
    with c3:
        metric_card("Avg Discount",
                     f"{df['discount_pct'].mean()*100:.1f}%",
                     "#F4A100")
    with c4:
        metric_card("Total Units", f"{df['quantity'].sum():,}",
                     "#7C3AED")

    st.markdown("###")

    # Treemap
    tree = (df.groupby(["category", "subcategory"])
            ["line_total"].sum().reset_index())
    fig = px.treemap(tree, path=["category", "subcategory"],
                     values="line_total", color="line_total",
                     color_continuous_scale="Blues")
    fig.update_layout(title="Revenue Treemap: Category / Subcategory",
                      height=520, **LAYOUT_DEFAULTS)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # Quarterly
        q_rev = (df.groupby(["year", "quarter"])
                 ["line_total"].sum().reset_index())
        q_rev["label"] = (q_rev["year"].astype(str) + " Q" +
                          q_rev["quarter"].astype(str))
        st.plotly_chart(
            styled_bar_v(q_rev["label"], q_rev["line_total"],
                         "Quarterly Revenue", text_fmt="$"),
            use_container_width=True)

    with c2:
        # Weekend vs Weekday
        wk = (df.groupby("is_weekend")
              .agg(revenue=("line_total", "sum"),
                   avg_val=("line_total", "mean"),
                   orders=("order_sk", "nunique")).reset_index())
        wk["label"] = wk["is_weekend"].map(
            {True: "Weekend", False: "Weekday"})
        fig = make_subplots(1, 3, subplot_titles=(
            "Total Revenue", "Avg Order Value", "Order Count"))
        for i, col_name in enumerate(["revenue", "avg_val", "orders"]):
            fig.add_trace(go.Bar(
                x=wk["label"], y=wk[col_name],
                marker_color=["#0061F2", "#F4A100"],
                showlegend=False), row=1, col=i+1)
        fig.update_layout(title="Weekend vs Weekday", height=380,
                          **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    # Discount impact
    section_header("Discount Impact Analysis")
    df["disc_band"] = pd.cut(
        df["discount_pct"],
        bins=[-0.01, 0, 0.05, 0.10, 0.20, 1.0],
        labels=["No Discount", "0-5%", "5-10%", "10-20%", "20%+"])
    disc = (df.groupby("disc_band", observed=True)
            .agg(revenue=("line_total", "sum"),
                 orders=("order_sk", "nunique"),
                 avg_val=("line_total", "mean")).reset_index())
    fig = make_subplots(1, 3, subplot_titles=(
        "Revenue", "Orders", "Avg Value"))
    for i, m in enumerate(["revenue", "orders", "avg_val"]):
        fig.add_trace(go.Bar(
            x=disc["disc_band"], y=disc[m],
            marker_color=PALETTE[:len(disc)],
            showlegend=False), row=1, col=i+1)
    fig.update_layout(title="Performance by Discount Band",
                      height=380, **LAYOUT_DEFAULTS)
    st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE: CUSTOMER INTELLIGENCE
# ===================================================================
elif page == "Customer Intelligence":
    page_header("Customer Intelligence",
                "Customer behavior, demographics, and lifetime value")

    n_cust = df["customer_sk"].nunique()
    rev_per_cust = df["line_total"].sum() / n_cust if n_cust else 0
    repeat = (df.groupby("customer_sk")["order_sk"].nunique() > 1).mean() * 100
    avg_items = df.groupby("customer_sk")["quantity"].sum().mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Unique Customers", f"{n_cust:,}", "#0061F2")
    with c2:
        metric_card("Revenue / Customer",
                     fmt_currency(rev_per_cust), "#00BA88")
    with c3:
        metric_card("Repeat Purchase Rate",
                     f"{repeat:.1f}%", "#F4A100")
    with c4:
        metric_card("Avg Items / Customer",
                     f"{avg_items:.1f}", "#7C3AED")

    st.markdown("###")

    c1, c2 = st.columns(2)
    with c1:
        seg_rev = (df.groupby("customer_segment")
                   .agg(revenue=("line_total", "sum"),
                        customers=("customer_sk", "nunique"))
                   .sort_values("revenue", ascending=False).reset_index())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=seg_rev["customer_segment"], y=seg_rev["revenue"],
            name="Revenue", marker_color="#0061F2", yaxis="y"))
        fig.add_trace(go.Scatter(
            x=seg_rev["customer_segment"], y=seg_rev["customers"],
            name="Customers", mode="lines+markers",
            marker=dict(size=10, color="#F4A100"),
            line=dict(width=2.5, color="#F4A100"), yaxis="y2"))
        fig.update_layout(
            title="Revenue & Customer Count by Segment",
            yaxis=dict(title="Revenue", gridcolor="rgba(0,0,0,0.05)"),
            yaxis2=dict(title="Customers", overlaying="y", side="right"),
            legend=dict(x=0.01, y=0.99), height=420, **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        age_rev = (df.groupby("age_group")["line_total"]
                   .sum().sort_values(ascending=True))
        st.plotly_chart(
            styled_bar_h(age_rev.index, age_rev.values,
                         "Revenue by Age Group",
                         gradient=GRADIENT_GREEN, text_fmt="$"),
            use_container_width=True)

    # Top countries
    country_rev = (df.groupby("country")["line_total"]
                   .sum().sort_values(ascending=True).tail(15))
    st.plotly_chart(
        styled_bar_h(country_rev.index, country_rev.values,
                     "Top 15 Countries by Revenue",
                     gradient=GRADIENT_PURPLE, text_fmt="$",
                     height=480),
        use_container_width=True)

    # Segment insights
    section_header("Segment Insights")
    seg_detail = (df.groupby("customer_segment")
                  .agg(revenue=("line_total", "sum"),
                       customers=("customer_sk", "nunique"),
                       orders=("order_sk", "nunique"),
                       avg_review=("review_score", "mean"),
                       avg_discount=("discount_pct", "mean"))
                  .sort_values("revenue", ascending=False)
                  .reset_index())
    seg_detail["rev_per_customer"] = seg_detail["revenue"] / seg_detail["customers"]
    seg_detail["avg_discount"] = (seg_detail["avg_discount"] * 100).round(1)
    seg_detail["avg_review"] = seg_detail["avg_review"].round(2)
    seg_detail["revenue"] = seg_detail["revenue"].apply(fmt_currency)
    seg_detail["rev_per_customer"] = seg_detail["rev_per_customer"].apply(fmt_currency)
    st.dataframe(seg_detail, use_container_width=True, hide_index=True)


# ===================================================================
# PAGE: PRODUCT PERFORMANCE
# ===================================================================
elif page == "Product Performance":
    page_header("Product Performance",
                "Category breakdown, review analysis, and product metrics")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Products", f"{df['product_sk'].nunique():,}",
                     "#0061F2")
    with c2:
        metric_card("Categories", f"{df['category'].nunique()}",
                     "#00BA88")
    with c3:
        metric_card("Subcategories", f"{df['subcategory'].nunique()}",
                     "#F4A100")
    with c4:
        metric_card("Avg Rev / Product",
                     fmt_currency(df["line_total"].sum() /
                                  max(df["product_sk"].nunique(), 1)),
                     "#7C3AED")

    st.markdown("###")

    c1, c2 = st.columns(2)
    with c1:
        cat_stats = (df.groupby("category")
                     .agg(revenue=("line_total", "sum"),
                          avg_review=("review_score", "mean"))
                     .sort_values("revenue", ascending=False)
                     .head(10).reset_index())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=cat_stats["category"], y=cat_stats["revenue"],
            name="Revenue", marker_color="#0061F2"))
        fig.add_trace(go.Scatter(
            x=cat_stats["category"], y=cat_stats["avg_review"],
            name="Avg Review", yaxis="y2",
            mode="lines+markers",
            line=dict(color="#F4A100", width=2.5),
            marker=dict(size=9, color="#F4A100")))
        fig.update_layout(
            title="Top 10 Categories: Revenue & Review Score",
            yaxis=dict(title="Revenue", gridcolor="rgba(0,0,0,0.05)"),
            yaxis2=dict(title="Avg Review", overlaying="y",
                        side="right", range=[0, 5]),
            height=420, **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sub_rev = (df.groupby("subcategory")["line_total"]
                   .sum().sort_values(ascending=True).tail(15))
        st.plotly_chart(
            styled_bar_h(sub_rev.index, sub_rev.values,
                         "Top 15 Subcategories",
                         gradient=GRADIENT_ORANGE, text_fmt="$"),
            use_container_width=True)

    # Review distribution
    section_header("Review Score Distribution")
    review_dist = df["review_score"].value_counts().sort_index()
    review_colors = ["#EF4444", "#F97316", "#EAB308",
                     "#22C55E", "#0061F2"]
    fig = go.Figure(go.Bar(
        x=review_dist.index.astype(str), y=review_dist.values,
        marker_color=review_colors[:len(review_dist)],
        text=[f"{v:,}" for v in review_dist.values],
        textposition="outside"))
    fig.update_layout(title="Review Score Distribution",
                      xaxis_title="Score", yaxis_title="Count",
                      height=380, **LAYOUT_DEFAULTS)
    st.plotly_chart(fig, use_container_width=True)

    # Category x Review heatmap
    cat_review = (df.groupby(["category", "review_score"])
                  .size().reset_index(name="count"))
    pivot = cat_review.pivot(index="category",
                             columns="review_score",
                             values="count").fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns],
        y=pivot.index,
        colorscale="Blues", texttemplate="%{z:,.0f}",
        textfont=dict(size=9)))
    fig.update_layout(title="Category x Review Score Heatmap",
                      height=450, **LAYOUT_DEFAULTS)
    st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE: ORDER & PAYMENT
# ===================================================================
elif page == "Order & Payment":
    page_header("Order & Payment Analysis",
                "Order status, payment channels, and shipping analysis")

    n_orders = df["order_sk"].nunique()
    status_counts = df["order_status"].value_counts()
    delivered = status_counts.get("Delivered", 0)
    del_rate = delivered / len(df) * 100 if len(df) else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Orders", f"{n_orders:,}", "#0061F2")
    with c2:
        metric_card("Avg Line Value",
                     fmt_currency(df["line_total"].mean()),
                     "#00BA88")
    with c3:
        metric_card("Delivery Rate", f"{del_rate:.1f}%", "#F4A100")
    with c4:
        metric_card("Payment Methods",
                     f"{df['payment_method'].nunique()}", "#7C3AED")

    st.markdown("###")

    c1, c2 = st.columns(2)
    with c1:
        status = (df["order_status"].value_counts()
                  .sort_values(ascending=True))
        st.plotly_chart(
            styled_bar_h(status.index, status.values,
                         "Order Status Distribution",
                         text_fmt="#"),
            use_container_width=True)
    with c2:
        pay_rev = (df.groupby("payment_method")
                   .agg(revenue=("line_total", "sum"),
                        orders=("order_sk", "nunique"))
                   .sort_values("revenue", ascending=False)
                   .reset_index())
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pay_rev["payment_method"], y=pay_rev["revenue"],
            name="Revenue", marker_color="#0061F2"))
        fig.add_trace(go.Scatter(
            x=pay_rev["payment_method"], y=pay_rev["orders"],
            name="Orders", yaxis="y2", mode="lines+markers",
            marker=dict(size=10, color="#F4A100"),
            line=dict(width=2.5, color="#F4A100")))
        fig.update_layout(
            title="Payment: Revenue & Orders",
            yaxis=dict(title="Revenue", gridcolor="rgba(0,0,0,0.05)"),
            yaxis2=dict(title="Orders", overlaying="y", side="right"),
            height=420, **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    # Shipping countries
    ship = (df.groupby("shipping_country")["line_total"]
            .sum().sort_values(ascending=True).tail(20))
    st.plotly_chart(
        styled_bar_h(ship.index, ship.values,
                     "Top 20 Shipping Destinations by Revenue",
                     gradient=GRADIENT_GREEN, text_fmt="$",
                     height=520),
        use_container_width=True)


# ===================================================================
# PAGE: SUPPLIER ANALYSIS
# ===================================================================
elif page == "Supplier Analysis":
    page_header("Supplier Analysis",
                "Supplier revenue, product distribution, and performance")

    n_sup = df["supplier_sk"].nunique()
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Suppliers", f"{n_sup:,}", "#0061F2")
    with c2:
        metric_card("Avg Rev / Supplier",
                     fmt_currency(df["line_total"].sum() / max(n_sup, 1)),
                     "#00BA88")
    with c3:
        metric_card("Supplier Countries",
                     f"{df['supplier_country'].nunique()}", "#F4A100")

    st.markdown("###")

    c1, c2 = st.columns(2)
    with c1:
        sup_rev = (df.groupby("supplier_name")["line_total"]
                   .sum().sort_values(ascending=True).tail(15))
        st.plotly_chart(
            styled_bar_h(sup_rev.index, sup_rev.values,
                         "Top 15 Suppliers by Revenue",
                         gradient=GRADIENT_GREEN, text_fmt="$",
                         height=480),
            use_container_width=True)
    with c2:
        sup_c = (df.groupby("supplier_country")["line_total"]
                 .sum().sort_values(ascending=True).tail(10))
        st.plotly_chart(
            styled_bar_h(sup_c.index, sup_c.values,
                         "Top Supplier Countries",
                         gradient=GRADIENT_PURPLE, text_fmt="$",
                         height=480),
            use_container_width=True)

    section_header("Supplier Performance Table")
    sup_tbl = (df.groupby("supplier_name")
               .agg(revenue=("line_total", "sum"),
                    orders=("order_sk", "nunique"),
                    products=("product_sk", "nunique"),
                    avg_review=("review_score", "mean"),
                    avg_discount=("discount_pct", "mean"))
               .sort_values("revenue", ascending=False)
               .head(25).reset_index())
    sup_tbl["avg_discount"] = (sup_tbl["avg_discount"] * 100).round(1)
    sup_tbl["avg_review"] = sup_tbl["avg_review"].round(2)
    sup_tbl["revenue"] = sup_tbl["revenue"].apply(fmt_currency)
    st.dataframe(sup_tbl, use_container_width=True, hide_index=True)


# ===================================================================
# PAGE: CUSTOMER SEGMENTATION (K-Means)
# ===================================================================
elif page == "Customer Segmentation (K-Means)":
    page_header("Customer Segmentation",
                "RFM-based K-Means clustering results with "
                "4 customer segments")

    if "customer_profile" not in clusters:
        st.warning("Customer cluster data not found.")
        st.stop()

    cc = clusters["customer_profile"]

    # Segment KPIs
    cols = st.columns(len(cc))
    seg_colors = ["#0061F2", "#00BA88", "#F4A100", "#7C3AED"]
    for col, (_, row) in zip(cols, cc.iterrows()):
        with col:
            metric_card(
                row["segment_label"],
                f"{row['customer_count']:,}",
                seg_colors[int(row["cluster"]) % 4],
                f"Revenue: {fmt_currency(row['total_revenue'])}")

    st.markdown("###")

    # Radar chart
    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        metrics = ["avg_recency", "avg_frequency", "avg_monetary",
                   "avg_order_value", "avg_review_score"]
        labels = ["Recency", "Frequency", "Monetary",
                  "Avg Order Value", "Avg Review"]
        # Normalize for radar
        norm_vals = cc[metrics].copy()
        for m in metrics:
            rng = norm_vals[m].max() - norm_vals[m].min()
            norm_vals[m] = ((norm_vals[m] - norm_vals[m].min()) /
                            rng if rng > 0 else 0.5)

        for i, (_, row) in enumerate(cc.iterrows()):
            vals = [norm_vals.iloc[i][m] for m in metrics]
            vals.append(vals[0])  # close the polygon
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels + [labels[0]],
                fill="toself", name=row["segment_label"],
                line=dict(color=seg_colors[i], width=2),
                fillcolor=f"rgba({int(seg_colors[i][1:3],16)},"
                          f"{int(seg_colors[i][3:5],16)},"
                          f"{int(seg_colors[i][5:7],16)},0.12)"))
        fig.update_layout(
            title="RFM Cluster Radar (Normalized)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.1],
                                       gridcolor="rgba(0,0,0,0.08)")),
            height=480, **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure(go.Bar(
            x=cc["segment_label"], y=cc["total_revenue"],
            marker_color=seg_colors[:len(cc)],
            text=[fmt_currency(v) for v in cc["total_revenue"]],
            textposition="outside"))
        fig.update_layout(title="Total Revenue by Segment",
                          height=480, **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    # Detailed profile table
    section_header("Cluster Profile Details")
    display_cc = cc[["segment_label", "customer_count",
                     "avg_recency", "avg_frequency",
                     "avg_monetary", "avg_order_value",
                     "avg_review_score", "total_revenue",
                     "rfm_score"]].copy()
    display_cc.columns = ["Segment", "Customers", "Avg Recency (days)",
                          "Avg Frequency", "Avg Monetary",
                          "Avg Order Value", "Avg Review",
                          "Total Revenue", "RFM Score"]
    st.dataframe(display_cc.style.format({
        "Avg Monetary": "${:,.0f}",
        "Avg Order Value": "${:,.0f}",
        "Total Revenue": "${:,.0f}",
        "Avg Review": "{:.2f}",
        "Avg Recency (days)": "{:.0f}",
        "Avg Frequency": "{:.1f}",
    }), use_container_width=True, hide_index=True)

    # Customer distribution by segment
    if "customer_detail" in clusters:
        section_header("Individual Customer Distribution")
        cd = clusters["customer_detail"]

        c1, c2 = st.columns(2)
        with c1:
            seg_age = (cd.groupby(["segment_label", "age_group"])
                       .size().reset_index(name="count"))
            fig = px.bar(seg_age, x="segment_label", y="count",
                         color="age_group",
                         color_discrete_sequence=PALETTE,
                         barmode="group")
            fig.update_layout(title="Age Group by Segment",
                              height=400, **LAYOUT_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            seg_country = (cd.groupby(["segment_label", "country"])
                           .size().reset_index(name="count"))
            top_countries = (seg_country.groupby("country")["count"]
                             .sum().nlargest(8).index)
            seg_country = seg_country[
                seg_country["country"].isin(top_countries)]
            fig = px.bar(seg_country, x="segment_label", y="count",
                         color="country",
                         color_discrete_sequence=PALETTE,
                         barmode="stack")
            fig.update_layout(title="Top Countries by Segment",
                              height=400, **LAYOUT_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: Monetary vs Frequency
        sample = cd.sample(min(5000, len(cd)), random_state=42)
        fig = px.scatter(
            sample, x="frequency", y="monetary",
            color="segment_label",
            color_discrete_sequence=seg_colors,
            opacity=0.5, size_max=8,
            hover_data=["customer_name", "country"])
        fig.update_layout(title="Monetary vs Frequency (sampled)",
                          height=480, **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE: PRODUCT CLUSTERING (K-Means)
# ===================================================================
elif page == "Product Clustering (K-Means)":
    page_header("Product Clustering",
                "K-Means product segmentation: Bestseller, "
                "Mid-Range, Slow Mover")

    if "product_profile" not in clusters:
        st.warning("Product cluster data not found.")
        st.stop()

    pc = clusters["product_profile"]
    prod_colors = ["#0061F2", "#00BA88", "#F4A100"]

    cols = st.columns(len(pc))
    for col, (_, row) in zip(cols, pc.iterrows()):
        with col:
            metric_card(
                row["product_label"],
                f"{row['product_count']:,} products",
                prod_colors[int(row["cluster"]) % 3],
                f"Revenue: {fmt_currency(row['cluster_total_revenue'])}")

    st.markdown("###")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        metrics = ["avg_sales_volume", "avg_total_revenue",
                   "avg_sale_price", "avg_review_score",
                   "avg_units_sold"]
        labels = ["Sales Volume", "Revenue", "Avg Price",
                  "Review Score", "Units Sold"]
        norm = pc[metrics].copy()
        for m in metrics:
            rng = norm[m].max() - norm[m].min()
            norm[m] = (norm[m] - norm[m].min()) / rng if rng > 0 else 0.5

        for i, (_, row) in enumerate(pc.iterrows()):
            vals = [norm.iloc[i][m] for m in metrics]
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels + [labels[0]],
                fill="toself", name=row["product_label"],
                line=dict(color=prod_colors[i], width=2),
                fillcolor=f"rgba({int(prod_colors[i][1:3],16)},"
                          f"{int(prod_colors[i][3:5],16)},"
                          f"{int(prod_colors[i][5:7],16)},0.12)"))
        fig.update_layout(
            title="Product Cluster Radar (Normalized)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.1])),
            height=460, **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        for i, (_, row) in enumerate(pc.iterrows()):
            fig.add_trace(go.Bar(
                x=["Products", "Avg Volume", "Avg Revenue (K)",
                   "Avg Price"],
                y=[row["product_count"], row["avg_sales_volume"],
                   row["avg_total_revenue"] / 1000,
                   row["avg_sale_price"]],
                name=row["product_label"],
                marker_color=prod_colors[i]))
        fig.update_layout(title="Cluster Comparison",
                          barmode="group", height=460,
                          **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    section_header("Cluster Profile Table")
    st.dataframe(pc.style.format({
        "avg_total_revenue": "${:,.0f}",
        "avg_sale_price": "${:,.0f}",
        "cluster_total_revenue": "${:,.0f}",
    }), use_container_width=True, hide_index=True)

    # Product detail scatter
    if "product_detail" in clusters:
        section_header("Product Scatter: Revenue vs Sales Volume")
        pd_detail = clusters["product_detail"]
        sample = pd_detail.sample(min(3000, len(pd_detail)),
                                   random_state=42)
        fig = px.scatter(
            sample, x="sales_volume", y="total_revenue",
            color="product_label",
            color_discrete_sequence=prod_colors,
            opacity=0.5,
            hover_data=["product_name", "category"])
        fig.update_layout(title="Revenue vs Sales Volume (sampled)",
                          height=460, **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE: PREDICTIVE MODELS (ML)
# ===================================================================
elif page == "Predictive Models (ML)":
    page_header("Predictive Models",
                "Linear Regression, Random Forest, and "
                "Logistic Regression results")

    if ml_data.empty:
        st.warning("Model performance data not found.")
        st.stop()

    # Model cards
    cols = st.columns(3)
    ml_colors = ["#0061F2", "#00BA88", "#F4A100"]
    ml_icons = ["Revenue Forecasting", "Satisfaction Prediction",
                "Cancellation Detection"]
    for col, (_, row), color, icon in zip(
            cols, ml_data.iterrows(), ml_colors, ml_icons):
        with col:
            if pd.notna(row.get("r2")):
                main_val = f"R\u00b2 = {row['r2']:.4f}"
                sub = f"RMSE: {row['rmse']:,.2f} | MAE: {row['mae']:,.2f}"
            else:
                main_val = f"Accuracy = {row['accuracy']:.4f}"
                sub = f"F1 Score: {row['f1']:.4f}"
            metric_card(row["model"], main_val, color,
                        f"{icon} | {sub}")

    st.markdown("###")

    c1, c2 = st.columns(2)
    with c1:
        # Performance comparison
        fig = go.Figure()
        for i, (_, row) in enumerate(ml_data.iterrows()):
            score = row["r2"] if pd.notna(row.get("r2")) else row["accuracy"]
            fig.add_trace(go.Bar(
                x=[row["model"]], y=[score],
                marker_color=ml_colors[i],
                text=[f"{score:.4f}"], textposition="outside",
                showlegend=False))
        fig.update_layout(
            title="Model Performance (R\u00b2 / Accuracy)",
            yaxis=dict(title="Score", range=[0, 1],
                       gridcolor="rgba(0,0,0,0.05)"),
            height=420, **LAYOUT_DEFAULTS)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Error metrics for regression models
        reg_models = ml_data[ml_data["r2"].notna()]
        if len(reg_models) > 0:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=reg_models["model"],
                y=reg_models["rmse"],
                name="RMSE", marker_color="#0061F2"))
            fig.add_trace(go.Bar(
                x=reg_models["model"],
                y=reg_models["mae"],
                name="MAE", marker_color="#F4A100"))
            fig.update_layout(
                title="Regression Error Metrics",
                barmode="group", height=420,
                yaxis=dict(gridcolor="rgba(0,0,0,0.05)"),
                **LAYOUT_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

    # Model details table
    section_header("Model Details")
    display_ml = ml_data.copy()
    display_ml.columns = ["Model", "Target", "R\u00b2", "RMSE",
                           "MAE", "Accuracy", "F1", "Business Use"]
    st.dataframe(display_ml, use_container_width=True,
                 hide_index=True)

    # Business interpretation
    section_header("Business Interpretation")
    for _, row in ml_data.iterrows():
        if pd.notna(row.get("r2")):
            pct = row["r2"] * 100
            insight_box(
                f"<b>{row['model']}</b> explains "
                f"<b>{pct:.1f}%</b> of variance in "
                f"<b>{row['target']}</b>. "
                f"Business use: {row['business_use']}.")
        else:
            insight_box(
                f"<b>{row['model']}</b> achieves "
                f"<b>{row['accuracy']*100:.1f}%</b> accuracy "
                f"for <b>{row['target']}</b>. "
                f"F1={row['f1']:.4f}. "
                f"Business use: {row['business_use']}.")


# ===================================================================
# PAGE: MARKET BASKET ANALYSIS
# ===================================================================
elif page == "Market Basket Analysis":
    page_header("Market Basket Analysis",
                "FP-Growth association rules: product co-purchase "
                "patterns")

    if "summary" in mba:
        ms = mba["summary"]
        vals = dict(zip(ms["metric"], ms["value"]))
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Orders Analysed",
                        f"{int(vals.get('total_orders_analysed', 0)):,}",
                        "#0061F2")
        with c2:
            metric_card("Multi-Category Txns",
                        f"{int(vals.get('multi_category_transactions', 0)):,}",
                        "#00BA88",
                        f"{vals.get('multi_category_rate_pct', 0):.1f}%")
        with c3:
            metric_card("Category Rules",
                        f"{int(vals.get('category_rules_found', 0))}",
                        "#F4A100")
        with c4:
            metric_card("Subcategory Rules",
                        f"{int(vals.get('subcategory_rules', 0))}",
                        "#7C3AED")

    st.markdown("###")

    c1, c2 = st.columns(2)

    # Category rules
    if "mba_rules_category" in mba:
        rules_cat = mba["mba_rules_category"].sort_values(
            "lift", ascending=False)
        with c1:
            labels = [f"{r['antecedent_str']} -> {r['consequent_str']}"
                      for _, r in rules_cat.iterrows()]
            fig = go.Figure(go.Bar(
                y=labels[::-1], x=rules_cat["lift"][::-1].values,
                orientation="h",
                marker=dict(color=rules_cat["lift"][::-1].values,
                            colorscale=GRADIENT_ORANGE),
                text=[f"{v:.4f}" for v in rules_cat["lift"][::-1].values],
                textposition="outside"))
            fig.add_vline(x=1.0, line_dash="dash", line_color="red",
                          annotation_text="Lift=1")
            fig.update_layout(title="Category Rules by Lift",
                              height=420, **LAYOUT_DEFAULTS,
                              margin=dict(l=200, r=60, t=56, b=32))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Bubble chart: support vs confidence
            fig = go.Figure(go.Scatter(
                x=rules_cat["support"],
                y=rules_cat["confidence"],
                mode="markers+text",
                marker=dict(
                    size=rules_cat["lift"] * 30,
                    color=rules_cat["lift"],
                    colorscale="YlOrRd",
                    colorbar=dict(title="Lift"),
                    opacity=0.8),
                text=[f"{r['antecedent_str'][:12]}"
                      for _, r in rules_cat.iterrows()],
                textposition="top center",
                textfont=dict(size=9)))
            fig.update_layout(
                title="Support vs Confidence (size=Lift)",
                xaxis=dict(title="Support",
                           gridcolor="rgba(0,0,0,0.05)"),
                yaxis=dict(title="Confidence",
                           gridcolor="rgba(0,0,0,0.05)"),
                height=420, **LAYOUT_DEFAULTS)
            st.plotly_chart(fig, use_container_width=True)

    # Subcategory rules
    if "mba_rules_subcategory" in mba:
        section_header("Top Subcategory Association Rules")
        rules_sub = mba["mba_rules_subcategory"].sort_values(
            "lift", ascending=False).head(15)
        labels = [f"{r['antecedent_str'][:25]} -> {r['consequent_str'][:22]}"
                  for _, r in rules_sub.iterrows()]
        fig = go.Figure(go.Bar(
            y=labels[::-1], x=rules_sub["lift"][::-1].values,
            orientation="h",
            marker=dict(color=rules_sub["lift"][::-1].values,
                        colorscale=GRADIENT_PURPLE),
            text=[f"{v:.3f}" for v in rules_sub["lift"][::-1].values],
            textposition="outside"))
        fig.add_vline(x=1.0, line_dash="dash", line_color="red")
        fig.update_layout(title="Top 15 Subcategory Rules by Lift",
                          height=520, **LAYOUT_DEFAULTS,
                          margin=dict(l=280, r=60, t=56, b=32))
        st.plotly_chart(fig, use_container_width=True)

    # Frequent itemsets
    if "mba_freq_itemsets_category" in mba:
        section_header("Frequent Category Pairs")
        freq = mba["mba_freq_itemsets_category"]
        freq_2 = freq[freq["items"].apply(len) == 2].sort_values(
            "freq", ascending=False).head(10)
        if len(freq_2) > 0:
            freq_2["label"] = freq_2["items"].apply(
                lambda x: " + ".join(sorted(x)))
            fig = go.Figure(go.Bar(
                y=freq_2["label"][::-1],
                x=freq_2["freq"][::-1],
                orientation="h",
                marker=dict(color=freq_2["freq"][::-1].values,
                            colorscale=GRADIENT_GREEN),
                text=[f"{v:,}" for v in freq_2["freq"][::-1]],
                textposition="outside"))
            fig.update_layout(title="Most Frequent 2-Category Combos",
                              height=400, **LAYOUT_DEFAULTS,
                              margin=dict(l=200, r=60, t=56, b=32))
            st.plotly_chart(fig, use_container_width=True)

    # MBA metrics table
    section_header("MBA Metrics Reference")
    fig = go.Figure(go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Formula</b>",
                    "<b>Interpretation</b>"],
            fill_color="#0F172A", font=dict(color="white", size=12),
            align="center", height=32),
        cells=dict(
            values=[
                ["Support", "Confidence", "Lift (>1)", "Lift (=1)",
                 "Lift (<1)"],
                ["P(A and B)", "P(B|A)", "Confidence / P(B)", "-", "-"],
                ["How often A+B co-occur",
                 "Probability of B given A",
                 "Strong positive association",
                 "No association (random)",
                 "Negative association"]],
            fill_color=[["#F1F5F9", "white"] * 3],
            font=dict(size=11), align="center", height=28)))
    fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)


# ===================================================================
# PAGE: STAR SCHEMA EXPLORER
# ===================================================================
elif page == "Star Schema Explorer":
    page_header("Star Schema Explorer",
                "Browse and explore all dimension and fact tables")

    # Schema diagram
    section_header("Star Schema Architecture")
    c1, c2, c3 = st.columns([1, 3, 1])
    with c2:
        fig = go.Figure()
        # Center: FACT_SALES
        fig.add_shape(type="rect", x0=2.5, y0=2, x1=5.5, y1=3.5,
                      fillcolor="#0F172A", line=dict(color="#0F172A"))
        fig.add_annotation(x=4, y=2.75,
                           text="<b>FACT_SALES</b><br>916,167 rows",
                           font=dict(color="white", size=13),
                           showarrow=False)
        # Dimensions
        dims = [
            (0.2, 4.2, 2.2, 5.2, "DIM_CUSTOMER", "120K", "#0061F2"),
            (3, 4.8, 5, 5.8, "DIM_PRODUCT", "5K", "#00BA88"),
            (5.8, 4.2, 7.8, 5.2, "DIM_ORDER", "399K", "#F4A100"),
            (0.2, 0.3, 2.2, 1.3, "DIM_DATE", "2,557", "#7C3AED"),
            (3, -0.3, 5, 0.7, "DIM_REVIEW", "280K", "#E02D1B"),
            (5.8, 0.3, 7.8, 1.3, "DIM_SUPPLIER", "200", "#0097A7"),
        ]
        for x0, y0, x1, y1, name, rows, color in dims:
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                          fillcolor=color,
                          line=dict(color=color, width=2),
                          opacity=0.9)
            fig.add_annotation(
                x=(x0+x1)/2, y=(y0+y1)/2,
                text=f"<b>{name}</b><br>{rows} rows",
                font=dict(color="white", size=10),
                showarrow=False)
            # Line to center
            fig.add_shape(type="line",
                          x0=(x0+x1)/2, y0=y0 if y0 > 2.75 else y1,
                          x1=4, y1=3.5 if y0 > 2.75 else 2,
                          line=dict(color="#94A3B8", width=1.5,
                                    dash="dot"))
        fig.update_layout(
            xaxis=dict(visible=False, range=[-0.5, 8.5]),
            yaxis=dict(visible=False, range=[-0.8, 6.3]),
            height=420, plot_bgcolor="#F8FAFC",
            margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # Table explorer
    section_header("Table Explorer")
    table_choice = st.selectbox(
        "Select table to explore",
        ["FACT_SALES (joined)", "DIM_CUSTOMER", "DIM_PRODUCT",
         "DIM_ORDER", "DIM_DATE", "DIM_REVIEW", "DIM_SUPPLIER"])

    if table_choice == "FACT_SALES (joined)":
        explore_df = df
    else:
        explore_df = tables[table_choice]

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Rows", f"{len(explore_df):,}", "#0061F2")
    with c2:
        metric_card("Columns", f"{len(explore_df.columns)}", "#00BA88")
    with c3:
        mem = explore_df.memory_usage(deep=True).sum() / 1024 / 1024
        metric_card("Memory", f"{mem:.1f} MB", "#F4A100")

    # Column info
    with st.expander("Column Schema"):
        col_info = pd.DataFrame({
            "Column": explore_df.columns,
            "Type": [str(t) for t in explore_df.dtypes],
            "Non-Null": explore_df.notna().sum().values,
            "Null %": ((explore_df.isna().sum() /
                        len(explore_df) * 100).round(1).values),
            "Unique": [explore_df[c].nunique()
                       for c in explore_df.columns],
        })
        st.dataframe(col_info, use_container_width=True,
                     hide_index=True)

    # Filter
    search_col = st.selectbox("Filter by column",
                               ["(none)"] + list(explore_df.columns))
    if search_col != "(none)":
        uniq = explore_df[search_col].dropna().unique()
        if len(uniq) <= 50:
            sel = st.multiselect(f"Values", sorted(uniq, key=str))
            if sel:
                explore_df = explore_df[
                    explore_df[search_col].isin(sel)]
        else:
            term = st.text_input(f"Search in {search_col}")
            if term:
                explore_df = explore_df[
                    explore_df[search_col].astype(str)
                    .str.contains(term, case=False, na=False)]

    n_rows = st.slider("Rows", 10, 500, 50)
    st.dataframe(explore_df.head(n_rows),
                 use_container_width=True, hide_index=True)

    # Quick stats
    num_cols = explore_df.select_dtypes(include="number").columns
    if len(num_cols) > 0:
        with st.expander("Descriptive Statistics"):
            st.dataframe(explore_df[num_cols].describe().round(2).T,
                         use_container_width=True)

    csv = explore_df.head(10000).to_csv(index=False)
    st.download_button("Download CSV (10K rows)", csv,
                       "export.csv", "text/csv")


# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="font-size:10px;color:#475569 !important;'
    'line-height:1.5;">'
    'E-Commerce BI Dashboard<br>'
    'Star Schema | K-Means | ML | MBA<br>'
    'Built with Streamlit + Plotly</p>',
    unsafe_allow_html=True)
