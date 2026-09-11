import math
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================ #
# 1. SIDEBAR MODULE NAVIGATION & STYLING
# ============================================================ #
st.sidebar.markdown(
    """
    <div style="padding: 10px 0px 15px 0px;">
        <h2 style="margin:0; font-size: 22px; font-weight: 700; color: #1E293B;">🔎 Inspectra Suite</h2>
        <p style="margin:2px 0 0 0; font-size: 12px; color: #64748B;">Audit & Financial Intelligence</p>
    </div>
""",
    unsafe_allow_html=True,
)

app_mode = st.sidebar.radio(
    "Select Service Module:",
    [
        "🚨 Fraud & Anomaly Detection",
        "📊 Financial Statement & Ratio Analyzer",
    ],
    help="Switch between 100% transaction population screening and financial statement ratio analysis.",
)

# Custom Styling for Financial Statement Module
FINANCIAL_CSS = """
<style>
    /* Metric Card styling */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08), 0 4px 6px -2px rgba(0, 0, 0, 0.04);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        color: #0f172a !important;
        padding-top: 4px;
    }
    .metric-caption {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 6px;
        line-height: 1.35;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #1e293b;
        padding-bottom: 8px;
        border-bottom: 2px solid #e2e8f0;
        margin: 28px 0 16px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .footer-disclaimer {
        margin-top: 50px;
        padding: 16px 20px;
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        font-size: 0.82rem;
        color: #64748b;
        line-height: 1.5;
    }
</style>
"""

# ============================================================ #
# 2. FINANCIAL STATEMENT & CASH FLOW ANALYZER MODULE
# ============================================================ #


def calculate_comprehensive_analytics(d):
    """Calculates Ratios, Working Capital Metrics, and Cash Flow Diagnostics."""
    ca, cl = d.get("Current Assets", 0), d.get("Current Liabilities", 1)
    inv, cash = d.get("Inventory", 0), d.get("Cash & Equivalents", 0)
    ta, eq = d.get("Total Assets", 1), d.get("Total Equity", 1)
    debt, rev = d.get("Total Debt", 0), d.get("Revenue", 1)
    cogs, ni = d.get("Cost of Goods Sold", 0), d.get("Net Income", 0)
    ebit, interest = d.get("EBIT", 0), d.get("Interest Expense", 1)
    ar, ap = d.get("Accounts Receivable", 0), d.get("Accounts Payable", 0)

    cfo = d.get("Operating Cash Flow", 0)
    capex = d.get("Capital Expenditures", 0)

    gp = rev - cogs
    avg_inv = d.get("Average Inventory", inv if inv != 0 else 1)

    dso = (ar / rev) * 365 if rev else 0
    dio = (inv / cogs) * 365 if cogs else 0
    dpo = (ap / cogs) * 365 if cogs else 0

    return {
        "Current Ratio": round(ca / cl, 2) if cl else 0,
        "Quick Ratio": round((ca - inv) / cl, 2) if cl else 0,
        "Cash Ratio": round(cash / cl, 2) if cl else 0,
        "Gross Profit Margin (%)": round((gp / rev) * 100, 2) if rev else 0,
        "Net Profit Margin (%)": round((ni / rev) * 100, 2) if rev else 0,
        "Return on Assets (%)": round((ni / ta) * 100, 2) if ta else 0,
        "Debt to Equity": round(debt / eq, 2) if eq else 0,
        "Interest Coverage Ratio": round(ebit / interest, 2) if interest else 0,
        "Asset Turnover": round(rev / ta, 2) if ta else 0,
        "DSO (Days)": round(dso, 1),
        "Inventory Turnover": round(cogs / avg_inv, 2) if avg_inv else 0,
        "Net Working Capital": ca - cl,
        "DIO (Days)": round(dio, 1),
        "DPO (Days)": round(dpo, 1),
        "Cash Conversion Cycle (CCC)": round(dso + dio - dpo, 1),
        "Operating Cash Flow (CFO)": cfo,
        "Free Cash Flow (FCF)": cfo - capex,
        "Earnings Quality (CFO/NI)": round((cfo / ni), 2) if ni else 0,
        "CFO Coverage Ratio": round((cfo / cl), 2) if cl else 0,
    }


def render_financial_analyzer():
    st.markdown(FINANCIAL_CSS, unsafe_allow_html=True)

    st.title("📊 Financial Statement & Ratio Analyzer")
    st.caption(
        "Automated financial health diagnostics across Liquidity, Profitability, Solvency, Activity, Working Capital, and Cash Flow metrics."
    )

    uploaded_fs = st.file_uploader(
        "Upload Financial Statement (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="fs_uploader",
    )

    if uploaded_fs is not None:
        try:
            df = (
                pd.read_csv(uploaded_fs)
                if uploaded_fs.name.endswith(".csv")
                else pd.read_excel(uploaded_fs)
            )
            st.session_state["financial_df"] = df
            st.success("Financial statement loaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if "financial_df" in st.session_state:
        df = st.session_state["financial_df"]

        with st.expander("📋 View Uploaded Statement Inputs", expanded=False):
            st.dataframe(df, use_container_width=True)

        if len(df.columns) >= 2:
            data_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            analytics = calculate_comprehensive_analytics(data_dict)

            # 1. Liquidity Ratios
            st.markdown(
                '<div class="section-header">💧 1. Liquidity Ratios</div>',
                unsafe_allow_html=True,
            )
            l1, l2, l3 = st.columns(3)
            with l1:
                st.metric("Current Ratio", analytics["Current Ratio"])
                st.markdown(
                    '<p class="metric-caption">Measures ability to cover short-term debts with short-term assets.</p>',
                    unsafe_allow_html=True,
                )
            with l2:
                st.metric("Quick Ratio", analytics["Quick Ratio"])
                st.markdown(
                    '<p class="metric-caption">Evaluates immediate debt-paying ability without relying on inventory.</p>',
                    unsafe_allow_html=True,
                )
            with l3:
                st.metric("Cash Ratio", analytics["Cash Ratio"])
                st.markdown(
                    '<p class="metric-caption">Shows how effectively liquid cash reserves cover immediate debts.</p>',
                    unsafe_allow_html=True,
                )

            # 2. Profitability Ratios
            st.markdown(
                '<div class="section-header">📈 2. Profitability Ratios</div>',
                unsafe_allow_html=True,
            )
            p1, p2, p3 = st.columns(3)
            with p1:
                st.metric(
                    "Gross Margin", f"{analytics['Gross Profit Margin (%)']}%"
                )
                st.markdown(
                    '<p class="metric-caption">Measures core production efficiency and pricing leverage.</p>',
                    unsafe_allow_html=True,
                )
            with p2:
                st.metric(
                    "Net Profit Margin",
                    f"{analytics['Net Profit Margin (%)']}%",
                )
                st.markdown(
                    '<p class="metric-caption">Percentage of revenue remaining after all expenses and taxes.</p>',
                    unsafe_allow_html=True,
                )
            with p3:
                st.metric(
                    "Return on Assets", f"{analytics['Return on Assets (%)']}%"
                )
                st.markdown(
                    '<p class="metric-caption">Efficiency of asset deployment for profit generation.</p>',
                    unsafe_allow_html=True,
                )

            # 3. Solvency Ratios
            st.markdown(
                '<div class="section-header">⚖️ 3. Solvency Ratios</div>',
                unsafe_allow_html=True,
            )
            s1, s2 = st.columns(2)
            with s1:
                st.metric("Debt to Equity", analytics["Debt to Equity"])
                st.markdown(
                    '<p class="metric-caption">Evaluates capital structure leverage and financial risk.</p>',
                    unsafe_allow_html=True,
                )
            with s2:
                st.metric(
                    "Interest Coverage",
                    f"{analytics['Interest Coverage Ratio']}x",
                )
                st.markdown(
                    '<p class="metric-caption">Ability of operating profit to cover debt interest payments.</p>',
                    unsafe_allow_html=True,
                )

            # 4. Activity Ratios
            st.markdown(
                '<div class="section-header">🔄 4. Activity Ratios</div>',
                unsafe_allow_html=True,
            )
            a1, a2, a3 = st.columns(3)
            with a1:
                st.metric("Asset Turnover", f"{analytics['Asset Turnover']}x")
                st.markdown(
                    '<p class="metric-caption">Top-line revenue generated per unit of total assets.</p>',
                    unsafe_allow_html=True,
                )
            with a2:
                st.metric(
                    "Days Sales Outstanding", f"{analytics['DSO (Days)']} Days"
                )
                st.markdown(
                    '<p class="metric-caption">Average collection timeline for receivables.</p>',
                    unsafe_allow_html=True,
                )
            with a3:
                st.metric(
                    "Inventory Turnover", f"{analytics['Inventory Turnover']}x"
                )
                st.markdown(
                    '<p class="metric-caption">Frequency of inventory replacement over the cycle.</p>',
                    unsafe_allow_html=True,
                )

            # 5. Working Capital Management
            st.markdown(
                '<div class="section-header">💼 5. Working Capital Management</div>',
                unsafe_allow_html=True,
            )
            w1, w2, w3, w4 = st.columns(4)
            with w1:
                st.metric(
                    "Net Working Capital",
                    f"₹{analytics['Net Working Capital']:,.2f}",
                )
                st.markdown(
                    '<p class="metric-caption">Operational liquidity cushion for daily activities.</p>',
                    unsafe_allow_html=True,
                )
            with w2:
                st.metric(
                    "Days Inventory Out", f"{analytics['DIO (Days)']} Days"
                )
                st.markdown(
                    '<p class="metric-caption">Average duration inventory remains before sale.</p>',
                    unsafe_allow_html=True,
                )
            with w3:
                st.metric(
                    "Days Payables Out", f"{analytics['DPO (Days)']} Days"
                )
                st.markdown(
                    '<p class="metric-caption">Average timeline for settling trade payables.</p>',
                    unsafe_allow_html=True,
                )
            with w4:
                st.metric(
                    "Cash Conversion Cycle",
                    f"{analytics['Cash Conversion Cycle (CCC)']} Days",
                )
                st.markdown(
                    '<p class="metric-caption">Total days needed to convert operations into cash.</p>',
                    unsafe_allow_html=True,
                )

            # 6. Cash Flow Analysis
            st.markdown(
                '<div class="section-header">💵 6. Cash Flow Analysis</div>',
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric(
                    "Operating Cash Flow",
                    f"₹{analytics['Operating Cash Flow (CFO)']:,.2f}",
                )
                st.markdown(
                    '<p class="metric-caption">Cash generated directly from core business operations.</p>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.metric(
                    "Free Cash Flow",
                    f"₹{analytics['Free Cash Flow (FCF)']:,.2f}",
                )
                st.markdown(
                    '<p class="metric-caption">Cash remaining after funding operations and capital expenditure.</p>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.metric(
                    "Earnings Quality",
                    f"{analytics['Earnings Quality (CFO/NI)']}x",
                )
                st.markdown(
                    '<p class="metric-caption">Ratio of operating cash flow to net reported earnings.</p>',
                    unsafe_allow_html=True,
                )
            with c4:
                st.metric(
                    "CFO Coverage", f"{analytics['CFO Coverage Ratio']}x"
                )
                st.markdown(
                    '<p class="metric-caption">Operating cash coverage of short-term obligations.</p>',
                    unsafe_allow_html=True,
                )

    # Clean Footer Disclaimer
    st.markdown(
        """
        <div class="footer-disclaimer">
            🔒 <strong>Privacy & Legal Disclaimer:</strong> All uploaded financial statements and data inputs are processed strictly in-memory and are never saved or stored on external servers. Calculated financial metrics and analytics serve solely as decision-support insights and do not constitute formal statutory audit conclusions, tax advice, or professional opinions.
        </div>
        """,
        unsafe_allow_html=True,
    )


# Route view execution
if app_mode == "📊 Financial Statement & Ratio Analyzer":
    render_financial_analyzer()
    st.stop()  # Prevents executing the remaining 5,100+ lines below when in ratio mode

# ============================================================ #
# EXISTING FRAUD & ANOMALY DETECTION ENGINE BELOW (UNTOUCHED)
# ============================================================ #
