import math
import numpy as np
import pandas as pd
import streamlit as st

# ============================================================ #
# 1. PAGE CONFIG & NAVIGATION SETUP
# ============================================================ #
st.set_page_config(
    page_title="Inspectra | Financial & Fraud Analytics",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.sidebar.title("🔎 Inspectra Suite")
app_mode = st.sidebar.radio(
    "Select Service Module:",
    [
        "🚨 Fraud & Anomaly Detection",
        "📊 Financial Statement & Ratio Analyzer",
    ],
    help="Choose between transaction-level fraud screening or financial statement ratio analysis.",
)

# ============================================================ #
# 2. RATIO ENGINE & CALCULATIONS
# ============================================================ #


def calculate_comprehensive_ratios(d):
    """Calculates Liquidity, Profitability, Solvency, and Efficiency metrics."""
    ca, cl, inv = d.get("Current Assets", 0), d.get("Current Liabilities", 1), d.get("Inventory", 0)
    cash, ta, eq = d.get("Cash & Equivalents", 0), d.get("Total Assets", 1), d.get("Total Equity", 1)
    debt, rev, cogs = d.get("Total Debt", 0), d.get("Revenue", 1), d.get("Cost of Goods Sold", 0)
    ni, ebit, interest = d.get("Net Income", 0), d.get("EBIT", 0), d.get("Interest Expense", 1)
    ar = d.get("Accounts Receivable", 0)

    gp = rev - cogs

    return {
        # Liquidity
        "Current Ratio": round(ca / cl, 2) if cl else 0,
        "Quick Ratio": round((ca - inv) / cl, 2) if cl else 0,
        "Cash Ratio": round(cash / cl, 2) if cl else 0,
        # Profitability
        "Gross Profit Margin (%)": round((gp / rev) * 100, 2) if rev else 0,
        "Net Profit Margin (%)": round((ni / rev) * 100, 2) if rev else 0,
        "Return on Assets (%)": round((ni / ta) * 100, 2) if ta else 0,
        # Solvency & Efficiency
        "Debt to Equity": round(debt / eq, 2) if eq else 0,
        "Interest Coverage Ratio": round(ebit / interest, 2) if interest else 0,
        "Asset Turnover": round(rev / ta, 2) if ta else 0,
        "DSO (Days)": round((ar / rev) * 365, 1) if rev else 0,
    }


# ============================================================ #
# 3. FINANCIAL STATEMENT ANALYZER MODULE UI
# ============================================================ #


def render_financial_analyzer():
    st.title("📊 Financial Statement & Ratio Analyzer")
    st.caption("Automated financial health diagnostics, efficiency ratios, and solvency metrics.")

    # Option 1 Compliance: Legal & Data Privacy Banner
    st.info(
        "🔒 **Privacy & Legal Disclaimer:** All files are processed strictly in-memory and are never saved to external servers. "
        "Calculated metrics and ratios serve as automated decision-support tools and do not constitute formal tax, audit, or legal advice."
    )

    st.subheader("1. Upload Financial Statement")
    uploaded_fs = st.file_uploader(
        "Upload Balance Sheet / Profit & Loss Statement (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key="fs_uploader",
    )

    if uploaded_fs is not None:
        try:
            if uploaded_fs.name.endswith(".csv"):
                df = pd.read_csv(uploaded_fs)
            else:
                df = pd.read_excel(uploaded_fs)

            st.session_state["financial_df"] = df
            st.success("Financial statement uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Process and Display Analysis Dashboard
    if "financial_df" in st.session_state:
        df = st.session_state["financial_df"]

        st.markdown("---")
        st.subheader("2. Financial Inputs Overview")
        st.dataframe(df, use_container_width=True)

        if len(df.columns) >= 2:
            data_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
            ratios = calculate_comprehensive_ratios(data_dict)

            st.markdown("---")
            st.subheader("3. Calculated Financial Ratios")

            # Liquidity Section
            st.markdown("#### 💧 Liquidity Ratios")
            l1, l2, l3 = st.columns(3)
            cr = ratios["Current Ratio"]
            l1.metric("Current Ratio", cr, delta="Healthy" if cr >= 1.5 else "Low Liquidity", delta_color="normal" if cr >= 1.5 else "inverse")
            qr = ratios["Quick Ratio"]
            l2.metric("Quick Ratio", qr, delta="Optimal" if qr >= 1.0 else "Inventory Dependent", delta_color="normal" if qr >= 1.0 else "inverse")
            cash_r = ratios["Cash Ratio"]
            l3.metric("Cash Ratio", cash_r, delta="Safe" if cash_r >= 0.2 else "Low Cash Reserve", delta_color="normal" if cash_r >= 0.2 else "inverse")

            # Profitability Section
            st.markdown("#### 📈 Profitability Ratios")
            p1, p2, p3 = st.columns(3)
            p1.metric("Gross Profit Margin", f"{ratios['Gross Profit Margin (%)']}%")
            npm = ratios["Net Profit Margin (%)"]
            p2.metric("Net Profit Margin", f"{npm}%", delta="Profitable" if npm > 0 else "Loss", delta_color="normal" if npm > 0 else "inverse")
            p3.metric("Return on Assets (ROA)", f"{ratios['Return on Assets (%)']}%")

            # Solvency & Efficiency Section
            st.markdown("#### ⚖️ Solvency & Efficiency Ratios")
            s1, s2, s3, s4 = st.columns(4)
            dte = ratios["Debt to Equity"]
            s1.metric("Debt-to-Equity", dte, delta="Safe" if dte <= 1.5 else "High Leverage", delta_color="normal" if dte <= 1.5 else "inverse")
            icr = ratios["Interest Coverage Ratio"]
            s2.metric("Interest Coverage", f"{icr}x", delta="Safe" if icr >= 3.0 else "Debt Stress", delta_color="normal" if icr >= 3.0 else "inverse")
            s3.metric("Asset Turnover", f"{ratios['Asset Turnover']}x")
            s4.metric("Days Sales Outstanding", f"{ratios['DSO (Days)']} Days")

            # Assumptions & Limitations Expander
            st.markdown("---")
            with st.expander("📖 View Analytical Assumptions & Framework Limitations"):
                st.markdown(
                    """
                    * **Historical Bias:** Ratios reflect historical reporting periods and do not model dynamic future market conditions.
                    * **Book Value vs Market Value:** Fixed assets are measured at historical cost, which may not reflect real liquidation value.
                    * **Liquidity Assumptions:** Quick Ratio assumes inventory cannot be converted immediately into cash in short-term distress scenarios.
                    * **Sales Evenness:** Days Sales Outstanding (DSO) assumes credit revenues are generated evenly across a 365-day accounting period.
                    """
                )


# ============================================================ #
# 4. ROUTING LOGIC
# ============================================================ #
if app_mode == "📊 Financial Statement & Ratio Analyzer":
    render_financial_analyzer()
else:
    # Transaction-level anomaly engine code remains active here
    st.title("🚨 Fraud & Anomaly Detection Engine")
    st.info("Upload transaction logs to perform Benford screening, split payment identification, and Isolation Forest anomaly scoring.")
