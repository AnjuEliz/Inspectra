"""
INSPECTRA
=========
100% Financial Data Inspection & Intelligent Audit Analytics

Purpose
-------
Inspectra is an analytical support application for Chartered Accountants,
audit professionals and tax consultants.

It analyses the complete uploaded transaction population and identifies
transactions that deserve further investigation.

IMPORTANT
---------
A red flag is an investigation signal. It is NOT proof of fraud, tax
evasion, misconduct or an audit conclusion.

Supported files
---------------
CSV
Excel (.xlsx / .xls)
PDF tables

Recommended transaction columns
--------------------------------
Date
Transaction ID
Employee ID
Employee Name
Department
Bank Account
Amount
Vendor
Description

The application intentionally DOES NOT generate sample data.
The four sample CSV files should remain outside app.py.
"""

import io
import math
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
)


# ============================================================
# APPLICATION SETTINGS
# ============================================================

APP_NAME = "Inspectra"
VERSION = "4.0"

APPROVAL_LIMIT = 50000
MIN_RECORDS = 50

SPLIT_WINDOW_DAYS = 7
SPLIT_MIN_TRANSACTIONS = 2

ML_CONTAMINATION = 0.08


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Inspectra | Financial Anomaly Engine",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f6f8fb;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    .inspectra-title {
        font-size: 46px;
        line-height: 1.05;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 3px;
    }

    .inspectra-subtitle {
        font-size: 16px;
        color: #52606d;
        margin-bottom: 18px;
    }

    .hero {
        background: linear-gradient(
            135deg,
            #102a43 0%,
            #1f5f8b 100%
        );
        padding: 27px;
        border-radius: 17px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 25px rgba(16,42,67,.14);
    }

    .hero h2 {
        color: white;
        font-size: 28px;
        margin: 0 0 7px 0;
    }

    .hero p {
        color: #e6eef5;
        margin: 0;
        font-size: 15px;
    }

    .card {
        background: white;
        border: 1px solid #e1e8ef;
        border-radius: 13px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .danger-card {
        background: #fff4f4;
        border-left: 6px solid #dc2626;
        border-radius: 10px;
        padding: 15px;
    }

    .warning-card {
        background: #fffbeb;
        border-left: 6px solid #d97706;
        border-radius: 10px;
        padding: 15px;
    }

    .success-card {
        background: #ecfdf5;
        border-left: 6px solid #059669;
        border-radius: 10px;
        padding: 15px;
    }

    .info-card {
        background: #eff6ff;
        border-left: 6px solid #2563eb;
        border-radius: 10px;
        padding: 15px;
    }

    .section-heading {
        font-size: 25px;
        font-weight: 750;
        color: #102a43;
        margin-top: 18px;
        margin-bottom: 10px;
    }

    .tiny {
        font-size: 12px;
        color: #697586;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = None

if "benford" not in st.session_state:
    st.session_state.benford = None

if "filename" not in st.session_state:
    st.session_state.filename = ""

if "raw_data" not in st.session_state:
    st.session_state.raw_data = None


# ============================================================
# GENERAL HELPERS
# ============================================================

def money(value):
    try:
        if pd.isna(value):
            return "₹0.00"
        return f"₹{float(value):,.2f}"
    except Exception:
        return "₹0.00"


def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def normalise_text(value):
    return (
        safe_text(value)
        .strip()
        .lower()
    )


def first_digit(value):
    try:
        value = abs(float(value))

        if value <= 0:
            return None

        value_string = f"{value:.12f}"

        for character in value_string:
            if character.isdigit() and character != "0":
                return int(character)

    except Exception:
        pass

    return None


def format_date(value):
    if pd.isna(value):
        return ""
    try:
        return value.strftime("%d-%m-%Y")
    except Exception:
        return str(value)


# ============================================================
# COLUMN RECOGNITION
# ============================================================

COLUMN_ALIASES = {
    "transaction_id": [
        "transaction_id",
        "transactionid",
        "txn_id",
        "txn",
        "transaction_no",
        "transaction_number",
        "voucher_no",
        "voucher_number",
        "document_no",
        "document_number",
    ],
    "date": [
        "date",
        "transaction_date",
        "payment_date",
        "posting_date",
        "voucher_date",
    ],
    "employee_id": [
        "employee_id",
        "employeeid",
        "employee_code",
        "emp_id",
        "emp_code",
        "staff_id",
    ],
    "employee_name": [
        "employee_name",
        "employeename",
        "employee",
        "staff_name",
        "name",
    ],
    "department": [
        "department",
        "department_code",
        "dept",
        "dept_code",
    ],
    "bank_account": [
        "bank_account",
        "bank_account_no",
        "bank_account_number",
        "account_number",
        "account_no",
        "bank_ac_no",
    ],
    "amount": [
        "amount",
        "payment_amount",
        "transaction_amount",
        "value",
        "debit",
        "payment",
        "net_amount",
        "gross_amount",
    ],
    "vendor": [
        "vendor",
        "vendor_name",
        "supplier",
        "supplier_name",
        "payee",
    ],
    "description": [
        "description",
        "narration",
        "particulars",
        "remarks",
        "memo",
    ],
}


def clean_column_names(df):
    df = df.copy()

    df.columns = [
        re.sub(
            r"[^a-z0-9_]",
            "_",
            str(column).strip().lower(),
        ).strip("_")
        for column in df.columns
    ]

    rename_map = {}

    for standard_name, aliases in COLUMN_ALIASES.items():

        for column in df.columns:

            if column in aliases:
                rename_map[column] = standard_name
                break

    return df.rename(columns=rename_map)


# ============================================================
# FILE INPUT
# ============================================================

def read_csv_file(uploaded_file):
    """Read CSV safely from a Streamlit UploadedFile."""
    uploaded_file.seek(0)
    raw = uploaded_file.read()

    if not raw:
        raise ValueError("The uploaded CSV file is empty.")

    errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            return pd.read_csv(
                io.BytesIO(raw),
                encoding=encoding,
                sep=None,
                engine="python",
            )
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    raise ValueError(
        "Could not read the CSV file. "
        "Please check that it is a valid CSV. "
        + " | ".join(errors[:2])
    )


def read_excel_file(uploaded_file):
    """
    Read .xlsx/.xls from the Streamlit UploadedFile.

    .xlsx -> openpyxl
    .xls  -> xlrd

    The first non-empty worksheet is used automatically.
    """
    uploaded_file.seek(0)
    raw = uploaded_file.read()

    if not raw:
        raise ValueError("The uploaded Excel file is empty.")

    extension = os.path.splitext(uploaded_file.name.lower())[1]

    try:
        if extension == ".xlsx":
            # Load with openpyxl explicitly for modern Excel workbooks.
            workbook = pd.ExcelFile(
                io.BytesIO(raw),
                engine="openpyxl",
            )
        elif extension == ".xls":
            workbook = pd.ExcelFile(
                io.BytesIO(raw),
                engine="xlrd",
            )
        else:
            raise ValueError(
                "Unsupported Excel extension. Use .xlsx or .xls."
            )

    except ImportError as exc:
        if extension == ".xlsx":
            raise ValueError(
                "Excel .xlsx support is missing. "
                "Run: pip install openpyxl"
            ) from exc
        raise ValueError(
            "Excel .xls support is missing. "
            "Run: pip install xlrd"
        ) from exc

    except Exception as exc:
        raise ValueError(
            f"Inspectra could not open '{uploaded_file.name}'. "
            "The file may be corrupted, password-protected, "
            "or not a valid Excel workbook. "
            f"Technical detail: {exc}"
        ) from exc

    non_empty_sheets = []

    for sheet_name in workbook.sheet_names:
        try:
            sheet_df = pd.read_excel(
                io.BytesIO(raw),
                sheet_name=sheet_name,
                engine=(
                    "openpyxl"
                    if extension == ".xlsx"
                    else "xlrd"
                ),
            )
        except Exception as exc:
            continue

        if not sheet_df.empty and not sheet_df.dropna(how="all").empty:
            non_empty_sheets.append(
                (sheet_name, sheet_df)
            )

    if not non_empty_sheets:
        raise ValueError(
            "The Excel workbook was opened, but no non-empty worksheet "
            "with usable tabular data was found."
        )

    return non_empty_sheets[0][1]


def extract_pdf_table(uploaded_file):
    """
    Extract the best table from a text-based PDF.

    The function prefers structured PDF tables. If none are found,
    it falls back to simple delimited text lines.
    """
    try:
        import pdfplumber
    except ImportError as exc:
        raise ValueError(
            "PDF support requires pdfplumber. "
            "Run: pip install pdfplumber"
        ) from exc

    uploaded_file.seek(0)
    raw = uploaded_file.read()

    if not raw:
        raise ValueError("The uploaded PDF file is empty.")

    all_tables = []
    fallback_rows = []

    try:
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables() or []

                for table in tables:
                    cleaned = []
                    for row in table:
                        if row is None:
                            continue

                        values = [
                            "" if value is None else str(value).strip()
                            for value in row
                        ]

                        if any(values):
                            cleaned.append(values)

                    if len(cleaned) >= 2:
                        all_tables.append(cleaned)

                # Fallback for simple text-based PDFs.
                if not tables:
                    page_text = page.extract_text() or ""

                    for line in page_text.splitlines():
                        line = line.strip()

                        if not line:
                            continue

                        if "	" in line:
                            parts = [x.strip() for x in line.split("	")]
                        elif "|" in line:
                            parts = [x.strip() for x in line.split("|")]
                        else:
                            # Comma fallback only when it looks tabular.
                            parts = [x.strip() for x in line.split(",")]

                        if len(parts) >= 4:
                            fallback_rows.append(parts)

    except Exception as exc:
        raise ValueError(
            f"Inspectra could not read the PDF '{uploaded_file.name}'. "
            f"Technical detail: {exc}"
        ) from exc

    # Select the structured table with the most rows.
    if all_tables:
        table = max(all_tables, key=len)
        width = max(len(row) for row in table)
        table = [
            row + [""] * (width - len(row))
            for row in table
        ]

        # Use the first row as headers.
        headers = [
            safe_text(value).strip()
            for value in table[0]
        ]

        # If headers are mostly empty, don't treat the first data row as headers.
        if sum(bool(h) for h in headers) < max(2, width // 2):
            raise ValueError(
                "A PDF table was found, but its header row could not be identified."
            )

        return pd.DataFrame(
            table[1:],
            columns=headers,
        )

    if len(fallback_rows) >= 2:
        width = max(len(row) for row in fallback_rows)
        rows = [
            row + [""] * (width - len(row))
            for row in fallback_rows
        ]
        return pd.DataFrame(
            rows[1:],
            columns=rows[0],
        )

    raise ValueError(
        "No usable tabular data was found in the PDF. "
        "Text-based PDFs with selectable transaction tables are supported. "
        "Scanned/image-only PDFs require OCR before upload."
    )


def load_file(uploaded_file):
    extension = os.path.splitext(
        uploaded_file.name.lower()
    )[1]

    if extension == ".csv":
        df = read_csv_file(uploaded_file)

    elif extension in [".xlsx", ".xls"]:
        df = read_excel_file(uploaded_file)

    elif extension == ".pdf":
        df = extract_pdf_table(uploaded_file)

    else:
        raise ValueError(
            "Unsupported file format. "
            "Use CSV, Excel or PDF."
        )

    if df is None or df.empty:
        raise ValueError(
            "The uploaded file was read successfully, "
            "but it contains no transaction records."
        )

    return clean_column_names(df)


# ============================================================
# VALIDATION
# ============================================================

def validate_columns(df):

    required = [
        "date",
        "employee_id",
        "employee_name",
        "department",
        "bank_account",
        "amount",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    return missing


def prepare_data(df):

    df = df.copy()

    if "transaction_id" not in df.columns:
        df["transaction_id"] = [
            f"TXN-{index + 1:06d}"
            for index in range(len(df))
        ]

    for column in [
        "employee_id",
        "employee_name",
        "department",
        "bank_account",
        "vendor",
        "description",
    ]:

        if column not in df.columns:
            df[column] = ""

        df[column] = (
            df[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    df["amount"] = (
        df["amount"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("Rs.", "", regex=False)
        .str.strip()
    )

    df["amount"] = pd.to_numeric(
        df["amount"],
        errors="coerce",
    )

    df["transaction_number"] = (
        np.arange(
            1,
            len(df) + 1,
        )
    )

    return df


# ============================================================
# BENFORD'S LAW
# ============================================================

def benford_expected():
    return {
        digit: math.log10(
            1 + 1 / digit
        )
        for digit in range(1, 10)
    }


def calculate_benford(df):

    expected = benford_expected()

    valid_digits = [
        first_digit(amount)
        for amount in df["amount"]
    ]

    valid_digits = [
        digit
        for digit in valid_digits
        if digit is not None
    ]

    if len(valid_digits) < 20:

        return {
            "usable": False,
            "chi_square": None,
            "p_value": None,
            "expected": expected,
            "observed": {},
            "counts": {},
        }

    counts = {
        digit: valid_digits.count(digit)
        for digit in range(1, 10)
    }

    total = len(valid_digits)

    observed = {
        digit: counts[digit] / total
        for digit in range(1, 10)
    }

    chi_square = 0.0

    for digit in range(1, 10):

        expected_count = (
            total * expected[digit]
        )

        if expected_count > 0:

            chi_square += (
                (
                    counts[digit]
                    - expected_count
                ) ** 2
            ) / expected_count

    # Numerical approximation of chi-square survival
    # probability for 8 degrees of freedom.
    x = chi_square / 2

    p_value = math.exp(-x) * (
        1
        + x
        + x**2 / 2
        + x**3 / 6
    )

    return {
        "usable": True,
        "chi_square": chi_square,
        "p_value": p_value,
        "expected": expected,
        "observed": observed,
        "counts": counts,
    }


# ============================================================
# SPLIT PAYMENT ANALYSIS
# ============================================================

def detect_split_payment_trails(df):

    flags = {
        index: False
        for index in df.index
    }

    reasons = {
        index: []
        for index in df.index
    }

    trails = {
        index: []
        for index in df.index
    }

    working = df.copy()

    working["date_only"] = (
        working["date"].dt.normalize()
    )

    # We deliberately do NOT require the same employee only.
    # A real-world split-payment trail can also involve the
    # same vendor, account or department.
    grouping_options = [
        ["bank_account"],
        ["employee_id"],
        ["vendor"],
        ["department", "bank_account"],
    ]

    seen_groups = set()

    for group_columns in grouping_options:

        for group_key, group in working.groupby(
            group_columns,
            dropna=False,
        ):

            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            group_identifier = (
                tuple(group_columns),
                tuple(group_key),
            )

            if group_identifier in seen_groups:
                continue

            seen_groups.add(group_identifier)

            group = group.sort_values(
                "date"
            )

            if len(group) < SPLIT_MIN_TRANSACTIONS:
                continue

            for anchor_index in group.index:

                anchor_date = group.loc[
                    anchor_index,
                    "date",
                ]

                anchor_amount = group.loc[
                    anchor_index,
                    "amount",
                ]

                if pd.isna(anchor_date):
                    continue

                if pd.isna(anchor_amount):
                    continue

                if anchor_amount >= APPROVAL_LIMIT:
                    continue

                start_date = anchor_date
                end_date = (
                    anchor_date
                    + pd.Timedelta(
                        days=SPLIT_WINDOW_DAYS
                    )
                )

                window = group[
                    (
                        group["date"]
                        >= start_date
                    )
                    &
                    (
                        group["date"]
                        <= end_date
                    )
                    &
                    (
                        group["amount"]
                        < APPROVAL_LIMIT
                    )
                    &
                    (
                        group["amount"]
                        > 0
                    )
                ]

                if len(window) < 2:
                    continue

                total_value = window[
                    "amount"
                ].sum()

                if total_value < APPROVAL_LIMIT:
                    continue

                transaction_ids = (
                    window[
                        "transaction_id"
                    ]
                    .astype(str)
                    .tolist()
                )

                for index in window.index:

                    flags[index] = True

                    related = [
                        transaction_id
                        for transaction_id
                        in transaction_ids
                        if transaction_id
                        != str(
                            df.loc[
                                index,
                                "transaction_id",
                            ]
                        )
                    ]

                    trail_text = (
                        "Related transactions: "
                        + ", ".join(
                            transaction_ids
                        )
                    )

                    reason = (
                        "Possible split-payment pattern. "
                        "Multiple payments below the "
                        f"{money(APPROVAL_LIMIT)} approval "
                        "threshold were identified within "
                        f"{SPLIT_WINDOW_DAYS} days. "
                        f"The combined value is "
                        f"{money(total_value)}."
                    )

                    if group_columns == [
                        "bank_account"
                    ]:

                        reason += (
                            " The payments are linked through "
                            "the same bank account."
                        )

                    elif group_columns == [
                        "employee_id"
                    ]:

                        reason += (
                            " The payments are linked through "
                            "the same employee ID."
                        )

                    elif group_columns == [
                        "vendor"
                    ]:

                        reason += (
                            " The payments are linked through "
                            "the same vendor."
                        )

                    reasons[index].append(
                        reason
                    )

                    trails[index].extend(
                        related
                    )

                    trails[index].append(
                        trail_text
                    )

    for index in trails:
        trails[index] = list(
            dict.fromkeys(
                trails[index]
            )
        )

    return flags, reasons, trails


# ============================================================
# SHARED BANK ACCOUNT / GHOST PAYROLL ANALYSIS
# ============================================================

def detect_shared_accounts(df):

    flags = {
        index: False
        for index in df.index
    }

    reasons = {
        index: []
        for index in df.index
    }

    trails = {
        index: []
        for index in df.index
    }

    account_summary = (
        df.groupby(
            "bank_account",
            dropna=False,
        )
        .agg(
            employee_count=(
                "employee_id",
                "nunique",
            ),
            employee_names=(
                "employee_name",
                lambda values:
                ", ".join(
                    sorted(
                        set(
                            str(value)
                            for value in values
                            if str(value).strip()
                        )
                    )
                ),
            ),
            transaction_count=(
                "transaction_id",
                "count",
            ),
            total_amount=(
                "amount",
                "sum",
            ),
        )
        .reset_index()
    )

    suspicious = account_summary[
        (
            account_summary[
                "bank_account"
            ].astype(str).str.strip()
            != ""
        )
        &
        (
            account_summary[
                "employee_count"
            ]
            > 1
        )
    ]

    for _, account_row in suspicious.iterrows():

        account = account_row[
            "bank_account"
        ]

        employee_count = account_row[
            "employee_count"
        ]

        account_transactions = df[
            df["bank_account"]
            == account
        ]

        transaction_ids = (
            account_transactions[
                "transaction_id"
            ]
            .astype(str)
            .tolist()
        )

        employee_names = account_row[
            "employee_names"
        ]

        for index in account_transactions.index:

            flags[index] = True

            reasons[index].append(
                "Shared bank account warning. "
                f"This account is linked to "
                f"{employee_count} employee IDs "
                f"({employee_names}). "
                "This may be legitimate, but payroll "
                "master data and bank ownership should "
                "be verified."
            )

            trails[index].extend(
                [
                    transaction_id
                    for transaction_id
                    in transaction_ids
                    if transaction_id
                    != str(
                        df.loc[
                            index,
                            "transaction_id",
                        ]
                    )
                ]
            )

    return flags, reasons, trails, account_summary


# ============================================================
# DUPLICATE TRANSACTION ANALYSIS
# ============================================================

def detect_duplicates(df):

    flags = {
        index: False
        for index in df.index
    }

    reasons = {
        index: []
        for index in df.index
    }

    trails = {
        index: []
        for index in df.index
    }

    duplicate_columns = [
        "date",
        "employee_id",
        "bank_account",
        "amount",
        "department",
        "vendor",
    ]

    duplicate_columns = [
        column
        for column in duplicate_columns
        if column in df.columns
    ]

    if not duplicate_columns:
        return flags, reasons, trails

    duplicate_groups = df[
        df.duplicated(
            subset=duplicate_columns,
            keep=False,
        )
    ]

    for _, group in duplicate_groups.groupby(
        duplicate_columns,
        dropna=False,
    ):

        transaction_ids = (
            group[
                "transaction_id"
            ]
            .astype(str)
            .tolist()
        )

        for index in group.index:

            flags[index] = True

            others = [
                transaction_id
                for transaction_id
                in transaction_ids
                if transaction_id
                != str(
                    df.loc[
                        index,
                        "transaction_id",
                    ]
                )
            ]

            reasons[index].append(
                "Possible duplicate transaction. "
                "Another transaction contains the same "
                "key accounting characteristics: date, "
                "employee, bank account, amount and "
                "other matching fields."
            )

            trails[index].extend(
                others
            )

    return flags, reasons, trails


# ============================================================
# UNUSUAL AMOUNT ANALYSIS
# ============================================================

def detect_unusual_amounts(df):

    flags = {
        index: False
        for index in df.index
    }

    reasons = {
        index: []
        for index in df.index
    }

    amounts = (
        df["amount"]
        .dropna()
    )

    if len(amounts) < 10:
        return flags, reasons

    q1 = amounts.quantile(0.25)
    q3 = amounts.quantile(0.75)

    iqr = q3 - q1

    upper_limit = q3 + 3 * iqr

    for index, row in df.iterrows():

        amount = row["amount"]

        if (
            pd.notna(amount)
            and amount > upper_limit
        ):

            flags[index] = True

            reasons[index].append(
                "Unusually large transaction. "
                f"The amount {money(amount)} is "
                "substantially above the normal transaction "
                "range in this dataset. Verify the invoice, "
                "approval and business purpose."
            )

    return flags, reasons


# ============================================================
# BENFORD TRANSACTION SCREENING
# ============================================================

def detect_benford_indicators(
    df,
    benford,
):

    flags = {
        index: False
        for index in df.index
    }

    reasons = {
        index: []
        for index in df.index
    }

    if not benford["usable"]:
        return flags, reasons

    if benford["p_value"] >= 0.05:
        return flags, reasons

    for index, row in df.iterrows():

        digit = first_digit(
            row["amount"]
        )

        if digit is None:
            continue

        observed = benford[
            "observed"
        ][digit]

        expected = benford[
            "expected"
        ][digit]

        difference = abs(
            observed - expected
        )

        if difference >= 0.03:

            flags[index] = True

            reasons[index].append(
                "Benford review indicator. "
                "The first digit belongs to a digit group "
                "that contributes to an unusual overall "
                "first-digit distribution. This is a "
                "screening signal only and does not prove "
                "that this transaction is fraudulent."
            )

    return flags, reasons


# ============================================================
# MACHINE LEARNING
# ============================================================

def create_ml_features(df):

    features = pd.DataFrame(
        index=df.index
    )

    features["amount"] = (
        pd.to_numeric(
            df["amount"],
            errors="coerce",
        )
        .fillna(0)
    )

    features["log_amount"] = np.log1p(
        features["amount"].clip(
            lower=0
        )
    )

    features["department_frequency"] = (
        df["department"]
        .map(
            df[
                "department"
            ].value_counts()
        )
        .fillna(0)
    )

    features["employee_frequency"] = (
        df["employee_id"]
        .map(
            df[
                "employee_id"
            ].value_counts()
        )
        .fillna(0)
    )

    features["account_frequency"] = (
        df["bank_account"]
        .map(
            df[
                "bank_account"
            ].value_counts()
        )
        .fillna(0)
    )

    features["vendor_frequency"] = (
        df["vendor"]
        .map(
            df[
                "vendor"
            ].value_counts()
        )
        .fillna(0)
    )

    features["employee_account_count"] = (
        df["bank_account"]
        .map(
            df.groupby(
                "bank_account"
            )[
                "employee_id"
            ].nunique()
        )
        .fillna(0)
    )

    features["department_account_count"] = (
        df["bank_account"]
        .map(
            df.groupby(
                "bank_account"
            )[
                "department"
            ].nunique()
        )
        .fillna(0)
    )

    features["day_of_week"] = (
        df["date"]
        .dt.dayofweek
        .fillna(0)
    )

    features["day_of_month"] = (
        df["date"]
        .dt.day
        .fillna(0)
    )

    features["month"] = (
        df["date"]
        .dt.month
        .fillna(0)
    )

    features["amount_below_approval"] = (
        (
            features["amount"]
            < APPROVAL_LIMIT
        )
        .astype(int)
    )

    features["amount_near_approval"] = (
        (
            (
                features["amount"]
                >= APPROVAL_LIMIT * 0.85
            )
            &
            (
                features["amount"]
                < APPROVAL_LIMIT
            )
        )
        .astype(int)
    )

    return features.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0)


def run_machine_learning(df):

    features = create_ml_features(
        df
    )

    if len(features) < 10:

        return pd.DataFrame(
            {
                "ml_anomaly": False,
                "ml_score": 0.0,
            },
            index=df.index,
        )

    scaler = StandardScaler()

    X = scaler.fit_transform(
        features
    )

    model = IsolationForest(
        n_estimators=300,
        contamination=ML_CONTAMINATION,
        random_state=42,
        n_jobs=-1,
    )

    prediction = model.fit_predict(
        X
    )

    raw_score = (
        -model.decision_function(X)
    )

    minimum = raw_score.min()
    maximum = raw_score.max()

    if maximum > minimum:

        score = (
            (
                raw_score
                - minimum
            )
            /
            (
                maximum
                - minimum
            )
            * 100
        )

    else:

        score = np.zeros(
            len(raw_score)
        )

    return pd.DataFrame(
        {
            "ml_anomaly":
                prediction == -1,
            "ml_score":
                np.round(
                    score,
                    1,
                ),
        },
        index=df.index,
    )


# ============================================================
# INVESTIGATION TRAIL
# ============================================================

def build_investigation_trail(
    df,
    index,
    split_trails,
    account_trails,
    duplicate_trails,
):

    current = df.loc[
        index
    ]

    related_ids = []

    related_ids.extend(
        split_trails.get(
            index,
            [],
        )
    )

    related_ids.extend(
        account_trails.get(
            index,
            [],
        )
    )

    related_ids.extend(
        duplicate_trails.get(
            index,
            [],
        )
    )

    current_id = str(
        current[
            "transaction_id"
        ]
    )

    related_ids = [
        str(transaction_id)
        for transaction_id
        in related_ids
        if str(transaction_id)
        != current_id
    ]

    related_ids = list(
        dict.fromkeys(
            related_ids
        )
    )

    if not related_ids:
        return pd.DataFrame(
            columns=df.columns
        )

    return df[
        df["transaction_id"]
        .astype(str)
        .isin(
            related_ids
        )
    ].copy()


# ============================================================
# MASTER ANALYSIS
# ============================================================

def analyze_data(df):

    df = prepare_data(df)

    benford = calculate_benford(
        df
    )

    (
        split_flags,
        split_reasons,
        split_trails,
    ) = detect_split_payment_trails(
        df
    )

    (
        account_flags,
        account_reasons,
        account_trails,
        account_summary,
    ) = detect_shared_accounts(
        df
    )

    (
        duplicate_flags,
        duplicate_reasons,
        duplicate_trails,
    ) = detect_duplicates(
        df
    )

    (
        unusual_flags,
        unusual_reasons,
    ) = detect_unusual_amounts(
        df
    )

    (
        benford_flags,
        benford_reasons,
    ) = detect_benford_indicators(
        df,
        benford,
    )

    ml_results = run_machine_learning(
        df
    )

    results = df.copy()

    results["Split Payment"] = [
        split_flags[index]
        for index in results.index
    ]

    results["Shared Bank Account"] = [
        account_flags[index]
        for index in results.index
    ]

    results["Possible Duplicate"] = [
        duplicate_flags[index]
        for index in results.index
    ]

    results["Unusual Amount"] = [
        unusual_flags[index]
        for index in results.index
    ]

    results["Benford Review"] = [
        benford_flags[index]
        for index in results.index
    ]

    results["ML Anomaly"] = (
        ml_results[
            "ml_anomaly"
        ]
    )

    results["ML Score"] = (
        ml_results[
            "ml_score"
        ]
    )

    reasons = []
    scores = []
    levels = []

    for index, row in results.iterrows():

        row_reasons = []

        score = 0

        row_reasons.extend(
            split_reasons[index]
        )

        row_reasons.extend(
            account_reasons[index]
        )

        row_reasons.extend(
            duplicate_reasons[index]
        )

        row_reasons.extend(
            unusual_reasons[index]
        )

        row_reasons.extend(
            benford_reasons[index]
        )

        if row["ML Anomaly"]:

            row_reasons.append(
                "Machine-learning review indicator. "
                "The combination of transaction amount, "
                "account usage, employee activity, department "
                "behaviour, vendor activity and transaction "
                "timing is unusual compared with the rest "
                "of this population."
            )

        # Evidence-weighted risk score.
        if row["Split Payment"]:
            score += 35

        if row["Shared Bank Account"]:
            score += 35

        if row["Possible Duplicate"]:
            score += 25

        if row["Unusual Amount"]:
            score += 15

        if row["Benford Review"]:
            score += 10

        if row["ML Anomaly"]:
            score += 20

        score = min(
            score,
            100,
        )

        if score >= 60:
            level = "High"

        elif score >= 30:
            level = "Medium"

        elif score > 0:
            level = "Low"

        else:
            level = "Normal"

        if not row_reasons:

            row_reasons = [
                "No major anomaly was identified "
                "by the current Inspectra tests."
            ]

        # Remove repeated reasons while preserving order.
        row_reasons = list(
            dict.fromkeys(
                row_reasons
            )
        )

        reasons.append(
            " ".join(
                row_reasons
            )
        )

        scores.append(
            score
        )

        levels.append(
            level
        )

    results["Risk Score"] = scores

    results["Risk Level"] = levels

    results["Red Flag Reason"] = reasons

    results["Investigation Trail"] = [
        build_investigation_trail(
            results,
            index,
            split_trails,
            account_trails,
            duplicate_trails,
        )[
            "transaction_id"
        ]
        .astype(str)
        .tolist()
        if not build_investigation_trail(
            results,
            index,
            split_trails,
            account_trails,
            duplicate_trails,
        ).empty
        else []
        for index in results.index
    ]

    results["Potential Cash Impact"] = np.where(
        results["Risk Level"] == "Normal",
        0,
        results["amount"].fillna(0),
    )

    return (
        results,
        benford,
        account_summary,
    )


# ============================================================
# EXCEL WORKPAPER
# ============================================================

def create_excel_report(
    results,
    benford,
    account_summary,
    filename,
):

    output = io.BytesIO()

    red_flags = results[
        results["Risk Level"]
        != "Normal"
    ].copy()

    summary = pd.DataFrame(
        {
            "Metric": [
                "Source file",
                "Report generated",
                "Total transactions inspected",
                "Transactions requiring review",
                "High-risk transactions",
                "Medium-risk transactions",
                "Low-risk transactions",
                "Total uploaded transaction value",
                "Potential flagged cash exposure",
                "Split-payment flags",
                "Shared-account flags",
                "Duplicate flags",
                "Unusual amount flags",
                "Benford review indicators",
                "Machine-learning anomalies",
            ],
            "Value": [
                filename,
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                ),
                len(results),
                int(
                    (
                        results["Risk Level"]
                        != "Normal"
                    ).sum()
                ),
                int(
                    (
                        results["Risk Level"]
                        == "High"
                    ).sum()
                ),
                int(
                    (
                        results["Risk Level"]
                        == "Medium"
                    ).sum()
                ),
                int(
                    (
                        results["Risk Level"]
                        == "Low"
                    ).sum()
                ),
                money(
                    results["amount"].sum()
                ),
                money(
                    results[
                        "Potential Cash Impact"
                    ].sum()
                ),
                int(
                    results[
                        "Split Payment"
                    ].sum()
                ),
                int(
                    results[
                        "Shared Bank Account"
                    ].sum()
                ),
                int(
                    results[
                        "Possible Duplicate"
                    ].sum()
                ),
                int(
                    results[
                        "Unusual Amount"
                    ].sum()
                ),
                int(
                    results[
                        "Benford Review"
                    ].sum()
                ),
                int(
                    results[
                        "ML Anomaly"
                    ].sum()
                ),
            ],
        }
    )

    explanations = pd.DataFrame(
        {
            "Term": [
                "100% inspection",
                "Red flag",
                "Split payment",
                "Investigation trail",
                "Shared bank account",
                "Ghost employee warning",
                "Duplicate transaction",
                "Benford's Law",
                "Machine learning",
                "Isolation Forest",
                "Risk score",
                "Working capital",
                "Potential cash exposure",
            ],
            "Simple explanation": [
                "Inspectra evaluates every uploaded transaction.",
                "A warning that tells the professional that a transaction deserves further review.",
                "Several smaller payments may be used instead of one larger payment.",
                "The connected transactions that help the auditor trace the pattern.",
                "The same bank account appears against more than one employee record.",
                "A shared or unusual employee-bank relationship that should be verified before drawing a conclusion.",
                "Two or more records appear to contain the same important transaction details.",
                "A mathematical first-digit pattern used as a screening test.",
                "A computer technique that considers several characteristics together.",
                "The machine-learning method used to identify records that look unusual.",
                "A 0-100 indicator representing the combined strength of warning signals.",
                "Money tied up in the normal operating cycle of a business.",
                "The value of transactions currently marked for investigation.",
            ],
        }
    )

    benford_rows = []

    if benford["usable"]:

        for digit in range(1, 10):

            benford_rows.append(
                {
                    "First Digit": digit,
                    "Observed %": round(
                        benford[
                            "observed"
                        ][digit]
                        * 100,
                        2,
                    ),
                    "Expected %": round(
                        benford[
                            "expected"
                        ][digit]
                        * 100,
                        2,
                    ),
                }
            )

    trail_rows = []

    for _, row in red_flags.iterrows():

        trail_rows.append(
            {
                "Transaction ID":
                    row[
                        "transaction_id"
                    ],
                "Risk Level":
                    row[
                        "Risk Level"
                    ],
                "Risk Score":
                    row[
                        "Risk Score"
                    ],
                "Reason":
                    row[
                        "Red Flag Reason"
                    ],
                "Related Transaction IDs":
                    ", ".join(
                        row[
                            "Investigation Trail"
                        ]
                    ),
            }
        )

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        summary.to_excel(
            writer,
            sheet_name="Executive Summary",
            index=False,
        )

        red_flags.to_excel(
            writer,
            sheet_name="Red Flag Transactions",
            index=False,
        )

        results.to_excel(
            writer,
            sheet_name="All Transactions",
            index=False,
        )

        pd.DataFrame(
            trail_rows
        ).to_excel(
            writer,
            sheet_name="Investigation Trails",
            index=False,
        )

        account_summary.to_excel(
            writer,
            sheet_name="Account Analysis",
            index=False,
        )

        explanations.to_excel(
            writer,
            sheet_name="Simple Explanations",
            index=False,
        )

        pd.DataFrame(
            benford_rows
        ).to_excel(
            writer,
            sheet_name="Benford Analysis",
            index=False,
        )

        workbook = writer.book

        for worksheet in workbook.worksheets:

            worksheet.freeze_panes = "A2"

            for column_cells in worksheet.columns:

                max_length = 0

                column_letter = (
                    column_cells[
                        0
                    ].column_letter
                )

                for cell in column_cells:

                    try:
                        max_length = max(
                            max_length,
                            len(
                                str(
                                    cell.value
                                )
                            ),
                        )
                    except Exception:
                        pass

                worksheet.column_dimensions[
                    column_letter
                ].width = min(
                    max(
                        max_length + 2,
                        12,
                    ),
                    55,
                )

    output.seek(0)

    return output.getvalue()


# ============================================================
# PDF WORKPAPER
# ============================================================

def create_pdf_report(
    results,
    benford,
    filename,
):

    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=landscape(A4),
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "InspectraTitle",
        parent=styles["Title"],
        fontSize=22,
        alignment=TA_CENTER,
        textColor=colors.HexColor(
            "#102a43"
        ),
        spaceAfter=8,
    )

    heading_style = ParagraphStyle(
        "InspectraHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=colors.HexColor(
            "#102a43"
        ),
        spaceBefore=9,
        spaceAfter=7,
    )

    body_style = ParagraphStyle(
        "InspectraBody",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
    )

    small_style = ParagraphStyle(
        "InspectraSmall",
        parent=styles["BodyText"],
        fontSize=6.8,
        leading=9,
    )

    story = []

    story.append(
        Paragraph(
            "Inspectra — Financial Anomaly Review Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            f"Source file: {safe_text(filename)} "
            f"• Generated: "
            f"{datetime.now().strftime('%d-%m-%Y %H:%M')}",
            body_style,
        )
    )

    story.append(
        Spacer(1, 7)
    )

    red_flags = results[
        results["Risk Level"]
        != "Normal"
    ]

    summary_data = [
        [
            "Metric",
            "Result",
        ],
        [
            "Transactions inspected",
            str(len(results)),
        ],
        [
            "Transactions requiring review",
            str(len(red_flags)),
        ],
        [
            "High risk",
            str(
                (
                    results["Risk Level"]
                    == "High"
                ).sum()
            ),
        ],
        [
            "Medium risk",
            str(
                (
                    results["Risk Level"]
                    == "Medium"
                ).sum()
            ),
        ],
        [
            "Low risk",
            str(
                (
                    results["Risk Level"]
                    == "Low"
                ).sum()
            ),
        ],
        [
            "Total transaction value",
            money(
                results["amount"].sum()
            ),
        ],
        [
            "Potential flagged cash exposure",
            money(
                results[
                    "Potential Cash Impact"
                ].sum()
            ),
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            75 * mm,
            55 * mm,
        ],
    )

    summary_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#102a43"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold",
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor(
                        "#d9e2ec"
                    ),
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(
        Paragraph(
            "Executive Summary",
            heading_style,
        )
    )

    story.append(
        summary_table
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "Important interpretation",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            "Inspectra evaluates the complete uploaded population "
            "rather than relying only on random sampling. A red "
            "flag is an investigation signal, not proof of fraud "
            "or another wrongdoing. The professional should "
            "trace the transaction to invoices, approvals, bank "
            "records, payroll records and other appropriate "
            "supporting evidence before reaching a conclusion.",
            body_style,
        )
    )

    story.append(
        Paragraph(
            "Red Flag Transactions",
            heading_style,
        )
    )

    if red_flags.empty:

        story.append(
            Paragraph(
                "No transactions were flagged by the current "
                "inspection engines.",
                body_style,
            )
        )

    else:

        table_data = [
            [
                "Txn",
                "Date",
                "Employee",
                "Amount",
                "Risk",
                "Score",
                "Reason",
                "Investigation Trail",
            ]
        ]

        for _, row in red_flags.iterrows():

            reason = safe_text(
                row[
                    "Red Flag Reason"
                ]
            )

            trail = ", ".join(
                row[
                    "Investigation Trail"
                ]
            )

            if len(reason) > 400:
                reason = (
                    reason[:397]
                    + "..."
                )

            if len(trail) > 220:
                trail = (
                    trail[:217]
                    + "..."
                )

            table_data.append(
                [
                    safe_text(
                        row[
                            "transaction_id"
                        ]
                    ),
                    format_date(
                        row["date"]
                    ),
                    safe_text(
                        row[
                            "employee_name"
                        ]
                    ),
                    money(
                        row["amount"]
                    ),
                    safe_text(
                        row["Risk Level"]
                    ),
                    str(
                        row[
                            "Risk Score"
                        ]
                    ),
                    Paragraph(
                        reason,
                        small_style,
                    ),
                    Paragraph(
                        trail,
                        small_style,
                    ),
                ]
            )

        table = Table(
            table_data,
            repeatRows=1,
            colWidths=[
                24 * mm,
                21 * mm,
                32 * mm,
                25 * mm,
                17 * mm,
                15 * mm,
                95 * mm,
                55 * mm,
            ],
        )

        table.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor(
                            "#102a43"
                        ),
                    ),
                    (
                        "TEXTCOLOR",
                        (0, 0),
                        (-1, 0),
                        colors.white,
                    ),
                    (
                        "FONTNAME",
                        (0, 0),
                        (-1, 0),
                        "Helvetica-Bold",
                    ),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.25,
                        colors.HexColor(
                            "#d9e2ec"
                        ),
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [
                            colors.white,
                            colors.HexColor(
                                "#f8fafc"
                            ),
                        ],
                    ),
                    (
                        "PADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                ]
            )
        )

        story.append(
            table
        )

    story.append(
        PageBreak()
    )

    story.append(
        Paragraph(
            "Simple Explanation of Audit Analytics",
            heading_style,
        )
    )

    terms = [
        (
            "100% inspection",
            "Every uploaded transaction is evaluated."
        ),
        (
            "Red flag",
            "A warning that tells the auditor that a transaction deserves investigation."
        ),
        (
            "Split payment",
            "Several smaller payments may be related and could have been used instead of one larger payment."
        ),
        (
            "Investigation trail",
            "Related transactions shown together so the professional can trace the activity."
        ),
        (
            "Shared bank account",
            "The same account is associated with more than one employee record."
        ),
        (
            "Duplicate",
            "Two or more transactions appear to contain the same key accounting details."
        ),
        (
            "Benford's Law",
            "A mathematical first-digit pattern used as a screening test."
        ),
        (
            "Machine learning",
            "A computer method that considers several characteristics together to identify unusual records."
        ),
        (
            "Isolation Forest",
            "The machine-learning method used by Inspectra to identify records that appear unusual."
        ),
        (
            "Risk score",
            "A 0-100 indicator based on the number and strength of warning signals."
        ),
        (
            "Potential cash exposure",
            "The value represented by transactions currently marked for investigation."
        ),
    ]

    terms_table = [
        [
            Paragraph("<b>Term</b>", body_style),
            Paragraph("<b>Simple meaning</b>", body_style),
        ]
    ]

    for term, meaning in terms:

        terms_table.append(
            [
                Paragraph(
                    f"<b>{term}</b>",
                    body_style,
                ),
                Paragraph(
                    meaning,
                    body_style,
                ),
            ]
        )

    table = Table(
        terms_table,
        colWidths=[
            50 * mm,
            190 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.HexColor(
                        "#102a43"
                    ),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    colors.white,
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.25,
                    colors.HexColor(
                        "#d9e2ec"
                    ),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [
                        colors.white,
                        colors.HexColor(
                            "#f8fafc"
                        ),
                    ],
                ),
                (
                    "PADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(
        table
    )

    story.append(
        Spacer(1, 15)
    )

    story.append(
        Paragraph(
            "Professional-use note",
            heading_style,
        )
    )

    story.append(
        Paragraph(
            "Inspectra is an analytical support system. "
            "Its output should be independently verified by "
            "the responsible CA, auditor or tax professional. "
            "The system does not itself establish fraud, tax "
            "evasion, misconduct or any legal conclusion.",
            body_style,
        )
    )

    document.build(
        story
    )

    output.seek(0)

    return output.getvalue()



# ============================================================
# WORKING-CAPITAL DIAGNOSTICS
# ============================================================

def _find_numeric_column(df, aliases):
    """Find the first matching financial-base column, if supplied."""
    normalised = {
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c
        for c in df.columns
    }
    for alias in aliases:
        key = re.sub(r"[^a-z0-9]+", "_", alias.strip().lower()).strip("_")
        if key in normalised:
            return normalised[key]
    return None


def build_working_capital_diagnostics(results):
    """
    Turn flagged transactions into actionable liquidity/exposure diagnostics.

    Optional source columns:
    - Accounts Payable / AP
    - Accounts Receivable / AR
    - Net Working Capital / NWC
    - Customer (optional, for concentration analysis)
    - Transaction Type / Flow Type (optional)
    - Due Date (optional, for receivable/payable ageing)
    """

    data = results.copy()

    data["amount"] = pd.to_numeric(
        data.get("amount", 0), errors="coerce"
    ).fillna(0.0)

    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    else:
        data["date"] = pd.NaT

    flagged = data[
        data["Risk Level"].astype(str).str.strip().str.casefold() != "normal"
    ].copy()

    flagged_value = float(flagged["amount"].sum())

    # Optional financial bases supplied in the original upload.
    ap_col = _find_numeric_column(
        data,
        ["accounts payable", "accounts_payable", "active ap", "ap", "payables"],
    )
    ar_col = _find_numeric_column(
        data,
        ["accounts receivable", "accounts_receivable", "ar", "receivables"],
    )
    nwc_col = _find_numeric_column(
        data,
        ["net working capital", "net_working_capital", "nwc"],
    )

    def total_base(col):
        if not col:
            return 0.0
        return float(
            pd.to_numeric(data[col], errors="coerce").dropna().sum()
        )

    ap_value = total_base(ap_col)
    ar_value = total_base(ar_col)
    nwc_value = total_base(nwc_col)

    # Preference: AP -> AR -> NWC when supplied. Otherwise use transaction value
    # explicitly as a proxy, never as "actual working capital".
    if ap_value > 0:
        wc_base = ap_value
        wc_label = "Accounts Payable"
    elif ar_value > 0:
        wc_base = ar_value
        wc_label = "Accounts Receivable"
    elif abs(nwc_value) > 0:
        wc_base = abs(nwc_value)
        wc_label = "Net Working Capital"
    else:
        wc_base = float(abs(data["amount"].sum()))
        wc_label = "Uploaded transaction value (proxy)"

    at_risk_pct = (
        flagged_value / wc_base * 100
        if wc_base > 0 else 0.0
    )

    # Optional customer column for concentration.
    customer_col = _find_numeric_column(data, [])  # intentionally no numeric lookup
    for candidate in [
        "customer", "customer_name", "customer_id", "client", "client_name"
    ]:
        if candidate in {
            re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
            for c in data.columns
        }:
            customer_col = next(
                c for c in data.columns
                if re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
                == candidate
            )
            break

    # Flow classification.
    flow_aliases = [
        "flow type", "flow_type", "transaction type", "transaction_type",
        "type", "cash flow type", "cash_flow_type"
    ]
    flow_col = None
    normalised_columns = {
        re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_"): c
        for c in data.columns
    }
    for alias in flow_aliases:
        key = re.sub(r"[^a-z0-9]+", "_", alias).strip("_")
        if key in normalised_columns:
            flow_col = normalised_columns[key]
            break

    def classify_flow(row):
        if flow_col:
            supplied = safe_text(row.get(flow_col, "")).casefold()
            if any(word in supplied for word in ["inflow", "receipt", "receivable", "collection", "credit"]):
                return "Inflow Risk"
            if any(word in supplied for word in ["outflow", "payment", "payable", "vendor", "debit"]):
                return "Outflow Risk"

        description = safe_text(row.get("description", "")).casefold()
        vendor = safe_text(row.get("vendor", "")).casefold()
        customer = safe_text(row.get(customer_col, "")) if customer_col else ""
        text_blob = f"{description} {vendor} {customer}".casefold()

        inflow_terms = [
            "receivable", "customer collection", "collection due",
            "credit memo", "refund received", "sales receipt",
            "customer payment", "cash receipt"
        ]
        if any(term in text_blob for term in inflow_terms):
            return "Inflow Risk"

        return "Outflow Risk"

    if flagged.empty:
        flagged["Flow Type"] = pd.Series(index=flagged.index, dtype=str)
    else:
        flagged["Flow Type"] = flagged.apply(classify_flow, axis=1)

    outflow_value = float(
        flagged.loc[
            flagged["Flow Type"] == "Outflow Risk", "amount"
        ].sum()
    )
    inflow_value = float(
        flagged.loc[
            flagged["Flow Type"] == "Inflow Risk", "amount"
        ].sum()
    )

    # Ageing basis: use Due Date where supplied; otherwise transaction Date.
    due_col = None
    for alias in ["due date", "due_date", "payment due date", "payment_due_date"]:
        key = re.sub(r"[^a-z0-9]+", "_", alias).strip("_")
        if key in normalised_columns:
            due_col = normalised_columns[key]
            break

    if due_col:
        flagged["_age_reference"] = pd.to_datetime(
            flagged[due_col], errors="coerce"
        )
    else:
        flagged["_age_reference"] = flagged["date"]

    valid_dates = data["date"].dropna()
    as_of = (
        valid_dates.max()
        if not valid_dates.empty
        else pd.Timestamp.today().normalize()
    )

    def age_days(value):
        if pd.isna(value):
            return np.nan
        return max(0, (as_of.normalize() - value.normalize()).days)

    if not flagged.empty:
        flagged["Age Days"] = flagged["_age_reference"].apply(age_days)

        def bucket(days):
            if pd.isna(days):
                return "Date unavailable"
            if days <= 30:
                return "0–30 days"
            if days <= 60:
                return "31–60 days"
            if days <= 90:
                return "61–90 days"
            return "90+ days"

        flagged["Ageing Bucket"] = flagged["Age Days"].apply(bucket)
    else:
        flagged["Age Days"] = pd.Series(dtype=float)
        flagged["Ageing Bucket"] = pd.Series(dtype=str)

    # Amount-weighted average age: a transparent operational indicator.
    total_flagged = float(flagged["amount"].sum())
    estimated_cash_frozen_days = (
        float(
            (flagged["amount"] * flagged["Age Days"].fillna(0)).sum()
            / total_flagged
        )
        if total_flagged > 0 else 0.0
    )

    ageing_order = [
        "0–30 days", "31–60 days", "61–90 days",
        "90+ days", "Date unavailable"
    ]
    ageing_matrix = (
        flagged.groupby("Ageing Bucket", dropna=False)
        .agg(
            Transactions=("transaction_id", "count"),
            Flagged_Value=("amount", "sum"),
        )
        .reindex(ageing_order, fill_value=0)
        .reset_index()
    ) if not flagged.empty else pd.DataFrame({
        "Ageing Bucket": ageing_order,
        "Transactions": [0] * len(ageing_order),
        "Flagged_Value": [0.0] * len(ageing_order),
    })

    # Risk-weighted ageing view: separate high/critical amounts.
    high_risk_mask = flagged["Risk Level"].astype(str).isin(["Critical", "High"]) if not flagged.empty else pd.Series(dtype=bool)
    risk_aging = (
        flagged.assign(
            High_or_Critical=np.where(high_risk_mask, "High/Critical", "Medium/Low")
        )
        .groupby(["Ageing Bucket", "High_or_Critical"], dropna=False)
        .agg(Flagged_Value=("amount", "sum"))
        .reset_index()
        if not flagged.empty else pd.DataFrame(
            columns=["Ageing Bucket", "High_or_Critical", "Flagged_Value"]
        )
    )

    # Weekly velocity.
    velocity_df = pd.DataFrame(
        columns=["Week", "Flagged_Transactions", "Flagged_Value"]
    )
    velocity_pct = None
    if not flagged.empty and flagged["date"].notna().any():
        velocity_df = (
            flagged.dropna(subset=["date"])
            .assign(Week=lambda x: x["date"].dt.to_period("W").astype(str))
            .groupby("Week")
            .agg(
                Flagged_Transactions=("transaction_id", "count"),
                Flagged_Value=("amount", "sum"),
            )
            .reset_index()
        )
        if len(velocity_df) >= 2:
            previous = float(velocity_df.iloc[-2]["Flagged_Value"])
            current = float(velocity_df.iloc[-1]["Flagged_Value"])
            if previous != 0:
                velocity_pct = (current - previous) / previous * 100

    # Top 10.
    top10 = (
        flagged.sort_values(
            ["Risk Score", "amount"],
            ascending=[False, False]
        ).head(10).copy()
        if not flagged.empty else flagged.copy()
    )

    # Vendor concentration.
    if not flagged.empty:
        vendor_values = (
            flagged["vendor"]
            if "vendor" in flagged.columns
            else pd.Series("", index=flagged.index)
        )
        vendor_concentration = (
            flagged.assign(
                Vendor=vendor_values.replace("", "Unknown / Not supplied").fillna("Unknown / Not supplied")
            )
            .groupby("Vendor", dropna=False)
            .agg(
                Flagged_Transactions=("transaction_id", "count"),
                Flagged_Value=("amount", "sum"),
            )
            .sort_values("Flagged_Value", ascending=False)
            .reset_index()
        )

        if customer_col:
            customer_concentration = (
                flagged.assign(
                    Customer=flagged[customer_col]
                    .replace("", "Unknown / Not supplied")
                    .fillna("Unknown / Not supplied")
                )
                .groupby("Customer", dropna=False)
                .agg(
                    Flagged_Transactions=("transaction_id", "count"),
                    Flagged_Value=("amount", "sum"),
                )
                .sort_values("Flagged_Value", ascending=False)
                .reset_index()
            )
        else:
            customer_concentration = pd.DataFrame(
                columns=["Customer", "Flagged_Transactions", "Flagged_Value"]
            )
    else:
        vendor_concentration = pd.DataFrame(
            columns=["Vendor", "Flagged_Transactions", "Flagged_Value"]
        )
        customer_concentration = pd.DataFrame(
            columns=["Customer", "Flagged_Transactions", "Flagged_Value"]
        )

    # DSO/DPO detour:
    # If actual receivable/payable balances are available, estimate days of
    # flagged cash represented relative to the relevant daily operating base.
    # Otherwise, clearly label as unavailable rather than inventing a ratio.
    dso_days = None
    dpo_days = None

    if ar_value > 0 and inflow_value > 0:
        # This is an exposure-to-balance estimate, not actual DSO.
        dso_days = inflow_value / ar_value * 30

    if ap_value > 0 and outflow_value > 0:
        # This is an exposure-to-balance estimate, not actual DPO.
        dpo_days = outflow_value / ap_value * 30

    # Hold list / assignment list.
    risk_priority = {"Critical": 5, "High": 4, "Medium": 3, "Low": 2, "Normal": 1}
    hold_list = flagged.copy()
    if not hold_list.empty:
        hold_list["_priority"] = hold_list["Risk Level"].map(risk_priority).fillna(0)
        hold_list = (
            hold_list.sort_values(
                ["_priority", "amount"],
                ascending=[False, False]
            )
            .drop(columns="_priority")
        )

    return {
        "flagged": flagged.drop(columns=["_age_reference"], errors="ignore"),
        "flagged_value": flagged_value,
        "flagged_count": len(flagged),
        "working_capital_base": wc_base,
        "working_capital_label": wc_label,
        "working_capital_ratio": at_risk_pct,
        "ap_value": ap_value,
        "ar_value": ar_value,
        "nwc_value": nwc_value,
        "outflow_value": outflow_value,
        "inflow_value": inflow_value,
        "estimated_cash_frozen_days": estimated_cash_frozen_days,
        "ageing_matrix": ageing_matrix,
        "risk_aging": risk_aging,
        "velocity_df": velocity_df,
        "velocity_pct": velocity_pct,
        "top10": top10,
        "vendor_concentration": vendor_concentration,
        "customer_concentration": customer_concentration,
        "dso_detour_days": dso_days,
        "dpo_detour_days": dpo_days,
        "hold_list": hold_list.drop(columns=["_age_reference"], errors="ignore"),
        "as_of": as_of,
    }


def create_wc_hold_list_excel(diagnostics):
    output = io.BytesIO()

    hold_list = diagnostics["hold_list"].copy()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        hold_list.to_excel(
            writer,
            sheet_name="Payment Hold List",
            index=False,
        )

        summary = pd.DataFrame({
            "Metric": [
                "Flagged transactions",
                "Flagged exposure",
                "Working-capital base",
                "Working-capital reference",
                "% working capital at risk",
                "Outflow risk",
                "Inflow risk",
                "Estimated cash-frozen days",
                "DSO detour estimate",
                "DPO detour estimate",
            ],
            "Value": [
                diagnostics["flagged_count"],
                diagnostics["flagged_value"],
                diagnostics["working_capital_base"],
                diagnostics["working_capital_label"],
                diagnostics["working_capital_ratio"] / 100,
                diagnostics["outflow_value"],
                diagnostics["inflow_value"],
                diagnostics["estimated_cash_frozen_days"],
                diagnostics["dso_detour_days"] if diagnostics["dso_detour_days"] is not None else "Not available",
                diagnostics["dpo_detour_days"] if diagnostics["dpo_detour_days"] is not None else "Not available",
            ],
        })
        summary.to_excel(writer, sheet_name="Liquidity Summary", index=False)

        diagnostics["ageing_matrix"].to_excel(
            writer, sheet_name="Ageing Matrix", index=False
        )
        diagnostics["vendor_concentration"].to_excel(
            writer, sheet_name="Vendor Concentration", index=False
        )
        diagnostics["customer_concentration"].to_excel(
            writer, sheet_name="Customer Concentration", index=False
        )

        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for col in ws.columns:
                width = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in col
                )
                ws.column_dimensions[col[0].column_letter].width = min(
                    max(width + 2, 12), 55
                )

    output.seek(0)
    return output.getvalue()


def create_wc_assignment_excel(diagnostics):
    output = io.BytesIO()
    df = diagnostics["flagged"].copy()

    if not df.empty:
        df["Suggested Team"] = np.select(
            [
                df["Risk Level"].eq("Critical"),
                df["Risk Level"].eq("High"),
                df["Risk Level"].isin(["Medium", "Low"]),
            ],
            [
                "Senior Audit / Finance Control",
                "Audit Senior",
                "Audit Associate / Review Team",
            ],
            default="Review Team",
        )

        df["Suggested Action"] = np.select(
            [
                df["Risk Level"].eq("Critical"),
                df["Risk Level"].eq("High"),
                df["Risk Level"].isin(["Medium", "Low"]),
            ],
            [
                "Immediate evidence tracing and payment-control review",
                "Priority supporting-document and approval review",
                "Review pattern and supporting evidence",
            ],
            default="Professional review",
        )

        df["Assignment Status"] = "Pending assignment"

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(
            writer,
            sheet_name="Audit Assignments",
            index=False,
        )

    output.seek(0)
    return output.getvalue()




# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🔎 Inspectra"
    )

    st.caption(
        "100% Financial Data Inspection"
    )

    st.markdown("---")

    st.markdown(
        "### Designed for"
    )

    st.write(
        "• Chartered Accountants\n\n"
        "• Audit professionals\n\n"
        "• Tax consultants\n\n"
        "• Finance teams"
    )

    st.markdown("---")

    st.markdown(
        "### Behind the scenes"
    )

    st.write(
        "✓ 100% population analysis\n\n"
        "✓ Split-payment trails\n\n"
        "✓ Employee-bank relationships\n\n"
        "✓ Duplicate detection\n\n"
        "✓ Benford screening\n\n"
        "✓ Machine-learning anomaly scoring"
    )

    st.markdown("---")

    st.info(
        "Red flags are investigation signals, "
        "not automatic findings."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="inspectra-title">🔎 Inspectra</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="inspectra-subtitle">'
    "100% Financial Data Inspection • Explainable Red Flags "
    "• Investigation Trails • Audit Workpaper Support"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h2>Inspect everything. Follow the trail.</h2>
        <p>
        Inspectra combines rule-based analytics, Benford screening
        and machine learning to identify unusual financial activity
        and explain why each transaction deserves attention.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_upload, tab_flags, tab_trails, tab_analysis, tab_guide = st.tabs(
    [
        "📤 Upload & Inspect",
        "🚩 Red Flags",
        "🔗 Investigation Trails",
        "📊 Analysis & Reports",
        "📚 Simple Guide",
    ]
)


# ============================================================
# UPLOAD
# ============================================================

with tab_upload:

    st.markdown(
        '<div class="section-heading">'
        "Upload accounting data"
        "</div>",
        unsafe_allow_html=True,
    )

    st.write(
        "Upload your General Ledger, payroll register, "
        "payment register or other transaction export. "
        "Inspectra will analyse the complete population."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV, Excel (.xlsx/.xls) or PDF",
        type=[
            "csv",
            "xlsx",
            "xls",
            "pdf",
        ],
        help=(
            "CSV and Excel are recommended for best results. "
            "PDF extraction works best when the PDF contains "
            "selectable table data."
        ),
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            """
            <div class="info-card">
            <b>Recommended data fields</b><br><br>
            Date • Transaction ID • Employee ID • Employee Name
            • Department • Bank Account • Amount • Vendor
            • Description
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:

        st.markdown(
            """
            <div class="success-card">
            <b>100% population testing</b><br><br>
            Inspectra checks every uploaded transaction rather
            than depending only on random sampling.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded_file:

        st.write(
            f"**File selected:** `{uploaded_file.name}`"
        )

        if st.button(
            "🚀 Run 100% Inspection",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Reading file and running the Inspectra engines..."
            ):

                try:

                    raw_data = load_file(
                        uploaded_file
                    )

                    missing = validate_columns(
                        raw_data
                    )

                    if missing:

                        st.error(
                            "The file is missing required fields:"
                        )

                        st.write(
                            ", ".join(
                                missing
                            )
                        )

                        st.info(
                            "Inspectra needs these fields to "
                            "perform reliable transaction-level "
                            "analysis."
                        )

                    elif len(raw_data) < MIN_RECORDS:

                        st.error(
                            f"The file contains only "
                            f"{len(raw_data)} records. "
                            f"At least {MIN_RECORDS} records "
                            "are required."
                        )

                    else:

                        (
                            results,
                            benford,
                            account_summary,
                        ) = analyze_data(
                            raw_data
                        )

                        st.session_state.results = (
                            results
                        )

                        st.session_state.benford = (
                            benford
                        )

                        st.session_state.account_summary = (
                            account_summary
                        )

                        st.session_state.filename = (
                            uploaded_file.name
                        )

                        st.session_state.raw_data = (
                            raw_data
                        )

                        st.success(
                            "Inspection completed successfully."
                        )

                        st.rerun()

                except Exception as exc:

                    st.error(
                        "Inspectra could not process this file."
                    )

                    st.exception(exc)

    if st.session_state.results is not None:

        results = st.session_state.results

        st.markdown(
            '<div class="section-heading">'
            "Inspection overview"
            "</div>",
            unsafe_allow_html=True,
        )

        total = len(results)

        flagged = int(
            (
                results["Risk Level"]
                != "Normal"
            ).sum()
        )

        high = int(
            (
                results["Risk Level"]
                == "High"
            ).sum()
        )

        cash = results[
            "Potential Cash Impact"
        ].sum()

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Transactions inspected",
            f"{total:,}",
        )

        c2.metric(
            "Red flags",
            f"{flagged:,}",
        )

        c3.metric(
            "High risk",
            f"{high:,}",
        )

        c4.metric(
            "Potential exposure",
            money(cash),
        )

        st.markdown(
            "### Uploaded population preview"
        )

        st.dataframe(
            results.head(20),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# RED FLAGS
# ============================================================

with tab_flags:

    st.markdown(
        '<div class="section-heading">'
        "🚩 Red Flag Transactions"
        "</div>",
        unsafe_allow_html=True,
    )

    results = st.session_state.results

    if results is None:

        st.info(
            "Upload a file and run the inspection first."
        )

    else:

        red_flags = results[
            results["Risk Level"]
            != "Normal"
        ].copy()

        if red_flags.empty:

            st.markdown(
                """
                <div class="success-card">
                <b>No red flags were identified.</b><br><br>
                Inspectra did not identify a major anomaly using
                the current inspection engines. This does not
                replace normal audit procedures.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Transactions for review",
                len(red_flags),
            )

            c2.metric(
                "High",
                int(
                    (
                        red_flags[
                            "Risk Level"
                        ]
                        == "High"
                    ).sum()
                ),
            )

            c3.metric(
                "Medium",
                int(
                    (
                        red_flags[
                            "Risk Level"
                        ]
                        == "Medium"
                    ).sum()
                ),
            )

            c4.metric(
                "Flagged value",
                money(
                    red_flags[
                        "Potential Cash Impact"
                    ].sum()
                ),
            )

            selected_risk = st.multiselect(
                "Show risk levels",
                [
                    "High",
                    "Medium",
                    "Low",
                ],
                default=[
                    "High",
                    "Medium",
                    "Low",
                ],
            )

            filtered = red_flags[
                red_flags[
                    "Risk Level"
                ].isin(
                    selected_risk
                )
            ]

            st.dataframe(
                filtered[
                    [
                        "transaction_id",
                        "date",
                        "employee_name",
                        "department",
                        "bank_account",
                        "amount",
                        "Risk Level",
                        "Risk Score",
                        "Red Flag Reason",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "### Transaction-level explanation"
            )

            for _, row in filtered.iterrows():

                if row["Risk Level"] == "High":
                    icon = "🔴"

                elif row["Risk Level"] == "Medium":
                    icon = "🟠"

                else:
                    icon = "🟡"

                with st.expander(
                    f"{icon} "
                    f"{row['transaction_id']} — "
                    f"{money(row['amount'])} — "
                    f"{row['Risk Level']} risk"
                ):

                    a, b, c, d = st.columns(4)

                    a.write(
                        f"**Date:** "
                        f"{format_date(row['date'])}"
                    )

                    b.write(
                        f"**Employee:** "
                        f"{row['employee_name']}"
                    )

                    c.write(
                        f"**Department:** "
                        f"{row['department']}"
                    )

                    d.write(
                        f"**Risk score:** "
                        f"{row['Risk Score']}/100"
                    )

                    st.markdown(
                        "#### 🚩 Why was this transaction flagged?"
                    )

                    st.warning(
                        row[
                            "Red Flag Reason"
                        ]
                    )

                    st.markdown(
                        "#### 👨‍💼 What should the auditor do?"
                    )

                    st.write(
                        "Trace the transaction to the original "
                        "invoice, supporting document, approval "
                        "record, bank statement and ledger entry. "
                        "Where payroll is involved, verify the "
                        "employee master record and bank details. "
                        "Use the Investigation Trail to review "
                        "related transactions before documenting "
                        "the final conclusion."
                    )

                    trail = row[
                        "Investigation Trail"
                    ]

                    if trail:

                        st.markdown(
                            "#### 🔗 Related transactions"
                        )

                        st.write(
                            ", ".join(
                                trail
                            )
                        )


# ============================================================
# INVESTIGATION TRAILS
# ============================================================

with tab_trails:

    st.markdown(
        '<div class="section-heading">'
        "🔗 Investigation Trails"
        "</div>",
        unsafe_allow_html=True,
    )

    results = st.session_state.results

    if results is None:

        st.info(
            "Run an inspection first."
        )

    else:

        red_flags = results[
            results["Risk Level"]
            != "Normal"
        ].copy()

        if red_flags.empty:

            st.success(
                "There are no investigation trails to display."
            )

        else:

            st.write(
                "This screen helps you move beyond a single "
                "transaction. Select a flagged transaction and "
                "Inspectra will show the related records that "
                "may form part of the same trail."
            )

            transaction_options = (
                red_flags[
                    "transaction_id"
                ]
                .astype(str)
                .tolist()
            )

            selected_transaction = st.selectbox(
                "Select a flagged transaction",
                transaction_options,
            )

            selected_row = red_flags[
                red_flags[
                    "transaction_id"
                ].astype(str)
                == selected_transaction
            ].iloc[0]

            st.markdown(
                f"### {selected_transaction}"
            )

            st.markdown(
                f"""
                <div class="danger-card">
                <b>Risk:</b> {selected_row['Risk Level']}
                &nbsp;&nbsp; | &nbsp;&nbsp;
                <b>Score:</b> {selected_row['Risk Score']}/100
                <br><br>
                <b>Reason:</b> {safe_text(selected_row['Red Flag Reason'])}
                </div>
                """,
                unsafe_allow_html=True,
            )

            trail_ids = selected_row[
                "Investigation Trail"
            ]

            if not trail_ids:

                st.info(
                    "No directly connected transactions were "
                    "identified for this red flag."
                )

            else:

                st.markdown(
                    "### Related transactions"
                )

                related = results[
                    results[
                        "transaction_id"
                    ]
                    .astype(str)
                    .isin(
                        [
                            str(transaction_id)
                            for transaction_id
                            in trail_ids
                        ]
                    )
                ].copy()

                st.dataframe(
                    related[
                        [
                            "transaction_id",
                            "date",
                            "employee_id",
                            "employee_name",
                            "department",
                            "bank_account",
                            "amount",
                            "vendor",
                            "Risk Level",
                            "Risk Score",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown(
                    "### Simple interpretation"
                )

                st.write(
                    "These records are connected because Inspectra "
                    "found a relationship through payment timing, "
                    "employee, bank account, vendor or matching "
                    "transaction characteristics. The connection "
                    "is a lead for investigation, not proof that "
                    "the transactions are improper."
                )


# ============================================================
# ANALYSIS
# ============================================================

with tab_analysis:

    st.markdown(
        '<div class="section-heading">'
        "📊 Analysis & Reports"
        "</div>",
        unsafe_allow_html=True,
    )

    results = st.session_state.results

    if results is None:

        st.info(
            "Run an inspection first."
        )

    else:

        benford = st.session_state.benford

        # ----------------------------------------------------
        # Interactive Risk Distribution
        # ----------------------------------------------------

        st.markdown(
            "### Risk distribution"
        )

        st.caption(
            "Click any bar to show ONLY the transactions belonging to that risk level."
        )

        risk_order = [
            "High",
            "Medium",
            "Low",
            "Normal",
        ]

        risk_counts = (
            results[
                "Risk Level"
            ]
            .astype(str)
            .str.strip()
            .value_counts()
            .reindex(
                risk_order,
                fill_value=0,
            )
        )

        risk_chart_df = (
            risk_counts
            .rename_axis("Risk Level")
            .reset_index(name="Transactions")
        )

        import plotly.express as px

        risk_fig = px.bar(
            risk_chart_df,
            x="Risk Level",
            y="Transactions",
            text="Transactions",
            category_orders={
                "Risk Level": risk_order
            },
        )

        risk_fig.update_traces(
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Transactions: %{y}"
                "<extra></extra>"
            )
        )

        risk_fig.update_layout(
            clickmode="event+select",
            xaxis_title="Risk level",
            yaxis_title="Number of transactions",
            showlegend=False,
        )

        if "inspectra_selected_risk" not in st.session_state:
            st.session_state.inspectra_selected_risk = None

        risk_event = st.plotly_chart(
            risk_fig,
            use_container_width=True,
            key="inspectra_interactive_risk_chart",
            on_select="rerun",
            selection_mode="points",
        )

        try:
            selected_points = risk_event.selection["points"]
        except Exception:
            selected_points = []

        if selected_points:
            clicked_risk = selected_points[0].get("x")

            if clicked_risk in risk_order:
                st.session_state.inspectra_selected_risk = clicked_risk

        selected_risk = (
            st.session_state.inspectra_selected_risk
        )

        if selected_risk:

            filtered_results = results[
                results[
                    "Risk Level"
                ]
                .astype(str)
                .str.strip()
                .str.casefold()
                == selected_risk.casefold()
            ].copy()

            st.markdown(
                f"### {selected_risk} Risk Transactions "
                f"({len(filtered_results)})"
            )

            st.caption(
                f"Showing only transactions classified as {selected_risk} risk."
            )

            if filtered_results.empty:

                st.info(
                    f"No transactions are currently classified as {selected_risk}."
                )

            else:

                st.dataframe(
                    filtered_results,
                    use_container_width=True,
                    hide_index=True,
                )

        else:

            filtered_results = results.copy()

            st.markdown(
                f"### All Transactions ({len(filtered_results)})"
            )

            st.dataframe(
                filtered_results,
                use_container_width=True,
                hide_index=True,
            )

        if st.button(
            "Show All Transactions",
            key="inspectra_clear_risk_filter",
        ):

            st.session_state.inspectra_selected_risk = None
            st.rerun()

        # ----------------------------------------------------
        # Detection engines
        # ----------------------------------------------------

        st.markdown(
            "### Detection engine results"
        )

        engine_summary = pd.DataFrame(
            {
                "Detection engine": [
                    "Split-payment tracing",
                    "Shared-account analysis",
                    "Duplicate detection",
                    "Unusual amount analysis",
                    "Benford screening",
                    "Machine-learning analysis",
                ],
                "Flags": [
                    int(
                        results[
                            "Split Payment"
                        ].sum()
                    ),
                    int(
                        results[
                            "Shared Bank Account"
                        ].sum()
                    ),
                    int(
                        results[
                            "Possible Duplicate"
                        ].sum()
                    ),
                    int(
                        results[
                            "Unusual Amount"
                        ].sum()
                    ),
                    int(
                        results[
                            "Benford Review"
                        ].sum()
                    ),
                    int(
                        results[
                            "ML Anomaly"
                        ].sum()
                    ),
                ],
            }
        )

        st.dataframe(
            engine_summary,
            use_container_width=True,
            hide_index=True,
        )

        # ----------------------------------------------------
        # Benford
        # ----------------------------------------------------

        st.markdown(
            "### 🔢 Benford's Law"
        )

        if not benford["usable"]:

            st.info(
                "There are not enough usable numerical values "
                "for a meaningful Benford analysis."
            )

        else:

            b1, b2 = st.columns(2)

            b1.metric(
                "Chi-square statistic",
                round(
                    benford[
                        "chi_square"
                    ],
                    3,
                ),
            )

            b2.metric(
                "Approximate p-value",
                round(
                    benford[
                        "p_value"
                    ],
                    4,
                ),
            )

            benford_table = pd.DataFrame(
                {
                    "First digit": range(1, 10),
                    "Observed %": [
                        round(
                            benford[
                                "observed"
                            ][digit]
                            * 100,
                            2,
                        )
                        for digit in range(1, 10)
                    ],
                    "Expected %": [
                        round(
                            benford[
                                "expected"
                            ][digit]
                            * 100,
                            2,
                        )
                        for digit in range(1, 10)
                    ],
                }
            )

            st.dataframe(
                benford_table,
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "Benford's Law is a screening technique. "
                "An unusual distribution should trigger "
                "investigation, not an automatic conclusion."
            )

        # ----------------------------------------------------
        # Actionable Working-Capital Diagnostic
        # ----------------------------------------------------

        st.markdown(
            "### 💰 Working-capital diagnostic"
        )

        diagnostics = build_working_capital_diagnostics(
            results
        )

        st.caption(
            "This section turns flagged transactions into an "
            "exposure, ageing, liquidity and prioritisation view."
        )

        # ----------------------------------------------------
        # Operational liquidity / impact ratios
        # ----------------------------------------------------

        st.markdown(
            "#### 1. Operational Liquidity & Impact"
        )

        l1, l2, l3, l4, l5 = st.columns(5)

        l1.metric(
            "Flagged exposure",
            money(diagnostics["flagged_value"]),
        )

        l2.metric(
            "% working capital at risk",
            f"{diagnostics['working_capital_ratio']:.1f}%",
        )

        l3.metric(
            "Flagged transactions",
            f"{diagnostics['flagged_count']:,}",
        )

        l4.metric(
            "Estimated cash-frozen days",
            f"{diagnostics['estimated_cash_frozen_days']:.0f}",
        )

        velocity_display = (
            f"{diagnostics['velocity_pct']:+.1f}%"
            if diagnostics["velocity_pct"] is not None
            else "N/A"
        )

        l5.metric(
            "WoW flagged-value velocity",
            velocity_display,
        )

        st.info(
            f"Reference used for the working-capital percentage: "
            f"**{diagnostics['working_capital_label']}** "
            f"({money(diagnostics['working_capital_base'])}). "
            "For datasets that do not contain AP, AR or NWC balances, "
            "Inspectra uses uploaded transaction value only as a clearly-labelled proxy."
        )

        if (
            diagnostics["ap_value"] == 0
            and diagnostics["ar_value"] == 0
            and diagnostics["nwc_value"] == 0
        ):
            st.caption(
                "Tip: include Accounts Payable, Accounts Receivable or "
                "Net Working Capital in the source file to calculate a "
                "balance-based % Working Capital at Risk."
            )

        # Cash impact cards
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(
                f"""
                <div class="danger-card">
                <b>Outflow Risk</b><br>
                <span style="font-size:23px;font-weight:700;">
                {money(diagnostics['outflow_value'])}
                </span><br>
                Vendor payments, duplicate disbursements, split-payment patterns
                and other flagged items that may represent cash leaving the business.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c2:
            st.markdown(
                f"""
                <div class="info-card">
                <b>Inflow Risk</b><br>
                <span style="font-size:23px;font-weight:700;">
                {money(diagnostics['inflow_value'])}
                </span><br>
                Receivable, collection or other flagged items that may delay expected
                cash inflows where such activity can be identified from the source.
                </div>
                """,
                unsafe_allow_html=True,
            )

        d1, d2 = st.columns(2)

        with d1:
            if diagnostics["dso_detour_days"] is None:
                st.metric("DSO detour estimate", "Not available")
            else:
                st.metric(
                    "DSO detour estimate",
                    f"{diagnostics['dso_detour_days']:.1f} days",
                )

        with d2:
            if diagnostics["dpo_detour_days"] is None:
                st.metric("DPO detour estimate", "Not available")
            else:
                st.metric(
                    "DPO detour estimate",
                    f"{diagnostics['dpo_detour_days']:.1f} days",
                )

        st.caption(
            "DSO/DPO detour values are exposure-to-balance indicators, not "
            "statutory DSO/DPO calculations. Actual DSO/DPO needs appropriate "
            "period revenue/purchase and opening/closing balance data."
        )

        # ----------------------------------------------------
        # Ageing & velocity
        # ----------------------------------------------------

        st.markdown(
            "#### 2. Ageing & Velocity Breakdown"
        )

        a1, a2 = st.columns(2)

        with a1:
            st.markdown("**Ageing matrix**")
            st.dataframe(
                diagnostics["ageing_matrix"].style.format({
                    "Flagged_Value": "₹{:,.2f}",
                    "Transactions": "{:,.0f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with a2:
            st.markdown("**High/Critical exposure by age**")

            if diagnostics["risk_aging"].empty:
                st.info("No flagged ageing data is available.")
            else:
                risk_age_display = diagnostics["risk_aging"].pivot(
                    index="Ageing Bucket",
                    columns="High_or_Critical",
                    values="Flagged_Value",
                ).fillna(0).reset_index()

                st.dataframe(
                    risk_age_display.style.format(
                        {col: "₹{:,.2f}" for col in risk_age_display.columns[1:]}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("**Flagged-value velocity over time**")

        if diagnostics["velocity_df"].empty:
            st.info(
                "Not enough dated flagged transactions are available to calculate velocity."
            )
        else:
            velocity_chart = diagnostics["velocity_df"].set_index("Week")[
                "Flagged_Value"
            ]
            st.line_chart(
                velocity_chart,
                use_container_width=True,
            )

        if diagnostics["velocity_pct"] is not None:
            direction = (
                "increased"
                if diagnostics["velocity_pct"] > 0
                else "decreased"
            )
            st.write(
                f"Flagged value **{direction} by "
                f"{abs(diagnostics['velocity_pct']):.1f}%** "
                "versus the previous available week."
            )

        # ----------------------------------------------------
        # Prioritisation / Pareto
        # ----------------------------------------------------

        st.markdown(
            "#### 3. Prioritisation & Action Triggers"
        )

        st.markdown("**Top-10 flagged exposure**")

        if diagnostics["top10"].empty:
            st.success(
                "No flagged transactions are available for prioritisation."
            )
        else:
            top_cols = [
                c for c in [
                    "transaction_id",
                    "date",
                    "employee_id",
                    "employee_name",
                    "department",
                    "vendor",
                    "amount",
                    "Risk Level",
                    "Risk Score",
                    "Red Flag Reason",
                ]
                if c in diagnostics["top10"].columns
            ]

            st.dataframe(
                diagnostics["top10"][top_cols],
                use_container_width=True,
                hide_index=True,
            )

            top10_total = float(
                diagnostics["top10"]["amount"].sum()
            )
            top10_share = (
                top10_total / diagnostics["flagged_value"] * 100
                if diagnostics["flagged_value"] > 0
                else 0
            )

            st.caption(
                f"Top-10 flagged value: {money(top10_total)} "
                f"({top10_share:.1f}% of total flagged exposure)."
            )

        # Vendor/customer concentration
        vc1, vc2 = st.columns(2)

        with vc1:
            st.markdown("**Vendor concentration**")
            if diagnostics["vendor_concentration"].empty:
                st.info("No vendor concentration data is available.")
            else:
                st.dataframe(
                    diagnostics["vendor_concentration"].head(10).style.format({
                        "Flagged_Value": "₹{:,.2f}",
                        "Flagged_Transactions": "{:,.0f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                top_vendor = diagnostics["vendor_concentration"].iloc[0]
                st.caption(
                    f"Highest concentration: **{top_vendor['Vendor']}** "
                    f"accounts for {money(top_vendor['Flagged_Value'])} "
                    f"across {int(top_vendor['Flagged_Transactions'])} flagged transaction(s)."
                )

        with vc2:
            st.markdown("**Customer concentration**")
            if diagnostics["customer_concentration"].empty:
                st.info(
                    "Customer concentration requires a Customer/Client column "
                    "in the uploaded data."
                )
            else:
                st.dataframe(
                    diagnostics["customer_concentration"].head(10).style.format({
                        "Flagged_Value": "₹{:,.2f}",
                        "Flagged_Transactions": "{:,.0f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                top_customer = diagnostics["customer_concentration"].iloc[0]
                st.caption(
                    f"Highest concentration: **{top_customer['Customer']}** "
                    f"accounts for {money(top_customer['Flagged_Value'])} "
                    f"across {int(top_customer['Flagged_Transactions'])} flagged transaction(s)."
                )

        # ----------------------------------------------------
        # Direct action triggers
        # ----------------------------------------------------

        st.markdown("**Action triggers**")

        act1, act2 = st.columns(2)

        with act1:
            st.markdown("##### 🧊 Freeze Payments")

            st.write(
                "Generates a review hold-list for finance/ERP teams. "
                "Inspectra does not automatically freeze or block payments."
            )

            st.download_button(
                "⬇️ Generate Payment Hold List",
                data=create_wc_hold_list_excel(
                    diagnostics
                ),
                file_name="Inspectra_Payment_Hold_List.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
                disabled=diagnostics["hold_list"].empty,
            )

        with act2:
            st.markdown("##### 👥 Assign to Audit")

            st.write(
                "Generates a prioritised assignment sheet for the audit team. "
                "Actual assignment remains under the audit manager's control."
            )

            st.download_button(
                "⬇️ Generate Audit Assignment List",
                data=create_wc_assignment_excel(
                    diagnostics
                ),
                file_name="Inspectra_Audit_Assignment_List.xlsx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
                disabled=diagnostics["flagged"].empty,
            )

        st.warning(
            "These action triggers create worklists only. They do not connect "
            "to or modify an ERP, bank account or payment system."
        )

        # ----------------------------------------------------
        # Downloads
        # ----------------------------------------------------

        st.markdown(
            "### 📥 Download Reports"
        )

        st.write(
            "Choose the format required for your report."
        )

        excel_bytes = create_excel_report(
            results,
            benford,
            st.session_state.account_summary,
            st.session_state.filename,
        )

        pdf_bytes = create_pdf_report(
            results,
            benford,
            st.session_state.filename,
        )

        d1, d2 = st.columns(2)

        with d1:

            st.download_button(
                "📊 Download Excel Workpaper",
                data=excel_bytes,
                file_name=(
                    "Inspectra_Audit_Workpaper.xlsx"
                ),
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet"
                ),
                use_container_width=True,
            )

        with d2:

            st.download_button(
                "📄 Download PDF Report",
                data=pdf_bytes,
                file_name=(
                    "Inspectra_Audit_Report.pdf"
                ),
                mime="application/pdf",
                use_container_width=True,
            )


# ============================================================
# SIMPLE GUIDE
# ============================================================

with tab_guide:

    st.markdown(
        '<div class="section-heading">'
        "📚 Inspectra — Simple Guide"
        "</div>",
        unsafe_allow_html=True,
    )

    st.write(
        "This guide explains the important terms in simple language. "
        "You do not need to be an audit, finance or data-science expert "
        "to use Inspectra."
    )

    st.info(
        "Think of Inspectra as a smart review assistant: it checks every "
        "uploaded transaction, points out unusual patterns, explains why "
        "they were flagged and helps you decide what should be reviewed next."
    )

    with st.expander("1. How do I use Inspectra?"):

        st.write(
            "Upload your General Ledger, payment register, payroll file or "
            "other transaction data. Check the recognised columns, then click "
            "'Run 100% Inspection'. Inspectra checks the complete uploaded "
            "population and presents the results in simple categories."
        )

    with st.expander("2. What is 100% inspection?"):

        st.write(
            "It means Inspectra checks every transaction in the uploaded file. "
            "It does not deliberately select only a small random sample."
        )

    with st.expander("3. What is a red flag?"):

        st.write(
            "A red flag is a warning that something looks unusual and deserves "
            "professional review. A red flag is not proof that fraud or an error "
            "has occurred."
        )

    with st.expander("4. What is a risk level?"):

        st.write(
            "Risk level is a simple way of showing how strongly the current "
            "tests point to an unusual transaction. Inspectra uses Normal, "
            "Low, Medium, High and, where applicable, Critical."
        )

    with st.expander("5. What is a risk score?"):

        st.write(
            "The risk score is a 0–100 indicator based on the warning signals "
            "found by Inspectra. A higher score means more or stronger warning "
            "signals. It is not a percentage chance of fraud."
        )

    with st.expander("6. What is split payment?"):

        st.write(
            "A split payment happens when several smaller payments may be "
            "related to one larger purchase. For example, several payments "
            "just below an approval limit may deserve review because together "
            "they are much larger."
        )

    with st.expander("7. What is an investigation trail?"):

        st.write(
            "An investigation trail connects related transactions so that a "
            "professional can follow the story instead of looking at one "
            "transaction in isolation."
        )

    with st.expander("8. What is a shared bank account?"):

        st.write(
            "It means the same bank account appears against more than one "
            "employee record. There can be legitimate reasons, so the bank "
            "ownership and employee master data should be checked."
        )

    with st.expander("9. What is a ghost employee warning?"):

        st.write(
            "Inspectra does not declare that an employee is fake. It identifies "
            "unusual payroll relationships, such as unexpected employee-bank "
            "account combinations, that deserve verification."
        )

    with st.expander("10. What is a duplicate transaction?"):

        st.write(
            "Two or more records look very similar in important fields such as "
            "date, employee, bank account, amount, department or vendor. "
            "The professional should determine whether the transaction was "
            "legitimately repeated or accidentally duplicated."
        )

    with st.expander("11. What is Benford's Law?"):

        st.write(
            "Benford's Law describes the expected pattern of first digits in "
            "many naturally occurring sets of numbers. Inspectra uses it as "
            "a screening test. A Benford warning does not prove that an entry "
            "was manipulated."
        )

    with st.expander("12. What is Chi-square testing?"):

        st.write(
            "Chi-square is a mathematical comparison used to see whether the "
            "observed first-digit pattern is materially different from the "
            "Benford pattern. In simple terms, it asks: 'Is the difference "
            "large enough to deserve attention?'"
        )

    with st.expander("13. What is machine learning?"):

        st.write(
            "Machine learning is a computer method that looks at several "
            "characteristics together. Inspectra can consider transaction "
            "amount, employee activity, bank-account usage, department "
            "activity, vendor activity and other features."
        )

    with st.expander("14. What is Isolation Forest?"):

        st.write(
            "Isolation Forest is the machine-learning method used by Inspectra "
            "to find records that look different from the rest of the uploaded "
            "population."
        )

    with st.expander("15. What is an anomaly?"):

        st.write(
            "An anomaly is a transaction or pattern that looks different from "
            "what is common in the data. An anomaly may be completely legitimate; "
            "it simply deserves a closer look."
        )

    with st.expander("16. What is working capital?"):

        st.write(
            "Working capital is the money a business needs for its day-to-day "
            "operations. It is closely connected with short-term assets such as "
            "receivables and inventory and short-term obligations such as payables."
        )

    with st.expander("17. What is Working-Capital Exposure?"):

        st.write(
            "Working-capital exposure is the amount of money represented by "
            "transactions that may require investigation. Inspectra uses this "
            "to show how much value may be affected by the identified warnings."
        )

    with st.expander("18. What is '% Working Capital at Risk'?"):

        st.write(
            "This compares the flagged transaction value with a working-capital "
            "reference such as Accounts Payable, Accounts Receivable or Net "
            "Working Capital when that information is available. It helps answer: "
            "'How large is the flagged amount compared with the available base?'"
        )

    with st.expander("18A. What is % Working Capital at Risk?"):

        st.write(
            "It shows how large the flagged transaction value is compared with "
            "the selected working-capital reference. Inspectra uses Accounts Payable, "
            "Accounts Receivable or Net Working Capital when these balances are "
            "provided in the uploaded data."
        )

    with st.expander("18B. What is a cash-impact category?"):

        st.write(
            "It tells you whether a flagged item mainly affects money going out "
            "(Outflow Risk) or money expected to come in (Inflow Risk)."
        )

    with st.expander("18C. What is an ageing matrix?"):

        st.write(
            "It groups flagged values by how old they are, such as 0–30, 31–60, "
            "61–90 and 90+ days. It helps identify older exposure that may need "
            "faster review."
        )

    with st.expander("18D. What is velocity?"):

        st.write(
            "Velocity measures how quickly flagged transaction value is changing "
            "over time. A positive percentage means flagged value increased versus "
            "the previous available week."
        )

    with st.expander("18E. What is vendor concentration?"):

        st.write(
            "It shows whether many flagged transactions or a large flagged amount "
            "is connected to one vendor. This helps the reviewer decide where to "
            "start detailed testing."
        )

    with st.expander("18F. What is a Payment Hold List?"):

        st.write(
            "It is a downloadable list of flagged transactions that finance may "
            "review before releasing further payments. Inspectra does not actually "
            "freeze payments."
        )

    with st.expander("18G. What is an Audit Assignment List?"):

        st.write(
            "It is a downloadable worklist that suggests review teams and actions "
            "based on risk level. The audit manager remains responsible for the actual assignment."
        )

    with st.expander("18H. What are DSO and DPO detour estimates?"):

        st.write(
            "They are exposure indicators that compare flagged inflow/outflow value "
            "with available AR/AP balances. They are not the company's actual statutory "
            "DSO or DPO unless the required period balances and revenue/purchase data are supplied."
        )

    with st.expander("19. What is Outflow Risk?"):

        st.write(
            "Outflow Risk relates to transactions where company cash is going out, "
            "such as vendor payments, duplicate payments or possible split purchases. "
            "These items may deserve review before further payments are made."
        )

    with st.expander("20. What is Inflow Risk?"):

        st.write(
            "Inflow Risk relates to money the business expects to receive, such as "
            "customer collections or receivables. A warning can indicate that cash "
            "may be delayed or that the underlying transaction needs review."
        )

    with st.expander("21. What is ageing?"):

        st.write(
            "Ageing tells you how long a flagged transaction has been outstanding "
            "or how old it is relative to the latest transaction date in the uploaded "
            "data. Older high-risk items can deserve faster attention."
        )

    with st.expander("22. What are the ageing buckets?"):

        st.write(
            "Inspectra groups flagged items into simple age bands: 0–30 days, "
            "31–60 days, 61–90 days and 90+ days. This makes it easier to see "
            "where older exposure is concentrated."
        )

    with st.expander("23. What are 'estimated cash-frozen days'?"):

        st.write(
            "This is an operational indicator showing the amount-weighted age of "
            "flagged transactions. It gives a rough sense of how long flagged value "
            "has been tied up. It is not a statutory accounting ratio."
        )

    with st.expander("24. What is velocity?"):

        st.write(
            "Velocity shows how quickly flagged value is increasing or decreasing "
            "over time. For example, +15% means the flagged value increased by "
            "about 15% compared with the previous available week."
        )

    with st.expander("25. What is vendor concentration?"):

        st.write(
            "Vendor concentration shows whether a large share of flagged value "
            "is connected to one vendor or a small group of vendors. A high "
            "concentration can help the audit team decide where to start."
        )

    with st.expander("26. What is a Pareto / Top-10 exposure list?"):

        st.write(
            "It lists the largest flagged transactions by value. This helps a "
            "professional focus first on the items that could have the biggest "
            "financial impact."
        )

    with st.expander("27. What is Cash Conversion Cycle (CCC)?"):

        st.write(
            "Cash Conversion Cycle is a way of thinking about how long cash is "
            "tied up in the operating cycle of a business. It connects the time "
            "taken to sell inventory, collect customer money and pay suppliers."
        )

    with st.expander("28. What is DSO?"):

        st.write(
            "Days Sales Outstanding (DSO) is a measure of how many days, on average, "
            "it takes a business to collect money from customers."
        )

    with st.expander("29. What is DPO?"):

        st.write(
            "Days Payable Outstanding (DPO) is a measure of how many days, on average, "
            "a business takes to pay suppliers."
        )

    with st.expander("30. What is Accounts Payable (AP)?"):

        st.write(
            "Accounts Payable is money the business owes to suppliers and other "
            "short-term creditors. It is usually a short-term liability."
        )

    with st.expander("31. What is Accounts Receivable (AR)?"):

        st.write(
            "Accounts Receivable is money customers owe the business for goods or "
            "services already supplied but not yet collected."
        )

    with st.expander("32. What is Net Working Capital (NWC)?"):

        st.write(
            "Net Working Capital is a broad measure of short-term operating liquidity. "
            "In simple terms, it compares short-term operating resources with "
            "short-term obligations."
        )

    with st.expander("33. What is Potential Cash Exposure?"):

        st.write(
            "It is the value of transactions currently marked for investigation. "
            "It should be treated as a review population, not as confirmed loss."
        )

    with st.expander("34. What is a Payment Hold List?"):

        st.write(
            "It is an Excel list of transactions that may need to be held for "
            "further review. Inspectra only creates the list; it does not actually "
            "freeze payments in your ERP or bank system."
        )

    with st.expander("35. What is Assign to Audit?"):

        st.write(
            "It creates a review list with suggested priority and action. The "
            "actual assignment of work remains under the control of the audit "
            "manager or responsible professional."
        )

    with st.expander("36. What should I do after a red flag appears?"):

        st.write(
            "Trace the transaction to the invoice, purchase order, approval, "
            "ledger entry, bank statement and other supporting evidence. For "
            "payroll items, check the employee master record and bank details. "
            "Use the Investigation Trail to understand related transactions."
        )

    with st.expander("37. Can Inspectra prove fraud?"):

        st.write(
            "No. Inspectra identifies patterns and transactions that deserve "
            "investigation. A qualified professional must independently verify "
            "the evidence and reach the appropriate conclusion."
        )

    with st.expander("38. What files can I upload?"):

        st.write(
            "Inspectra accepts CSV, Excel (.xlsx/.xls) and text-based PDF tables. "
            "Excel and CSV are generally easier to analyse because their columns "
            "and values are already structured."
        )

    with st.expander("39. What are the recommended columns?"):

        st.write(
            "The standard fields are: Date, Transaction ID, Employee ID, "
            "Employee Name, Department, Bank Account, Amount, Vendor and Description."
        )

    with st.expander("40. Why does Inspectra sometimes use the word 'possible'?"):

        st.write(
            "Words such as 'possible', 'indicator' and 'warning' are intentional. "
            "They remind the user that an analytical result needs evidence-based "
            "professional review before a conclusion is made."
        )

    st.markdown("### Simple workflow for a CA / auditor")

    st.success(
        "1️⃣ Upload the file → "
        "2️⃣ Check the recognised columns → "
        "3️⃣ Run 100% Inspection → "
        "4️⃣ Start with Critical/High risk → "
        "5️⃣ Click the Risk Distribution bars → "
        "6️⃣ Read the red-flag reason → "
        "7️⃣ Follow the Investigation Trail → "
        "8️⃣ Review evidence → "
        "9️⃣ Export the workpaper/report → "
        "🔟 Document the professional conclusion."
    )

    st.warning(
        "Important: Inspectra is an analytical support tool. Its warnings, "
        "risk scores, exposure estimates and ageing indicators should be "
        "independently verified by the responsible CA, auditor or tax professional."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    f"Inspectra v{VERSION} • "
    "100% transaction analytics • "
    "Explainable anomaly detection • "
    "Professional verification remains necessary."
)
