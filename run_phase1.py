from pathlib import Path
import json
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "raw" / "SAML-D.csv"
REPORT_DIR = BASE_DIR / "data" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SAML-D AML DATASET - PHASE 1")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC INFORMATION")
print("=" * 70)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f"  - {column}")


# ============================================================
# DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("2. DATA TYPES")
print("=" * 70)

print(df.dtypes)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("3. MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

if missing.sum() == 0:
    print("\n✓ No missing values found.")
else:
    print("\n⚠ Missing values found.")


# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("4. DUPLICATES")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count:,}")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("5. LAUNDERING LABEL DISTRIBUTION")
print("=" * 70)

label_counts = df["Is_laundering"].value_counts().sort_index()

print(label_counts)

label_percent = (
    df["Is_laundering"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nPercentage:")
print(label_percent)


# ============================================================
# LAUNDERING TYPES
# ============================================================

print("\n" + "=" * 70)
print("6. LAUNDERING TYPES")
print("=" * 70)

type_counts = df["Laundering_type"].value_counts()

print(type_counts.to_string())


# ============================================================
# AMOUNT STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("7. AMOUNT STATISTICS")
print("=" * 70)

print(df["Amount"].describe())


# ============================================================
# PAYMENT TYPES
# ============================================================

print("\n" + "=" * 70)
print("8. PAYMENT TYPES")
print("=" * 70)

print(df["Payment_type"].value_counts())


# ============================================================
# UNIQUE ACCOUNTS
# ============================================================

print("\n" + "=" * 70)
print("9. ACCOUNT INFORMATION")
print("=" * 70)

unique_senders = df["Sender_account"].nunique()
unique_receivers = df["Receiver_account"].nunique()

unique_accounts = pd.concat([
    df["Sender_account"],
    df["Receiver_account"]
]).nunique()

print(f"Unique sender accounts:   {unique_senders:,}")
print(f"Unique receiver accounts: {unique_receivers:,}")
print(f"Unique accounts overall:  {unique_accounts:,}")


# ============================================================
# DATE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("10. DATE INFORMATION")
print("=" * 70)

df["Date_parsed"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

print(f"Invalid dates: {df['Date_parsed'].isna().sum():,}")
print(f"First date:    {df['Date_parsed'].min()}")
print(f"Last date:     {df['Date_parsed'].max()}")
print(f"Unique dates:  {df['Date_parsed'].nunique():,}")


# ============================================================
# REPORT
# ============================================================

report = {
    "rows": int(len(df)),
    "columns": int(len(df.columns) - 1),  # exclude temporary Date_parsed
    "missing_values": int(df.isnull().sum().sum()),
    "duplicate_rows": int(duplicate_count),
    "normal_transactions": int(label_counts.get(0, 0)),
    "laundering_transactions": int(label_counts.get(1, 0)),
    "laundering_percentage": float(
        label_percent.get(1, 0)
    ),
    "unique_senders": int(unique_senders),
    "unique_receivers": int(unique_receivers),
    "unique_accounts": int(unique_accounts),
    "unique_laundering_types": int(
        df["Laundering_type"].nunique()
    ),
    "unique_payment_types": int(
        df["Payment_type"].nunique()
    ),
    "unique_payment_currencies": int(
        df["Payment_currency"].nunique()
    ),
    "unique_received_currencies": int(
        df["Received_currency"].nunique()
    ),
    "unique_sender_countries": int(
        df["Sender_bank_location"].nunique()
    ),
    "unique_receiver_countries": int(
        df["Receiver_bank_location"].nunique()
    ),
    "amount_min": float(df["Amount"].min()),
    "amount_max": float(df["Amount"].max()),
    "amount_mean": float(df["Amount"].mean()),
    "amount_median": float(df["Amount"].median()),
    "date_min": str(df["Date_parsed"].min().date()),
    "date_max": str(df["Date_parsed"].max().date()),
}

report_file = REPORT_DIR / "phase1_report.json"

with open(report_file, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=4)


print("\n" + "=" * 70)
print("PHASE 1 COMPLETE")
print("=" * 70)

print(f"\nReport saved to:")
print(report_file)

print("\n✓ Dataset is ready for Phase 2.")