import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

train = pd.read_csv(ROOT_DIR / "data" / "train.csv")
later = pd.read_csv(ROOT_DIR / "data" / "later.csv")
holdout = pd.read_csv(ROOT_DIR / "data" / "holdout_features.csv")

print("DATASET SHAPES")
print(train.shape, later.shape, holdout.shape)

print("\n1. LEAKING COLUMN")
missing = set(train.columns) - set(holdout.columns)
print("Missing from holdout:", missing)

normal = train[train["units_sold"] < 1000]
print("GST correlation with target after excluding target typos:",
      normal["units_billed_in_gst_return"].corr(normal["units_sold"]))

print("\n2. UNIT MIX-UP")
print(train.groupby("locality")["discount_pct"].agg(["min", "mean", "max"]))

print("\n3. SENTINEL VALUE")
print("stock_lines = 999:",
      int((train["stock_lines"] == 999).sum()))

print("\n4. DUPLICATE ROWS")
print("Extra duplicate rows:", int(train.duplicated().sum()))

print("\n5. STUCK SENSOR")
numeric = train.select_dtypes(include=np.number).columns
sorted_train = train.sort_values(["date", "record_id"]).reset_index(drop=True)

for col in numeric:
    if col in ["units_sold", "units_billed_in_gst_return"]:
        continue

    run_id = sorted_train[col].ne(sorted_train[col].shift()).cumsum()
    runs = sorted_train.groupby(run_id).agg(
        value=(col, "first"),
        count=(col, "size"),
        start_date=("date", "first"),
        end_date=("date", "last")
    )
    print(col, "longest repeated run:", int(runs["count"].max()))

print("No run of 10 or more consecutive identical readings was detected.")

print("\n6. TARGET TYPOS")
typos = train[train["units_sold"] > 2000]
print("Affected rows:", len(typos))
print(typos[["record_id", "date", "units_sold"]].to_string(index=False))

print("\n7. SPELLING VARIANTS")
print("Original shop_type values:")
print(train["shop_type"].value_counts().to_string())

print("\nClean shop_type values:")
print(later["shop_type"].value_counts().to_string())

normalized = train["shop_type"].astype(str).str.strip().str.lower()
print("Genuine categories after normalization:")
print(sorted(normalized.replace({"hospital store": "hospital_store"}).replace(
    {"ind.": "independent", "cha.": "chain", "hos.": "hospital_store"}
).unique()))
