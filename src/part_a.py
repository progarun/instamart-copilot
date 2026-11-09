"""Part A: Clean, Store, Validate and Query"""
import sqlite3
import pandas as pd
import numpy as np

def run_part_a():
    cust_raw = pd.read_csv("data/raw/customers.csv")
    prod_raw = pd.read_csv("data/raw/products.csv")
    ord_raw = pd.read_csv("data/raw/orders.csv")

    # 1. Clean customers
    cust = cust_raw.copy()
    cust["city"] = cust["city"].str.strip().str.title()
    gender_map = {
        "male": "male", "Male": "male", "M": "male", "m": "male", "MALE": "male",
        "female": "female", "Female": "female", "F": "female", "f": "female", "FEMALE": "female"
    }
    cust["gender"] = cust["gender"].fillna("").map(lambda x: gender_map.get(str(x).strip(), "unknown"))
    cust["age"] = pd.to_numeric(cust["age"], errors="coerce")
    cust.loc[(cust["age"] < 10) | (cust["age"] > 100), "age"] = np.nan
    cust["age"] = cust.groupby("city")["age"].transform(lambda g: g.fillna(g.median())).astype(int)
    cust["signup_date"] = pd.to_datetime(cust["signup_date"]).dt.strftime("%Y-%m-%d")

    # 2. Clean orders
    ord_df = ord_raw.copy()
    ord_df["order_ts"] = pd.to_datetime(ord_df["order_ts"])
    ord_df = ord_df.drop_duplicates()
    ord_df = ord_df.sort_values("order_ts").groupby("order_id", as_index=False).last()

    ord_df["order_amount"] = pd.to_numeric(ord_df["order_amount"], errors="coerce")
    ord_df = ord_df.merge(prod_raw[["product_id", "price_inr", "category"]], on="product_id", how="left")
    missing_amt = ord_df["order_amount"].isna()
    imputed_amt = (ord_df["price_inr"] * ord_df["quantity"] * (1 - ord_df["discount_pct"] / 100)).round(2)
    ord_df.loc[missing_amt, "order_amount"] = imputed_amt[missing_amt]

    ord_df["delivery_minutes"] = pd.to_numeric(ord_df["delivery_minutes"], errors="coerce")
    ord_df.loc[ord_df["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    comp = ord_df["status"] == "completed"
    cat_meds = ord_df[comp].groupby("category")["delivery_minutes"].median()
    for cat, med in cat_meds.items():
        fill_mask = comp & (ord_df["category"] == cat) & (ord_df["delivery_minutes"].isna())
        ord_df.loc[fill_mask, "delivery_minutes"] = med

    ord_df["order_ts"] = ord_df["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    ord_df = ord_df.drop(columns=["price_inr", "category"])

    # Persist clean CSVs and SQLite
    cust.to_csv("data/clean/customers.csv", index=False)
    prod_raw.to_csv("data/clean/products.csv", index=False)
    ord_df.to_csv("data/clean/orders.csv", index=False)

    conn = sqlite3.connect("data/instamart.db")
    cust.to_sql("customers", conn, if_exists="replace", index=False)
    prod_raw.to_sql("products", conn, if_exists="replace", index=False)
    ord_df.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

    # Validation assertions
    assert len(ord_df) == 12309
    assert (ord_df["status"] == "completed").sum() == 11110
    assert (ord_df["status"] == "cancelled").sum() == 1199
    assert (ord_df["status"] == "pending").sum() == 0
    assert ord_df["order_amount"].isna().sum() == 0
    assert ord_df[ord_df["status"] == "completed"]["delivery_minutes"].isna().sum() == 0
    assert ord_df[ord_df["status"] == "cancelled"]["delivery_minutes"].isna().sum() == 1199
    assert round(ord_df[ord_df["status"] == "completed"]["order_amount"].sum(), 2) == 2998668.80
    print("ALL CHECKPOINTS PASSED")

if __name__ == "__main__":
    run_part_a()
