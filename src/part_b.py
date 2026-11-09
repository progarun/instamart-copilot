"""Part B: Customer Segments and Lapse Prediction"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import joblib

def run_part_b():
    orders = pd.read_csv("data/clean/orders.csv")
    customers = pd.read_csv("data/clean/customers.csv")
    comp = orders[orders["status"] == "completed"].copy()
    comp["order_ts"] = pd.to_datetime(comp["order_ts"])
    snapshot = pd.to_datetime("2025-10-01 00:00:00")

    # 1. RFM Table
    rfm = comp.groupby("customer_id").agg(
        last_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
    rfm["recency"] = (snapshot - rfm["last_ts"]).dt.days
    rfm["monetary"] = rfm["monetary"].round(2)
    rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

    # 2. KMeans (K=4)
    scaler_rfm = StandardScaler()
    scaled_rfm = scaler_rfm.fit_transform(rfm[["recency", "frequency", "monetary"]])
    km4 = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm["segment_id"] = km4.fit_predict(scaled_rfm)

    mean_m = rfm.groupby("segment_id")["monetary"].mean()
    prem_id = mean_m.idxmax()
    name_map = {prem_id: "Premium"}
    other_ids = [i for i in range(4) if i != prem_id]
    sorted_f = rfm[rfm["segment_id"].isin(other_ids)].groupby("segment_id")["frequency"].mean().sort_values(ascending=False).index.tolist()
    name_map[sorted_f[0]] = "Frequent Buyers"
    name_map[sorted_f[1]] = "Occasional Buyers"
    name_map[sorted_f[2]] = "At Risk / Inactive"

    rfm["segment_name"] = rfm["segment_id"].map(name_map)
    rfm["is_premium"] = (rfm["segment_id"] == prem_id)
    rfm.to_csv("outputs/customer_segments.csv", index=False)

    # 3. Lapse modeling
    cutoff = pd.to_datetime("2025-09-01 00:00:00")
    pre = orders[pd.to_datetime(orders["order_ts"]) < cutoff].copy()
    pre_comp = pre[pre["status"] == "completed"].copy()
    pre_comp["order_ts"] = pd.to_datetime(pre_comp["order_ts"])

    cohort_custs = sorted(pre_comp["customer_id"].unique())
    post_comp = orders[(pd.to_datetime(orders["order_ts"]) >= cutoff) & 
                       (pd.to_datetime(orders["order_ts"]) <= pd.to_datetime("2025-09-30 23:59:59")) & 
                       (orders["status"] == "completed")]
    active_post = set(post_comp["customer_id"].unique())

    prod = pd.read_csv("data/clean/products.csv").set_index("product_id")["category"].to_dict()
    pre["category"] = pre["product_id"].map(prod)
    pre_comp["category"] = pre_comp["product_id"].map(prod)
    cust_meta = customers.set_index("customer_id")

    features = []
    for cid in cohort_custs:
        c_comp = pre_comp[pre_comp["customer_id"] == cid]
        c_all = pre[pre["customer_id"] == cid]
        meta = cust_meta.loc[cid]

        rec = (cutoff - c_comp["order_ts"].max()).days
        freq = len(c_comp)
        mon = c_comp["order_amount"].sum()
        avg_disc = c_comp["discount_pct"].mean()
        canc_rate = (c_all["status"] == "cancelled").sum() / len(c_all)
        avg_deliv = c_comp["delivery_minutes"].mean()
        fav_cat = c_comp["category"].mode()[0]
        tenure = (cutoff - pd.to_datetime(meta["signup_date"])).days
        lapsed = 1 if cid not in active_post else 0

        features.append({
            "customer_id": cid, "recency": rec, "frequency": freq, "monetary": mon,
            "avg_discount": avg_disc, "cancel_rate": canc_rate, "avg_delivery": avg_deliv,
            "fav_category": fav_cat, "city": meta["city"], "age": meta["age"],
            "gender": meta["gender"], "tenure": tenure, "lapsed": lapsed
        })

    df_feat = pd.DataFrame(features)
    X = df_feat.drop(columns=["customer_id", "lapsed"])
    y = df_feat["lapsed"]

    X_train, X_test, y_train, y_test, c_train, c_test = train_test_split(
        X, y, df_feat["customer_id"], test_size=0.2, stratify=y, random_state=42
    )

    num_cols = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure"]
    cat_cols = ["fav_category", "city", "gender"]

    ohe = OneHotEncoder(drop="first", sparse_output=False)
    ohe.fit(X_train[cat_cols])
    scaler = StandardScaler()
    scaler.fit(X_train[num_cols])

    def transform(df):
        return np.hstack([scaler.transform(df[num_cols]), ohe.transform(df[cat_cols])])

    X_tr_proc = transform(X_train)
    X_te_proc = transform(X_test)

    rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42)
    rf.fit(X_tr_proc, y_train)
    joblib.dump({"model": rf, "scaler": scaler, "ohe": ohe}, "outputs/lapse_model.joblib")

    # Retention List
    prem_custs = set(rfm[rfm["is_premium"]]["customer_id"])
    test_df = pd.DataFrame({"customer_id": c_test, "lapsed": y_test, "prob": rf.predict_proba(X_te_proc)[:, 1]})
    test_prem = test_df[test_df["customer_id"].isin(prem_custs)].sort_values("prob", ascending=False)
    ret_list = test_prem.head(10)[["customer_id", "prob", "lapsed"]].rename(
        columns={"prob": "lapse_probability", "lapsed": "actually_lapsed"}
    )
    ret_list.to_csv("outputs/retention_list.csv", index=False)
    print("Part B completed successfully.")

if __name__ == "__main__":
    run_part_b()
