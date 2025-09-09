import io
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)

# ---------------- Page ----------------
st.set_page_config(page_title="Retail Segmentation", layout="wide")
st.title("Retail Customer Segmentation — Random Forest")

# ---------------- Constants ----------------
NUM = ["Dwell_Aisle_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend"]
CAT = ["Discount_Used","Repeat_Visitor","Age_Group","Family_Status"]
TARGET = "Purchase_Type"
REQ_COLS = NUM + CAT + [TARGET]

# ---------------- Sidebar: upload dataset ----------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV with required columns", type=["csv"])

def sample_csv_bytes():
    df = pd.DataFrame([
        {"Dwell_Aisle_Min":18,"Visit_Duration_Total_Min":42,"Items_Purchased":3,"Total_Spend":160,
         "Discount_Used":"Yes","Repeat_Visitor":"Yes","Age_Group":"26-35","Family_Status":"Family",
         "Purchase_Type":"High Spender"},
        {"Dwell_Aisle_Min":6,"Visit_Duration_Total_Min":12,"Items_Purchased":0,"Total_Spend":0,
         "Discount_Used":"No","Repeat_Visitor":"No","Age_Group":"18-25","Family_Status":"Bachelor",
         "Purchase_Type":"Low Spender"}
    ])
    buf = io.BytesIO(); df.to_csv(buf, index=False); return buf.getvalue()

st.sidebar.download_button("Download sample CSV", sample_csv_bytes(),
                           file_name="sample_customers_labeled.csv", mime="text/csv")

# ---------------- Load data ----------------
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("No dataset uploaded. Using a tiny built-in sample. Upload your full dataset for better results.")
    df = pd.read_csv(io.BytesIO(sample_csv_bytes()))

missing = [c for c in REQ_COLS if c not in df.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

# Clean
df = df.dropna(subset=REQ_COLS).copy()

# ---------------- Split + Pipeline ----------------
X = df[NUM + CAT]
y = df[TARGET]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)  # default sparse=True is fine
])

rf = RandomForestClassifier(n_estimators=200, random_state=42)
pipe = Pipeline([("prep", preprocess), ("rf", rf)])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y if len(y.unique())>1 else None
)

# ---------------- Train ----------------
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)

# ---------------- Metrics ----------------
colA, colB, colC, colD = st.columns(4)
colA.metric("Accuracy", f"{accuracy_score(y_te, y_pred):.3f}")
colB.metric("Precision (weighted)", f"{precision_score(y_te, y_pred, average='weighted', zero_division=0):.3f}")
colC.metric("Recall (weighted)", f"{recall_score(y_te, y_pred, average='weighted', zero_division=0):.3f}")
colD.metric("F1 (weighted)", f"{f1_score(y_te, y_pred, average='weighted', zero_division=0):.3f}")

# ---------------- Visuals ----------------
left, right = st.columns(2)
with left:
    st.subheader("Segment distribution (dataset)")
    st.bar_chart(df[TARGET].value_counts())
with right:
    st.subheader("Confusion matrix (test set)")
    cm = pd.crosstab(pd.Series(y_te, name="Actual"),
                     pd.Series(y_pred, name="Predicted"))
    st.dataframe(cm, use_container_width=True)

st.markdown("---")

# ---------------- Quick one-row prediction ----------------
st.subheader("Single Prediction")
c1, c2 = st.columns(2)
with c1:
    dwell = st.number_input("Dwell_Aisle_Min", min_value=0, max_value=120, value=15)
    dur   = st.number_input("Visit_Duration_Total_Min", min_value=0, max_value=240, value=40)
    items = st.number_input("Items_Purchased", min_value=0, max_value=20, value=2)
    spend = st.number_input("Total_Spend", min_value=0.0, value=120.0, step=10.0)
with c2:
    disc = st.selectbox("Discount_Used", ["Yes","No"])
    rep  = st.selectbox("Repeat_Visitor", ["Yes","No"])
    age  = st.selectbox("Age_Group", ["18-25","26-35","36-50","51+"])
    fam  = st.selectbox("Family_Status", ["Bachelor","Couple","Family"])

if st.button("Predict this one"):
    one = pd.DataFrame([{
        "Dwell_Aisle_Min": dwell, "Visit_Duration_Total_Min": dur,
        "Items_Purchased": items, "Total_Spend": spend,
        "Discount_Used": disc, "Repeat_Visitor": rep, "Age_Group": age, "Family_Status": fam
    }])
    st.success(f"Predicted Segment: {pipe.predict(one)[0]}")

# ---------------- Batch predictions ----------------
st.subheader("Batch Predictions")
up2 = st.file_uploader("Upload CSV (unlabeled) for batch prediction", type=["csv"], key="batch")
if up2:
    df_infer = pd.read_csv(up2)
    need = NUM + CAT
    miss2 = [c for c in need if c not in df_infer.columns]
    if miss2:
        st.error("Missing columns: " + ", ".join(miss2))
    else:
        preds = pipe.predict(df_infer[need])
        out = df_infer.copy()
        out["Predicted_Segment"] = preds
        st.write("Preview")
        st.dataframe(out.head(12), use_container_width=True)
        st.write("Predicted segment distribution")
        st.bar_chart(out["Predicted_Segment"].value_counts())
        st.download_button("Download results",
                           data=out.to_csv(index=False).encode("utf-8"),
                           file_name="segmented_predictions.csv", mime="text/csv")

# ---------------- EDA snapshot (optional but helpful) ----------------
st.markdown("---")
st.subheader("Quick EDA Snapshot")
eda1, eda2 = st.columns(2)
with eda1:
    st.write("Numeric summary")
    st.dataframe(df[NUM].describe().T, use_container_width=True)
with eda2:
    st.write("Category counts")
    cats = {c: df[c].value_counts() for c in CAT}
    # join counts side-by-side for compact view
    cat_tbl = pd.concat(cats, axis=1).fillna(0).astype(int)
    st.dataframe(cat_tbl, use_container_width=True)
