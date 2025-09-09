# app.py
import io
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Retail Segmentation", layout="wide")
st.title("Retail Customer Segmentation — Random Forest")

# ----- Required columns -----
NUM = ["Dwell_Aisle_Min", "Visit_Duration_Total_Min", "Items_Purchased", "Total_Spend"]
CAT = ["Discount_Used", "Repeat_Visitor", "Age_Group", "Family_Status"]
TARGET = "Purchase_Type"
REQ_COLS = NUM + CAT + [TARGET]

# ----- Sidebar: upload or use sample -----
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV with required columns", type=["csv"])

def sample_csv_bytes():
    # 12 rows, multiple samples per class (Low/Medium/High/Premium) to avoid stratify errors
    rows = [
        # Low
        [6,12,0,0,"No","No","18-25","Bachelor","Low Spender"],
        [8,18,1,25,"Yes","No","18-25","Bachelor","Low Spender"],
        [10,22,1,40,"No","Yes","26-35","Couple","Low Spender"],
        # Medium
        [14,35,2,110,"Yes","Yes","26-35","Family","Medium Buyer"],
        [16,38,2,140,"No","Yes","26-35","Family","Medium Buyer"],
        [18,40,3,120,"Yes","No","36-50","Couple","Medium Buyer"],
        # High
        [22,55,3,220,"No","Yes","36-50","Family","High Spender"],
        [24,60,4,260,"Yes","Yes","36-50","Family","High Spender"],
        [20,52,3,200,"No","No","26-35","Couple","High Spender"],
        # Premium
        [28,80,4,360,"Yes","Yes","36-50","Family","Premium Buyer"],
        [30,90,5,420,"No","Yes","51+","Family","Premium Buyer"],
        [26,75,4,340,"Yes","No","36-50","Couple","Premium Buyer"],
    ]
    df = pd.DataFrame(rows, columns=NUM + CAT + [TARGET])
    bio = io.BytesIO(); df.to_csv(bio, index=False); return bio.getvalue()

st.sidebar.download_button("Download sample CSV", sample_csv_bytes(),
                           file_name="sample_customers_labeled.csv", mime="text/csv")

# ----- Load data -----
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("No dataset uploaded. Using a built-in sample. Upload your full dataset for better results.")
    df = pd.read_csv(io.BytesIO(sample_csv_bytes()))

missing = [c for c in REQ_COLS if c not in df.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

df = df.dropna(subset=REQ_COLS).copy()

# If only one class in target, we cannot train a classifier
class_counts = df[TARGET].value_counts()
if len(class_counts) < 2:
    st.error(f"Your data has only one class in '{TARGET}'. Please upload data with at least two classes.")
    st.stop()

# ----- Split + Pipeline -----
X, y = df[NUM + CAT], df[TARGET]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)
])

pipe = Pipeline([("prep", preprocess),
                 ("rf", RandomForestClassifier(n_estimators=200, random_state=42))])

# Safe stratification: only if every class has >= 2 samples
stratify_param = y if class_counts.min() >= 2 else None
test_size = 0.25 if len(y) >= 12 else 0.33  # slightly larger test for very small datasets
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=stratify_param
)

# ----- Train & Evaluate -----
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)

m1 = accuracy_score(y_te, y_pred)
m2 = precision_score(y_te, y_pred, average="weighted", zero_division=0)
m3 = recall_score(y_te, y_pred, average="weighted", zero_division=0)
m4 = f1_score(y_te, y_pred, average="weighted", zero_division=0)

a,b,c,d = st.columns(4)
a.metric("Accuracy", f"{m1:.3f}")
b.metric("Precision (weighted)", f"{m2:.3f}")
c.metric("Recall (weighted)", f"{m3:.3f}")
d.metric("F1 (weighted)", f"{m4:.3f}")

# ----- Visuals -----
L, R = st.columns(2)
with L:
    st.subheader("Segment distribution (dataset)")
    st.bar_chart(df[TARGET].value_counts())
with R:
    st.subheader("Confusion matrix (test set)")
    cm = pd.crosstab(pd.Series(y_te, name="Actual"), pd.Series(y_pred, name="Predicted"))
    st.dataframe(cm, use_container_width=True)

st.markdown("---")

# ----- Single Prediction -----
st.subheader("Single Prediction")
c1, c2 = st.columns(2)
with c1:
    dwell = st.number_input("Dwell_Aisle_Min", 0, 120, 15)
    dur   = st.number_input("Visit_Duration_Total_Min", 0, 240, 40)
    items = st.number_input("Items_Purchased", 0, 20, 2)
    spend = st.number_input("Total_Spend", 0.0, value=120.0, step=10.0)
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

# ----- Batch Predictions -----
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

# ----- Quick EDA Snapshot -----
st.markdown("---")
st.subheader("Quick EDA Snapshot")
eda1, eda2 = st.columns(2)
with eda1:
    st.write("Numeric summary")
    st.dataframe(df[NUM].describe().T, use_container_width=True)
with eda2:
    st.write("Category counts")
    cats = {c: df[c].value_counts() for c in CAT}
    cat_tbl = pd.concat(cats, axis=1).fillna(0).astype(int)
    st.dataframe(cat_tbl, use_container_width=True)
