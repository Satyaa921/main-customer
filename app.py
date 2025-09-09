import io
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Retail Segmentation", layout="wide")
st.title("Retail Customer Segmentation — Random Forest")

# ---------------- REQUIRED COLUMNS ----------------
NUM = ["Dwell_Aisle_Min", "Visit_Duration_Total_Min", "Items_Purchased", "Total_Spend"]
CAT = ["Discount_Used", "Repeat_Visitor", "Age_Group", "Family_Status"]
TARGET = "Purchase_Type"
REQ_COLS = NUM + CAT + [TARGET]

# ---------------- SIDEBAR ----------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV with required columns", type=["csv"])

def sample_csv_bytes():
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
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    return bio.getvalue()

st.sidebar.download_button("Download sample CSV",
                           data=sample_csv_bytes(),
                           file_name="sample_customers_labeled.csv",
                           mime="text/csv")

# ---------------- LOAD DATA ----------------
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.info("No dataset uploaded. Using a built-in sample. Upload your full dataset for better results.")
    df = pd.read_csv(io.BytesIO(sample_csv_bytes()))

# Check required columns
missing = [c for c in REQ_COLS if c not in df.columns]
if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

df = df.dropna(subset=REQ_COLS).copy()

# If only one class in target, stop
if len(df[TARGET].unique()) < 2:
    st.error(f"Your data must contain at least two classes in '{TARGET}'.")
    st.stop()

# ---------------- FEATURES AND TARGET ----------------
X = df[NUM + CAT]
y = df[TARGET]

# ---------------- SAFE SPLIT ----------------
TEST_SIZE = 0.25 if len(y) >= 40 else 0.33
try:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
except ValueError:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42
    )

# ---------------- PIPELINE ----------------
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)
])

pipe = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# ---------------- TRAIN ----------------
pipe.fit(X_tr, y_tr)
y_pred = pipe.predict(X_te)

# ---------------- METRICS ----------------
a, b, c, d = st.columns(4)
a.metric("Accuracy", f"{accuracy_score(y_te, y_pred):.3f}")
b.metric("Precision (weighted)", f"{precision_score(y_te, y_pred, average='weighted', zero_division=0):.3f}")
c.metric("Recall (weighted)", f"{recall_score(y_te, y_pred, average='weighted', zero_division=0):.3f}")
d.metric("F1 (weighted)", f"{f1_score(y_te, y_pred, average='weighted', zero_division=0):.3f}")

# ---------------- VISUALS ----------------
L, R = st.columns(2)
with L:
    st.subheader("Segment distribution (dataset)")
    st.bar_chart(df[TARGET].value_counts())
with R:
    st.subheader("Confusion matrix (test set)")
    cm = pd.crosstab(pd.Series(y_te, name="Actual"),
                     pd.Series(y_pred, name="Predicted"))
    st.dataframe(cm, use_container_width=True)

st.markdown("---")

# ---------------- SINGLE PREDICTION ----------------
st.subheader("Single Prediction")
c1, c2 = st.columns(2)
with c1:
    dwell = st.number_input("Dwell_Aisle_Min", 0, 120, 15)
    dur = st.number_input("Visit_Duration_Total_Min", 0, 240, 40)
    items = st.number_input("Items_Purchased", 0, 20, 2)
    spend = st.number_input("Total_Spend", 0.0, value=120.0, step=10.0)
with c2:
    disc = st.selectbox("Discount_Used", ["Yes", "No"])
    rep = st.selectbox("Repeat_Visitor", ["Yes", "No"])
    age = st.selectbox("Age_Group", ["18-25", "26-35", "36-50", "51+"])
    fam = st.selectbox("Family_Status", ["Bachelor", "Couple", "Family"])

if st.button("Predict this one"):
    one = pd.DataFrame([{
        "Dwell_Aisle_Min": dwell,
        "Visit_Duration_Total_Min": dur,
        "Items_Purchased": items,
        "Total_Spend": spend,
        "Discount_Used": disc,
        "Repeat_Visitor": rep,
        "Age_Group": age,
        "Family_Status": fam
    }])
    st.success(f"Predicted Segment: {pipe.predict(one)[0]}")

# ---------------- BATCH PREDICTIONS ----------------
st.subheader("Batch Predictions")
up2 = st.file_uploader("Upload CSV (unlabeled) for batch prediction", type=["csv"], key="batch")
if up2:
    df_infer = pd.read_csv(up2)
    miss2 = [c for c in NUM + CAT if c not in df_infer.columns]
    if miss2:
        st.error("Missing columns: " + ", ".join(miss2))
    else:
        preds = pipe.predict(df_infer[NUM + CAT])
        out = df_infer.copy()
        out["Predicted_Segment"] = preds
        st.write("Preview")
        st.dataframe(out.head(12), use_container_width=True)
        st.write("Predicted segment distribution")
        st.bar_chart(out["Predicted_Segment"].value_counts())
        st.download_button("Download results",
                           data=out.to_csv(index=False).encode("utf-8"),
                           file_name="segmented_predictions.csv",
                           mime="text/csv")

# ---------------- QUICK EDA SNAPSHOT ----------------
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
