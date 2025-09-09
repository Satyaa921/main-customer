# app.py
import io
import joblib
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Retail Segmentation", layout="centered")
st.markdown(
    "<h2 style='text-align:center'>Retail Customer Segmentation (Random Forest)</h2>"
    "<p style='text-align:center;color:#666'>Upload a CSV or use the quick form to get segment predictions.</p>",
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    return joblib.load("rf_pipeline.pkl")  # Pipeline(preprocess→RF) or bare RF
pipe = load_model()

# Required columns
REQ_NUM = ["Dwell_Aisle_Min", "Visit_Duration_Total_Min", "Items_Purchased", "Total_Spend"]
REQ_CAT = ["Discount_Used", "Repeat_Visitor", "Age_Group", "Family_Status"]
REQ = REQ_NUM + REQ_CAT

# Fallback encodings if a bare RF was saved
ENC = {
    "Discount_Used": {"No": 0, "Yes": 1},
    "Repeat_Visitor": {"No": 0, "Yes": 1},
    "Age_Group": {"18-25": 0, "26-35": 1, "36-50": 2, "51+": 3},
    "Family_Status": {"Bachelor": 0, "Couple": 1, "Family": 2},
}

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Return inputs ready for .predict() for either a Pipeline or a bare RF."""
    if isinstance(pipe, Pipeline):
        return df[REQ]
    out = df[REQ].copy()
    for c in REQ_CAT:
        out[c] = out[c].map(ENC[c])
    return out

# Sidebar: upload + sample
with st.sidebar:
    st.header("Input")
    up = st.file_uploader("Upload CSV", type=["csv"])

    def sample_csv():
        df = pd.DataFrame([{
            "Dwell_Aisle_Min": 18,
            "Visit_Duration_Total_Min": 42,
            "Items_Purchased": 3,
            "Total_Spend": 160,
            "Discount_Used": "Yes",
            "Repeat_Visitor": "Yes",
            "Age_Group": "26-35",
            "Family_Status": "Family",
        }])
        bio = io.BytesIO()
        df.to_csv(bio, index=False)
        return bio.getvalue()

    st.download_button("Download Sample CSV", data=sample_csv(),
                       file_name="sample_customers.csv", mime="text/csv")

# Quick one-row form
with st.expander("Quick one-row input"):
    c1, c2 = st.columns(2)
    with c1:
        dwell = st.number_input("Dwell_Aisle_Min", min_value=0, max_value=120, value=15)
        dur = st.number_input("Visit_Duration_Total_Min", min_value=0, max_value=240, value=40)
        items = st.number_input("Items_Purchased", min_value=0, max_value=20, value=2)
        spend = st.number_input("Total_Spend", min_value=0.0, value=120.0, step=10.0)
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
            "Family_Status": fam,
        }])
        pred = pipe.predict(prepare(one))[0]
        st.success(f"Predicted Segment: {pred}")

st.markdown("---")

# Batch predictions
st.subheader("Batch Predictions (CSV)")
if not up:
    st.info("CSV must have columns: " + ", ".join(REQ))
else:
    df = pd.read_csv(up)
    missing = [c for c in REQ if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
    else:
        out = df.copy()
        out["Predicted_Segment"] = pipe.predict(prepare(df))
        st.write("Preview")
        st.dataframe(out.head(12), use_container_width=True)
        st.write("Segment distribution")
        st.bar_chart(out["Predicted_Segment"].value_counts())
        st.download_button(
            "Download Results",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="segmented_predictions.csv",
            mime="text/csv",
        )

st.caption("This app expects the eight required input columns and a saved rf_pipeline.pkl.")
