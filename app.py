# app.py
import io, joblib, pandas as pd, streamlit as st
st.set_page_config(page_title="🧩 Retail Segmentation", page_icon="🧩", layout="centered")

st.markdown("""<div style='text-align:center'>
<h2>🧩 Retail Customer Segmentation (Random Forest)</h2>
<p style='color:#666'>Upload a CSV or use the quick form. Clean predictions. Minimal UI. Maximum clarity.</p>
</div>""", unsafe_allow_html=True)

@st.cache_resource
def load_pipe(): return joblib.load("rf_pipeline.pkl")  # Pipeline(preprocess → RandomForest)
pipe = load_pipe()

REQ_NUM = ["Dwell_Aisle_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend"]
REQ_CAT = ["Discount_Used","Repeat_Visitor","Age_Group","Family_Status"]
REQ = REQ_NUM + REQ_CAT

with st.sidebar:
    st.header(" Input")
    up = st.file_uploader("Upload CSV", type=["csv"])
    def sample_csv():
        df = pd.DataFrame([{
            "Dwell_Aisle_Min":18,"Visit_Duration_Total_Min":42,"Items_Purchased":3,"Total_Spend":160,
            "Discount_Used":"Yes","Repeat_Visitor":"Yes","Age_Group":"26-35","Family_Status":"Family"
        }])
        bio = io.BytesIO(); df.to_csv(bio, index=False); return bio.getvalue()
    st.download_button("Download Sample CSV", sample_csv(), "sample_customers.csv", "text/csv")

with st.expander(" Quick one-row input"):
    c1, c2 = st.columns(2)
    with c1:
        d = st.number_input("Dwell_Aisle_Min", 0, 120, 15)
        vt = st.number_input("Visit_Duration_Total_Min", 0, 240, 40)
        it = st.number_input("Items_Purchased", 0, 20, 2)
        sp = st.number_input("Total_Spend", 0.0, step=10.0, value=120.0)
    with c2:
        disc = st.selectbox("Discount_Used", ["Yes","No"])
        rep  = st.selectbox("Repeat_Visitor", ["Yes","No"])
        age  = st.selectbox("Age_Group", ["18-25","26-35","36-50","51+"])
        fam  = st.selectbox("Family_Status", ["Bachelor","Couple","Family"])
    if st.button(" Predict this one"):
        one = pd.DataFrame([{"Dwell_Aisle_Min":d,"Visit_Duration_Total_Min":vt,"Items_Purchased":it,"Total_Spend":sp,
                             "Discount_Used":disc,"Repeat_Visitor":rep,"Age_Group":age,"Family_Status":fam}])
        st.success(f"Segment: **{pipe.predict(one)[0]}**")

st.markdown("---")
st.subheader(" Batch Predictions (CSV)")
if not up:
    st.info("CSV must have columns: " + ", ".join(REQ))
else:
    df = pd.read_csv(up); missing = [c for c in REQ if c not in df.columns]
    if missing: st.error("Missing columns: " + ", ".join(missing))
    else:
        out = df.copy(); out["Predicted_Segment"] = pipe.predict(df[REQ])
        st.dataframe(out.head(12), use_container_width=True)
        st.bar_chart(out["Predicted_Segment"].value_counts())
        st.download_button("Download Results", out.to_csv(index=False).encode("utf-8"),
                           "segmented_predictions.csv", "text/csv")

st.caption("Tip: This RF pipeline uses dwell, duration, items, spend + a few categorical signals. Keep inputs tidy for best results.")
