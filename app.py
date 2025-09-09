import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Customer Type Detector", layout="centered")
st.markdown("<h1 style='text-align:center; background: -webkit-linear-gradient(left, #00b4d8, #0077b6);"
            "color:white; padding:10px;'>Retail Customer Intelligence Dashboard</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; font-size:18px; color:#555;'>
Enter customer details to predict their type and visualize their behavior <br>
in an interactive, modern way.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- SAMPLE TRAINING DATA ----------------
data = [
    [6,12,0,0,"No","No","18-25","Bachelor","Low Spender"],
    [8,18,1,25,"Yes","No","18-25","Bachelor","Low Spender"],
    [14,35,2,110,"Yes","Yes","26-35","Family","Medium Buyer"],
    [18,40,3,120,"No","Yes","26-35","Family","Medium Buyer"],
    [22,55,3,220,"Yes","No","36-50","Couple","High Spender"],
    [24,60,4,260,"No","Yes","36-50","Couple","High Spender"],
    [30,90,5,420,"Yes","Yes","51+","Family","Premium Buyer"],
    [28,85,4,350,"No","Yes","36-50","Family","Premium Buyer"],
]
columns = ["Dwell_Aisle_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend",
           "Discount_Used","Repeat_Visitor","Age_Group","Family_Status","Purchase_Type"]
df = pd.DataFrame(data, columns=columns)

# Feature columns
NUM = ["Dwell_Aisle_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend"]
CAT = ["Discount_Used","Repeat_Visitor","Age_Group","Family_Status"]
TARGET = "Purchase_Type"

X = df[NUM + CAT]
y = df[TARGET]

# ---------------- MODEL PIPELINE ----------------
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)
])

model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train model
model.fit(X, y)

# ---------------- USER INPUT FORM ----------------
st.subheader("Enter Customer Details")

c1, c2 = st.columns(2)
with c1:
    dwell = st.slider("Minutes spent in one aisle", 0, 120, 15)
    total_dur = st.slider("Total visit duration (minutes)", 0, 240, 45)
    items = st.number_input("Number of items purchased", 0, 20, 3)
    spend = st.number_input("Total amount spent", 0, 1000, 150, step=10)
with c2:
    disc = st.selectbox("Discount Used?", ["Yes", "No"])
    rep = st.selectbox("Repeat Visitor?", ["Yes", "No"])
    age = st.selectbox("Age Group", ["18-25", "26-35", "36-50", "51+"])
    fam = st.selectbox("Family Status", ["Bachelor", "Couple", "Family"])

st.markdown("---")

# ---------------- PREDICT BUTTON ----------------
if st.button("Detect Customer Type"):
    customer = pd.DataFrame([{
        "Dwell_Aisle_Min": dwell,
        "Visit_Duration_Total_Min": total_dur,
        "Items_Purchased": items,
        "Total_Spend": spend,
        "Discount_Used": disc,
        "Repeat_Visitor": rep,
        "Age_Group": age,
        "Family_Status": fam
    }])

    # Prediction
    proba = model.predict_proba(customer)[0]
    pred = model.predict(customer)[0]
    pred_prob = np.max(proba)

    # ---------------- PROFILE CARD ----------------
    st.markdown(f"""
    <div style='background-color:#f0f8ff; border-radius:10px; padding:20px; text-align:center;'>
        <h2 style='color:#0077b6;'>Predicted Customer Type</h2>
        <h1 style='color:#023e8a;'>{pred}</h1>
        <p style='font-size:18px; color:#555;'>Confidence Level: <b>{pred_prob*100:.1f}%</b></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- VISUAL 1: RADAR CHART ----------------
    avg_profile = df[NUM].mean().values
    user_profile = [dwell, total_dur, items, spend]

    radar_categories = NUM
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=user_profile,
        theta=radar_categories,
        fill='toself',
        name='Current Customer',
        line=dict(color='blue')
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=avg_profile,
        theta=radar_categories,
        fill='toself',
        name='Average Customer',
        line=dict(color='orange')
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="Customer Behavior vs Average"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ---------------- VISUAL 2: CONFIDENCE GAUGE ----------------
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_prob*100,
        title={'text': "Prediction Confidence (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 50], 'color': "#ffcccc"},
                   {'range': [50, 80], 'color': "#ffe680"},
                   {'range': [80, 100], 'color': "#ccffcc"}
               ]}
    ))
    st.plotly_chart(gauge_fig, use_container_width=True)

    # ---------------- VISUAL 3: SEGMENT DISTRIBUTION ----------------
    seg_counts = df[TARGET].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]
    pie_fig = px.pie(seg_counts, values="Count", names="Segment",
                     title="Dataset-Wide Segment Distribution",
                     color_discrete_sequence=px.colors.sequential.Blues)
    st.plotly_chart(pie_fig, use_container_width=True)

    # ---------------- INTERPRETATION ----------------
    st.markdown("### Insight")
    if pred == "Premium Buyer":
        st.info("💡 **Premium Buyers** usually have **high spend and longer visit times**. "
                "Consider offering exclusive loyalty rewards to retain them.")
    elif pred == "High Spender":
        st.info("💡 **High Spenders** value deals and may respond well to targeted promotions.")
    elif pred == "Medium Buyer":
        st.info("💡 **Medium Buyers** can be upsold by bundling products or offering time-limited discounts.")
    else:
        st.info("💡 **Low Spenders** may need personalized nudges to increase engagement, "
                "like welcome offers or free shipping.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit • Plotly • Random Forest Model • Demo Dataset")
