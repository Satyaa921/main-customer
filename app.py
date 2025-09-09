import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ------------------- PAGE SETUP -------------------
st.set_page_config(page_title="Customer Type Detector", layout="centered")
st.title("Retail Customer Type Detector")

st.markdown("""
Enter customer details below to predict **customer type**  
and see a quick visualization of their profile.
""")

# ------------------- SAMPLE TRAINING DATA -------------------
# Simulated small dataset for training inside app
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

# Features and target
NUM = ["Dwell_Aisle_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend"]
CAT = ["Discount_Used","Repeat_Visitor","Age_Group","Family_Status"]
TARGET = "Purchase_Type"

X = df[NUM + CAT]
y = df[TARGET]

# ------------------- MODEL PIPELINE -------------------
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)
])
model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestClassifier(n_estimators=150, random_state=42))
])

# Train model on our mini dataset
model.fit(X, y)

# ------------------- USER INPUT FORM -------------------
st.subheader("Enter Customer Details")

c1, c2 = st.columns(2)

with c1:
    dwell = st.number_input("Minutes spent in one aisle", 0, 120, 15)
    total_dur = st.number_input("Total visit duration (minutes)", 0, 240, 45)
    items = st.number_input("Number of items purchased", 0, 20, 3)
    spend = st.number_input("Total amount spent", 0, 1000, 150, step=10)

with c2:
    disc = st.selectbox("Discount Used?", ["Yes", "No"])
    rep = st.selectbox("Repeat Visitor?", ["Yes", "No"])
    age = st.selectbox("Age Group", ["18-25", "26-35", "36-50", "51+"])
    fam = st.selectbox("Family Status", ["Bachelor", "Couple", "Family"])

# ------------------- PREDICT BUTTON -------------------
if st.button("Detect Customer Type"):
    # Create dataframe for prediction
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

    # Predict
    pred = model.predict(customer)[0]

    st.success(f"**Predicted Customer Type:** {pred}")

    # ------------------- VISUALIZATION -------------------
    st.subheader("Customer Profile Visualization")

    fig, ax = plt.subplots(figsize=(5,3))
    values = [dwell, total_dur, items, spend]
    labels = ["Aisle Time", "Total Time", "Items", "Spend"]
    ax.bar(labels, values, color=["#6baed6","#9ecae1","#c6dbef","#08519c"])
    ax.set_title("Behavior Summary")
    ax.set_ylabel("Value")
    st.pyplot(fig)

    st.markdown("### Customer Segmentation Context")
    seg_counts = df[TARGET].value_counts()
    st.bar_chart(seg_counts)

    st.info("Tip: Premium Buyers usually have **high spend and visit time**, "
            "while Low Spenders have **short visits and low spend**.")

# ------------------- FOOTER -------------------
st.markdown("---")
st.caption("Built with Streamlit • Random Forest Model • Demo Dataset")
