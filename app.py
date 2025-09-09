import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="Customer Type Detector", layout="centered")
st.title("Retail Customer Type Detector")
st.markdown("Enter customer details to **predict customer type** and view simple insights for business decisions.")

st.markdown("---")

# ---------------- SAMPLE DATA FOR TRAINING ----------------
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

NUM = ["Dwell_Aisle_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend"]
CAT = ["Discount_Used","Repeat_Visitor","Age_Group","Family_Status"]
TARGET = "Purchase_Type"

# Model setup
X = df[NUM + CAT]
y = df[TARGET]
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)
])
model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestClassifier(n_estimators=150, random_state=42))
])
model.fit(X, y)

# ---------------- USER INPUT ----------------
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

# ---------------- PREDICTION ----------------
if st.button("Detect Customer Type"):
    # Prepare input
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

    st.markdown("---")
    st.subheader("Customer Behavior Summary")
    
    # ----------- VISUAL 1: CUSTOMER METRICS -----------
    fig, ax = plt.subplots(figsize=(5,3))
    metrics = [dwell, total_dur, items, spend]
    labels = ["Aisle Time", "Total Visit", "Items", "Spend"]
    ax.bar(labels, metrics, color=["#4c72b0","#55a868","#c44e52","#8172b2"])
    ax.set_ylabel("Value")
    ax.set_title("Current Customer Behavior")
    st.pyplot(fig)

    # ----------- VISUAL 2: DATASET SEGMENT DISTRIBUTION -----------
    st.subheader("Overall Customer Segment Distribution")
    seg_counts = df[TARGET].value_counts()
    fig2, ax2 = plt.subplots(figsize=(5,3))
    seg_counts.plot(kind='bar', ax=ax2, color="#4c72b0")
    ax2.set_ylabel("Number of Customers")
    ax2.set_xlabel("Customer Type")
    ax2.set_title("Distribution of All Customer Types")
    st.pyplot(fig2)

    # ----------- BUSINESS INTERPRETATION -----------
    st.markdown("### Business Insight")
    if pred == "Premium Buyer":
        st.info("Premium Buyers are highly engaged and spend a lot. "
                "Consider offering **exclusive loyalty rewards** to retain them.")
    elif pred == "High Spender":
        st.info("High Spenders value deals and can be targeted with **personalized promotions**.")
    elif pred == "Medium Buyer":
        st.info("Medium Buyers can be converted to high spenders by **bundling products** "
                "or **time-limited discounts**.")
    else:
        st.info("Low Spenders may need **basic engagement campaigns**, such as welcome offers or free shipping.")

st.markdown("---")
st.caption("Built with Streamlit • Random Forest • Simple Visual Insights")
