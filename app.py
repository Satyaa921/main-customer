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
st.caption("Simple predictions with creative inputs designed for product & business teams.")

# ---------------- TRAIN A TINY MODEL IN-APP (keeps deployment simple) ----------------
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
cols = ["Dwell_Aisle_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend",
        "Discount_Used","Repeat_Visitor","Age_Group","Family_Status","Purchase_Type"]
df = pd.DataFrame(data, columns=cols)

NUM = ["Dwell_Aisle_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend"]
CAT = ["Discount_Used","Repeat_Visitor","Age_Group","Family_Status"]
TARGET = "Purchase_Type"

X, y = df[NUM + CAT], df[TARGET]
preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
])
model = Pipeline([("prep", preprocess),
                  ("rf", RandomForestClassifier(n_estimators=150, random_state=42))])
model.fit(X, y)

# ---------------- HELPERS ----------------
def predict_and_show(customer_row: pd.DataFrame, title="Customer Behavior"):
    pred = model.predict(customer_row)[0]
    st.success(f"Predicted Customer Type: {pred}")

    # Visual 1: customer metrics
    fig, ax = plt.subplots(figsize=(5,3))
    metrics = [
        customer_row.iloc[0]["Dwell_Aisle_Min"],
        customer_row.iloc[0]["Visit_Duration_Total_Min"],
        customer_row.iloc[0]["Items_Purchased"],
        customer_row.iloc[0]["Total_Spend"],
    ]
    labels = ["Aisle Time", "Total Visit", "Items", "Spend"]
    ax.bar(labels, metrics, color=["#4c72b0","#55a868","#c44e52","#8172b2"])
    ax.set_ylabel("Value"); ax.set_title(title)
    st.pyplot(fig)

    # Visual 2: dataset segment distribution (context)
    st.write("Overall Customer Segment Distribution")
    seg_counts = df[TARGET].value_counts()
    fig2, ax2 = plt.subplots(figsize=(5,3))
    seg_counts.plot(kind='bar', ax=ax2, color="#4c72b0")
    ax2.set_ylabel("Customers"); ax2.set_xlabel("Type")
    st.pyplot(fig2)

# Default base values (used by all modes)
BASE = {
    "Dwell_Aisle_Min": 15,
    "Visit_Duration_Total_Min": 45,
    "Items_Purchased": 2,
    "Total_Spend": 120,
    "Discount_Used": "No",
    "Repeat_Visitor": "No",
    "Age_Group": "26-35",
    "Family_Status": "Bachelor",
}

# Preset personas (creative but simple starting points)
PRESETS = {
    "Window Shopper": {
        "Dwell_Aisle_Min": 8, "Visit_Duration_Total_Min": 18, "Items_Purchased": 0, "Total_Spend": 0,
        "Discount_Used": "No", "Repeat_Visitor": "No", "Age_Group": "18-25", "Family_Status": "Bachelor",
    },
    "Value Seeker": {
        "Dwell_Aisle_Min": 16, "Visit_Duration_Total_Min": 38, "Items_Purchased": 2, "Total_Spend": 110,
        "Discount_Used": "Yes", "Repeat_Visitor": "Yes", "Age_Group": "26-35", "Family_Status": "Family",
    },
    "Loyal Medium": {
        "Dwell_Aisle_Min": 18, "Visit_Duration_Total_Min": 42, "Items_Purchased": 3, "Total_Spend": 150,
        "Discount_Used": "No", "Repeat_Visitor": "Yes", "Age_Group": "36-50", "Family_Status": "Couple",
    },
    "Premium Family": {
        "Dwell_Aisle_Min": 28, "Visit_Duration_Total_Min": 85, "Items_Purchased": 4, "Total_Spend": 360,
        "Discount_Used": "Yes", "Repeat_Visitor": "Yes", "Age_Group": "36-50", "Family_Status": "Family",
    },
}

# ---------------- INPUT MODES ----------------
st.subheader("Input Modes")
tab1, tab2, tab3 = st.tabs(["Quick Form", "Persona Presets", "What-If Compare"])

# ---- Tab 1: Quick Form ----
with tab1:
    st.write("Enter details directly.")
    c1, c2 = st.columns(2)
    with c1:
        dwell = st.slider("Minutes in one aisle", 0, 120, BASE["Dwell_Aisle_Min"])
        total_dur = st.slider("Total visit duration (min)", 0, 240, BASE["Visit_Duration_Total_Min"])
        items = st.number_input("Items purchased", 0, 20, BASE["Items_Purchased"])
        spend = st.number_input("Total spend", 0, 1000, BASE["Total_Spend"], step=10)
    with c2:
        disc = st.selectbox("Discount used", ["Yes","No"], index=0 if BASE["Discount_Used"]=="Yes" else 1)
        rep  = st.selectbox("Repeat visitor", ["Yes","No"], index=0 if BASE["Repeat_Visitor"]=="Yes" else 1)
        age  = st.selectbox("Age group", ["18-25","26-35","36-50","51+"], index=["18-25","26-35","36-50","51+"].index(BASE["Age_Group"]))
        fam  = st.selectbox("Family status", ["Bachelor","Couple","Family"], index=["Bachelor","Couple","Family"].index(BASE["Family_Status"]))

    customer1 = pd.DataFrame([{
        "Dwell_Aisle_Min": dwell,
        "Visit_Duration_Total_Min": total_dur,
        "Items_Purchased": items,
        "Total_Spend": spend,
        "Discount_Used": disc,
        "Repeat_Visitor": rep,
        "Age_Group": age,
        "Family_Status": fam
    }])

    st.write("Input preview")
    st.dataframe(customer1, use_container_width=True)

    if st.button("Predict (Quick Form)"):
        predict_and_show(customer1, title="Current Customer Behavior")

# ---- Tab 2: Persona Presets ----
with tab2:
    st.write("Pick a preset and optionally tweak values.")
    preset_name = st.selectbox("Preset persona", list(PRESETS.keys()))
    preset = PRESETS[preset_name]

    c1, c2 = st.columns(2)
    with c1:
        dwell2 = st.slider("Minutes in one aisle", 0, 120, preset["Dwell_Aisle_Min"])
        total_dur2 = st.slider("Total visit duration (min)", 0, 240, preset["Visit_Duration_Total_Min"])
        items2 = st.number_input("Items purchased", 0, 20, preset["Items_Purchased"], key="items2")
        spend2 = st.number_input("Total spend", 0, 1000, preset["Total_Spend"], step=10, key="spend2")
    with c2:
        disc2 = st.selectbox("Discount used", ["Yes","No"], index=0 if preset["Discount_Used"]=="Yes" else 1, key="disc2")
        rep2  = st.selectbox("Repeat visitor", ["Yes","No"], index=0 if preset["Repeat_Visitor"]=="Yes" else 1, key="rep2")
        age2  = st.selectbox("Age group", ["18-25","26-35","36-50","51+"], index=["18-25","26-35","36-50","51+"].index(preset["Age_Group"]), key="age2")
        fam2  = st.selectbox("Family status", ["Bachelor","Couple","Family"], index=["Bachelor","Couple","Family"].index(preset["Family_Status"]), key="fam2")

    customer2 = pd.DataFrame([{
        "Dwell_Aisle_Min": dwell2,
        "Visit_Duration_Total_Min": total_dur2,
        "Items_Purchased": items2,
        "Total_Spend": spend2,
        "Discount_Used": disc2,
        "Repeat_Visitor": rep2,
        "Age_Group": age2,
        "Family_Status": fam2
    }])

    st.write("Input preview")
    st.dataframe(customer2, use_container_width=True)

    if st.button("Predict (Preset)"):
        predict_and_show(customer2, title=f"{preset_name} (tweaked) Behavior")

# ---- Tab 3: What-If Compare ----
with tab3:
    st.write("Compare two scenarios side by side to see how type changes.")
    left, right = st.columns(2)

    # Scenario A
    with left:
        st.markdown("Scenario A")
        a_dwell = st.slider("A: Minutes in aisle", 0, 120, 12)
        a_dur   = st.slider("A: Total duration", 0, 240, 30)
        a_items = st.number_input("A: Items", 0, 20, 1, key="a_items")
        a_spend = st.number_input("A: Spend", 0, 1000, 40, step=10, key="a_spend")
        a_disc  = st.selectbox("A: Discount", ["Yes","No"], index=1, key="a_disc")
        a_rep   = st.selectbox("A: Repeat", ["Yes","No"], index=1, key="a_rep")
        a_age   = st.selectbox("A: Age", ["18-25","26-35","36-50","51+"], index=1, key="a_age")
        a_fam   = st.selectbox("A: Family", ["Bachelor","Couple","Family"], index=0, key="a_fam")

    # Scenario B
    with right:
        st.markdown("Scenario B")
        b_dwell = st.slider("B: Minutes in aisle", 0, 120, 28)
        b_dur   = st.slider("B: Total duration", 0, 240, 85)
        b_items = st.number_input("B: Items", 0, 20, 4, key="b_items")
        b_spend = st.number_input("B: Spend", 0, 1000, 360, step=10, key="b_spend")
        b_disc  = st.selectbox("B: Discount", ["Yes","No"], index=0, key="b_disc")
        b_rep   = st.selectbox("B: Repeat", ["Yes","No"], index=0, key="b_rep")
        b_age   = st.selectbox("B: Age", ["18-25","26-35","36-50","51+"], index=2, key="b_age")
        b_fam   = st.selectbox("B: Family", ["Bachelor","Couple","Family"], index=2, key="b_fam")

    if st.button("Compare Scenarios"):
        A = pd.DataFrame([{
            "Dwell_Aisle_Min": a_dwell, "Visit_Duration_Total_Min": a_dur,
            "Items_Purchased": a_items, "Total_Spend": a_spend,
            "Discount_Used": a_disc, "Repeat_Visitor": a_rep, "Age_Group": a_age, "Family_Status": a_fam
        }])
        B = pd.DataFrame([{
            "Dwell_Aisle_Min": b_dwell, "Visit_Duration_Total_Min": b_dur,
            "Items_Purchased": b_items, "Total_Spend": b_spend,
            "Discount_Used": b_disc, "Repeat_Visitor": b_rep, "Age_Group": b_age, "Family_Status": b_fam
        }])

        predA = model.predict(A)[0]
        predB = model.predict(B)[0]

        st.success(f"Scenario A → {predA}")
        st.success(f"Scenario B → {predB}")

        # Side-by-side visuals
        vA, vB = st.columns(2)
        with vA:
            figA, axA = plt.subplots(figsize=(4,3))
            axA.bar(["Aisle","Visit","Items","Spend"], [a_dwell,a_dur,a_items,a_spend], color="#55a868")
            axA.set_title("Scenario A"); st.pyplot(figA)
        with vB:
            figB, axB = plt.subplots(figsize=(4,3))
            axB.bar(["Aisle","Visit","Items","Spend"], [b_dwell,b_dur,b_items,b_spend], color="#4c72b0")
            axB.set_title("Scenario B"); st.pyplot(figB)

# Footer
st.markdown("---")
st.caption("Three input modes: Quick Form, Persona Presets, What-If Compare. Designed for clarity and business impact.")
