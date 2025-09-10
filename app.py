import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# =================== PAGE SETUP ===================
st.set_page_config(page_title="Dubai Retail – Customer Type & Insights", layout="centered")
st.title("Dubai Retail — Customer Type & Store Insights")
st.caption("Enter a shopper’s details → get type, benchmarks, and simple visuals managers can act on.")

st.markdown("---")

# =================== DEMO TRAINING DATA (DUBAI-FLAVORED) ===================
# Each row ≈ a visit. Small, curated set that reflects Dubai context.
rows = [
    # dwell, cashier, total, items, spend, disc, repeat, age, fam, resident, nationality, gender, timeband, day, section, type
    [ 8, 2, 18, 0,   0, "No","No","18-25","Bachelor","Tourist","India","Male","Afternoon","Tuesday","Accessories","Low Spender"],
    [10, 3, 24, 1,  35, "Yes","No","18-25","Bachelor","Resident","UAE","Male","Evening","Monday","Men's Wear","Low Spender"],
    [16, 4, 38, 2, 110, "Yes","Yes","26-35","Family","Resident","Philippines","Female","Evening","Thursday","Women’s Wear","Medium Buyer"],
    [18, 5, 42, 3, 140, "No","Yes","26-35","Family","Resident","India","Female","Night","Friday","Women’s Wear","Medium Buyer"],
    [20, 4, 55, 3, 220, "Yes","No","36-50","Couple","Resident","Pakistan","Male","Afternoon","Saturday","Men's Wear","High Spender"],
    [22, 6, 60, 4, 260, "No","Yes","36-50","Couple","Resident","UAE","Male","Evening","Saturday","Women’s Wear","High Spender"],
    [28, 7, 85, 4, 350, "Yes","Yes","36-50","Family","Tourist","Saudi Arabia","Female","Night","Thursday","Women’s Wear","Premium Buyer"],
    [30, 8, 90, 5, 420, "Yes","Yes","51+","Family","Tourist","UK","Female","Night","Friday","Accessories","Premium Buyer"],
    [14, 3, 32, 2, 120, "No","Yes","26-35","Couple","Resident","Egypt","Male","Morning","Sunday","Kids","Medium Buyer"],
    [25, 4, 58, 3, 230, "Yes","Yes","36-50","Family","Resident","India","Female","Evening","Sunday","Women’s Wear","High Spender"],
    [ 6, 1, 15, 0,   0, "No","No","18-25","Bachelor","Tourist","Pakistan","Male","Morning","Wednesday","Kids","Low Spender"],
    [12, 2, 28, 1,  60, "Yes","No","26-35","Couple","Resident","Philippines","Female","Afternoon","Tuesday","Accessories","Low Spender"],
    [22, 5, 62, 3, 240, "Yes","Yes","36-50","Couple","Resident","UAE","Female","Evening","Friday","Men's Wear","High Spender"],
    [26, 6, 76, 4, 320, "No","Yes","36-50","Family","Resident","India","Male","Evening","Thursday","Women’s Wear","High Spender"],
]
COLS = ["Dwell_Aisle_Min","Cashier_Time_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend",
        "Discount_Used","Repeat_Visitor","Age_Group","Family_Status","Resident_Type","Nationality",
        "Gender","Time_Band","Day_of_Week","Section","Purchase_Type"]
df_demo = pd.DataFrame(rows, columns=COLS)

# =================== MODEL (TRAINED IN-APP) ===================
NUM = ["Dwell_Aisle_Min","Cashier_Time_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend"]
CAT = ["Discount_Used","Repeat_Visitor","Age_Group","Family_Status","Resident_Type",
       "Nationality","Gender","Time_Band","Day_of_Week","Section"]
TARGET = "Purchase_Type"

X = df_demo[NUM + CAT]
y = df_demo[TARGET]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)
])
model = Pipeline([
    ("prep", preprocess),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42))
])
model.fit(X, y)

# =================== INPUT UI (SIMPLE & CREATIVE) ===================
st.subheader("Enter Customer Details")

NATIONS = ["UAE","India","Pakistan","Philippines","Saudi Arabia","Egypt","UK"]
TIME_BANDS = ["Morning","Afternoon","Evening","Night"]
DAYS = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
SECTIONS = ["Men's Wear","Women’s Wear","Kids","Accessories"]

c1, c2 = st.columns(2)
with c1:
    dwell = st.slider("Aisle dwell (minutes)", 0, 120, 18)
    cashier = st.slider("Cashier time (minutes)", 0, 30, 4)
    total_dur = st.slider("Total visit duration (minutes)", 0, 240, 45)
    items = st.number_input("Items purchased", 0, 20, 3)
    spend = st.number_input("Total spend", 0, 4000, 150, step=10)
with c2:
    disc = st.selectbox("Discount used", ["Yes","No"], index=0)
    repeat = st.selectbox("Repeat visitor", ["Yes","No"], index=1)
    age = st.selectbox("Age group", ["18-25","26-35","36-50","51+"], index=1)
    fam = st.selectbox("Family status", ["Bachelor","Couple","Family"], index=2)
    resident = st.selectbox("Resident type", ["Resident","Tourist"], index=0)
    gender = st.selectbox("Gender", ["Male","Female"], index=0)
    nation = st.selectbox("Nationality", NATIONS, index=1)
    timeband = st.selectbox("Time band", TIME_BANDS, index=2)
    day = st.selectbox("Day of week", DAYS, index=4)
    section = st.selectbox("Section visited", SECTIONS, index=1)

st.markdown("---")

# =================== PREDICT BUTTON ===================
if st.button("Detect Customer Type"):
    customer = pd.DataFrame([{
        "Dwell_Aisle_Min": dwell, "Cashier_Time_Min": cashier, "Visit_Duration_Total_Min": total_dur,
        "Items_Purchased": items, "Total_Spend": spend,
        "Discount_Used": disc, "Repeat_Visitor": repeat, "Age_Group": age, "Family_Status": fam,
        "Resident_Type": resident, "Nationality": nation, "Gender": gender,
        "Time_Band": timeband, "Day_of_Week": day, "Section": section
    }])

    # Predict
    pred = model.predict(customer)[0]
    proba = model.predict_proba(customer)[0]
    conf = float(np.max(proba))

    st.success(f"Predicted Customer Type: {pred}  |  Confidence: {conf*100:.1f}%")

    # ========== VISUAL 1: CUSTOMER BEHAVIOR ==========
    st.subheader("Customer Behavior")
    fig, ax = plt.subplots(figsize=(5,3))
    vals = [dwell, cashier, total_dur, items, spend]
    labels = ["Aisle dwell", "Cashier time", "Total visit", "Items", "Spend"]
    ax.bar(labels, vals, color=["#4c72b0","#55a868","#c44e52","#8172b2","#937860"])
    ax.set_ylabel("Value")
    ax.set_title("Single-visit behavior overview")
    st.pyplot(fig)

    # ========== VISUAL 2: BENCHMARKS TABLE ==========
    st.subheader("Benchmarks (for staffing & merchandising)")
    # Section Avg
    sec_df = df_demo[df_demo["Section"] == section]
    # Section@Time Avg
    sectime_df = df_demo[(df_demo["Section"] == section) & (df_demo["Time_Band"] == timeband)]

    def avg_or_nan(d, col):
        return float(d[col].mean()) if not d.empty else np.nan

    bench = pd.DataFrame({
        "Metric": ["Aisle dwell (min)", "Cashier (min)", "Total visit (min)", "Items", "Spend"],
        "This Customer": [dwell, cashier, total_dur, items, spend],
        f"{section} Avg": [
            avg_or_nan(sec_df, "Dwell_Aisle_Min"),
            avg_or_nan(sec_df, "Cashier_Time_Min"),
            avg_or_nan(sec_df, "Visit_Duration_Total_Min"),
            avg_or_nan(sec_df, "Items_Purchased"),
            avg_or_nan(sec_df, "Total_Spend"),
        ],
        f"{section} @ {timeband} Avg": [
            avg_or_nan(sectime_df, "Dwell_Aisle_Min"),
            avg_or_nan(sectime_df, "Cashier_Time_Min"),
            avg_or_nan(sectime_df, "Visit_Duration_Total_Min"),
            avg_or_nan(sectime_df, "Items_Purchased"),
            avg_or_nan(sectime_df, "Total_Spend"),
        ],
    })
    st.dataframe(bench.style.format({"This Customer":"{:.1f}",
                                     f"{section} Avg":"{:.1f}",
                                     f"{section} @ {timeband} Avg":"{:.1f}"}), use_container_width=True)

    # ========== VISUAL 3: CONTEXT BAR (FOOTFALL BY SECTION) ==========
    st.subheader("Footfall by Section (context from recent visits)")
    foot = df_demo["Section"].value_counts().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(5,3))
    foot.plot(kind="bar", ax=ax2, color="#4c72b0")
    ax2.set_xlabel("Section"); ax2.set_ylabel("Visitors"); ax2.set_title("Recent footfall by section")
    st.pyplot(fig2)

    # ========== MANAGER NARRATIVE (AUTO-FILLED) ==========
    st.subheader("Manager Narrative")
    # peak day/time and crowding from demo data (you could swap to live logs)
    peak_day = df_demo["Day_of_Week"].value_counts().idxmax()
    peak_time = df_demo["Time_Band"].value_counts().idxmax()
    crowd_section = df_demo["Section"].value_counts().idxmax()

    stmt1 = (
        f"There were {len(df_demo)} people from {df_demo['Nationality'].value_counts().idxmax()} "
        f"visiting mostly on {peak_day} and {peak_time}. Most crowded the {crowd_section} section. "
        f"Total billing observed in these logs is {df_demo['Total_Spend'].sum():,.0f}."
    )
    fam_mask = df_demo["Family_Status"].isin(["Family","Couple"])
    families_items = df_demo.loc[fam_mask, "Items_Purchased"].mean() if fam_mask.any() else 0
    bachelors_items = df_demo.loc[~fam_mask, "Items_Purchased"].mean() if (~fam_mask).any() else 0
    stmt2 = (
        f"Age groups present: {', '.join(df_demo['Age_Group'].value_counts().index.tolist())}. "
        f"Families bought ~{families_items:.1f} items vs bachelors ~{bachelors_items:.1f}. "
        f\"This customer is predicted '{pred}' with {conf*100:.0f}% confidence.\"
    )
    st.write(stmt1)
    st.write(stmt2)

    # ========== OPERATIONAL HINT (SIMPLE & ACTIONABLE) ==========
    st.subheader("Operational Hint")
    hint = ""
    if section == "Women’s Wear" and timeband in ["Evening","Night"]:
        hint = "Staff senior stylists in Women’s Wear for evening/night; consider curated bundles at cashiers."
    elif section == "Men's Wear" and timeband in ["Afternoon","Evening"]:
        hint = "Promote formalwear combos; ensure adequate fitting room throughput."
    elif section == "Accessories":
        hint = "Small upsells at checkout; feature travel-ready accessories for tourists."
    elif section == "Kids":
        hint = "Weekend play-zone draws; quick-bill counter reduces drop-offs."
    else:
        hint = "Align staff to observed footfall; keep cashier time under target during peaks."
    st.info(hint)

# =================== FOOTER ===================
st.markdown("---")
st.caption("This app trains a small Random Forest in-app (no external files needed). Visuals focus on decisions store managers care about.")
