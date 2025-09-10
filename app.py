import io
import datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# ====================== PAGE ======================
st.set_page_config(page_title="Dubai Retail – Customer Type & Insights", layout="centered")
st.title("Dubai Retail — Customer Type & Operational Insights")
st.caption("Enter a shopper’s details → get predicted type, what-if, and visuals managers can act on.")

st.markdown("---")

# ====================== OPTIONAL STORE LOGS (for benchmarks & context) ======================
st.sidebar.header("Store Logs (optional)")
upl = st.sidebar.file_uploader("Upload CSV of recent visits (optional)", type=["csv"])

def demo_logs_bytes():
    rows = [
        # dwell, cashier, total, items, spend, disc, repeat, age, fam, resident, nationality, gender, time, day, section, type
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
    cols = ["Dwell_Aisle_Min","Cashier_Time_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend",
            "Discount_Used","Repeat_Visitor","Age_Group","Family_Status","Resident_Type","Nationality",
            "Gender","Time_Band","Day_of_Week","Section","Purchase_Type"]
    df = pd.DataFrame(rows, columns=cols)
    buf = io.BytesIO(); df.to_csv(buf, index=False); return buf.getvalue()

if upl:
    logs = pd.read_csv(upl)
else:
    st.info("No logs uploaded — using a small Dubai demo log for benchmarks and context.")
    logs = pd.read_csv(io.BytesIO(demo_logs_bytes()))

# -------- soft column mapping so your CSV can have slightly different names --------
def pick(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

MAP = {
    "Dwell": pick(logs, ["Dwell_Aisle_Min","Dwell_Time_Min"]),
    "Cashier": pick(logs, ["Cashier_Time_Min","Checkout_Time_Min"]),
    "Visit": pick(logs, ["Visit_Duration_Total_Min","Total_Visit_Min"]),
    "Items": pick(logs, ["Items_Purchased","Items"]),
    "Spend": pick(logs, ["Total_Spend","Bill_Amount"]),
    "Discount": pick(logs, ["Discount_Used"]),
    "Repeat": pick(logs, ["Repeat_Visitor"]),
    "Age": pick(logs, ["Age_Group","AgeBucket"]),
    "Family": pick(logs, ["Family_Status"]),
    "Resident": pick(logs, ["Resident_Type"]),
    "Nationality": pick(logs, ["Nationality","Country"]),
    "Gender": pick(logs, ["Gender"]),
    "Time": pick(logs, ["Time_Band","TimeBand"]),
    "Day": pick(logs, ["Day_of_Week","Day"]),
    "Section": pick(logs, ["Section","Aisle_Section"]),
    "Type": pick(logs, ["Purchase_Type","Segment"])
}

REQUIRED = ["Dwell","Cashier","Visit","Items","Spend","Discount","Repeat","Age","Family","Resident","Nationality","Gender","Time","Day","Section","Type"]
if any(MAP[k] is None for k in REQUIRED):
    st.error("Your logs are missing required columns (or names differ too much). Expected: "
             "dwell, cashier, visit, items, spend, discount, repeat, age, family, resident, nationality, gender, time, day, section, type.")
    st.stop()

# ====================== TRAIN MODEL IN-APP ======================
NUM = [MAP["Dwell"], MAP["Cashier"], MAP["Visit"], MAP["Items"], MAP["Spend"]]
CAT = [MAP["Discount"], MAP["Repeat"], MAP["Age"], MAP["Family"], MAP["Resident"],
       MAP["Nationality"], MAP["Gender"], MAP["Time"], MAP["Day"], MAP["Section"]]
TARGET = MAP["Type"]

X = logs[NUM + CAT]
y = logs[TARGET]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)
])
model = Pipeline([("prep", preprocess), ("rf", RandomForestClassifier(n_estimators=200, random_state=42))])
model.fit(X, y)

# ====================== INPUTS ======================
st.subheader("Enter Customer Details")

NATIONS = sorted(logs[MAP["Nationality"]].dropna().unique().tolist()) or ["UAE","India","Pakistan","Philippines","Saudi Arabia","Egypt","UK"]
TIME_BANDS = ["Morning","Afternoon","Evening","Night"]
DAYS = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
SECTIONS = sorted(logs[MAP["Section"]].dropna().unique().tolist()) or ["Men's Wear","Women’s Wear","Kids","Accessories"]

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
    nation = st.selectbox("Nationality", NATIONS, index=min(1, len(NATIONS)-1))
    timeband = st.selectbox("Time band", TIME_BANDS, index=2)
    day = st.selectbox("Day of week", DAYS, index=4)
    section = st.selectbox("Section visited", SECTIONS, index=min(1, len(SECTIONS)-1))

st.markdown("---")

# ====================== PREDICT (ENHANCED + TABS) ======================
if st.button("Detect Customer Type"):
    cust = pd.DataFrame([{
        MAP["Dwell"]: dwell, MAP["Cashier"]: cashier, MAP["Visit"]: total_dur,
        MAP["Items"]: items, MAP["Spend"]: spend, MAP["Discount"]: disc, MAP["Repeat"]: repeat,
        MAP["Age"]: age, MAP["Family"]: fam, MAP["Resident"]: resident, MAP["Nationality"]: nation,
        MAP["Gender"]: gender, MAP["Time"]: timeband, MAP["Day"]: day, MAP["Section"]: section
    }])

    pred = model.predict(cust)[0]
    proba = model.predict_proba(cust)[0]; conf = float(np.max(proba))

    st.success(f"Predicted Customer Type: {pred}  |  Confidence: {conf*100:.1f}%")

    # --- Benchmarks for deltas ---
    sec_df = logs[logs[MAP["Section"]] == section]
    st_df  = logs[(logs[MAP["Section"]] == section) & (logs[MAP["Time"]] == timeband)]

    def mean_or_nan(d, col):
        try:
            return float(d[col].astype(float).mean())
        except Exception:
            return float("nan")

    sec_avg = {
        "dwell": mean_or_nan(sec_df, MAP["Dwell"]),
        "cash":  mean_or_nan(sec_df, MAP["Cashier"]),
        "visit": mean_or_nan(sec_df, MAP["Visit"]),
        "items": mean_or_nan(sec_df, MAP["Items"]),
        "spend": mean_or_nan(sec_df, MAP["Spend"]),
    }

    # ---------- Tabs: Results | What-If | Store Heatmap | Drivers | Save Scenario ----------
    tab_res, tab_whatif, tab_heat, tab_drv, tab_save = st.tabs(
        ["Results", "What-If", "Store Heatmap", "Drivers", "Save Scenario"]
    )

    # ========== Results ==========
    with tab_res:
        st.subheader("Customer Behavior")
        fig, ax = plt.subplots(figsize=(5, 3))
        vals = [dwell, cashier, total_dur, items, spend]
        labels = ["Aisle dwell", "Cashier time", "Total visit", "Items", "Spend"]
        ax.bar(labels, vals, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#937860"])
        ax.set_ylabel("Value"); ax.set_title("Single-visit behavior overview")
        st.pyplot(fig)

        # --- Delta KPIs vs Section Avg (simple & business-friendly) ---
        d1, d2, d3, d4, d5 = st.columns(5)
        def mk(delta):
            if pd.isna(delta): return ("–","white")
            return ("▲","green") if delta >= 0 else ("▼","red")

        for (label, curr, avg, holder) in [
            ("Dwell", dwell, sec_avg["dwell"], d1),
            ("Cashier", cashier, sec_avg["cash"], d2),
            ("Visit", total_dur, sec_avg["visit"], d3),
            ("Items", items, sec_avg["items"], d4),
            ("Spend", spend, sec_avg["spend"], d5),
        ]:
            delta = curr - avg if not pd.isna(avg) else np.nan
            arrow, _ = mk(delta)
            with holder:
                st.metric(label, f"{curr:.1f}", f"{arrow} {delta:.1f}" if not pd.isna(delta) else "n/a")

        st.subheader("Benchmarks")
        bench = pd.DataFrame({
            "Metric": ["Aisle dwell (min)", "Cashier (min)", "Total visit (min)", "Items", "Spend"],
            "This Customer": [dwell, cashier, total_dur, items, spend],
            f"{section} Avg": [sec_avg["dwell"], sec_avg["cash"], sec_avg["visit"], sec_avg["items"], sec_avg["spend"]],
            f"{section} @ {timeband} Avg": [
                mean_or_nan(st_df, MAP["Dwell"]), mean_or_nan(st_df, MAP["Cashier"]),
                mean_or_nan(st_df, MAP["Visit"]), mean_or_nan(st_df, MAP["Items"]),
                mean_or_nan(st_df, MAP["Spend"]),
            ],
        })
        st.dataframe(
            bench.style.format({"This Customer":"{:.1f}", f"{section} Avg":"{:.1f}", f"{section} @ {timeband} Avg":"{:.1f}"}),
            use_container_width=True
        )

        # Manager Narrative (concise, linked)
        top_nat = logs[MAP["Nationality"]].value_counts().idxmax()
        peak_day  = logs[MAP["Day"]].value_counts().idxmax()
        peak_time = logs[MAP["Time"]].value_counts().idxmax()
        crowd_sec = logs[MAP["Section"]].value_counts().idxmax()
        total_billing = logs[MAP["Spend"]].sum()
        billed_num = int((logs[MAP["Spend"]] > 0).sum())
        billed_amt = float(logs.loc[logs[MAP["Spend"]] > 0, MAP["Spend"]].sum())

        st.markdown(
            f"There were **{len(logs)}** people from **{top_nat}** visiting mostly on **{peak_day}** and **{peak_time}**. "
            f"Most crowded **{crowd_sec}**; total billing **{total_billing:,.0f}**. "
            f"Out of these, **{billed_num}** billed **{billed_amt:,.0f}**. "
            f"This customer is **{pred}** with **{conf*100:.0f}%** confidence."
        )

    # ========== What-If ==========
    with tab_whatif:
        st.subheader("What-If Simulator")
        colL, colR = st.columns(2)
        with colL:
            promo = st.slider("Promo (%)", 0, 30, 10, step=5)  # simple uplift on spend
            staffing = st.selectbox("Extra staffing at cashier?", ["No", "Yes"])
        with colR:
            # simulate: promo increases spend; staffing reduces cashier time
            sim_spend = spend * (1 + promo / 100)
            sim_cashier = max(0, cashier - (2 if staffing == "Yes" else 0))
            sim_row = cust.copy()
            sim_row[MAP["Spend"]] = sim_spend
            sim_row[MAP["Cashier"]] = sim_cashier

            sim_pred = model.predict(sim_row)[0]
            sim_proba = model.predict_proba(sim_row)[0]; sim_conf = float(np.max(sim_proba))

            st.info(
                f"Simulated Result → Type: **{sim_pred}** | Confidence: **{sim_conf*100:.1f}%** "
                f"(Spend={sim_spend:.0f}, Cashier Time={sim_cashier:.1f}m)"
            )

        # mini bar to visualize change
        fig3, ax3 = plt.subplots(figsize=(5, 2.6))
        ax3.bar(["Spend (now)","Spend (what-if)"], [spend, sim_spend], color=["#4c72b0","#55a868"])
        ax3.set_title("Promo Impact on Spend"); ax3.set_ylabel("Currency")
        st.pyplot(fig3)

    # ========== Store Heatmap ==========
    with tab_heat:
        st.subheader("Section × Time Heatmap (footfall by count)")
        grid = logs.pivot_table(index=MAP["Section"], columns=MAP["Time"], values=MAP["Items"], aggfunc="count").fillna(0)
        st.dataframe(grid.astype(int), use_container_width=True)

    # ========== Drivers (Global) ==========
    with tab_drv:
        st.subheader("What drives the model? (global feature groups)")
        rf = model.named_steps["rf"]
        feat_names = model.named_steps["prep"].get_feature_names_out()
        imps = pd.Series(rf.feature_importances_, index=feat_names)

        # group one-hot back to base features
        group_map = {
            MAP["Dwell"]: "Dwell_Aisle_Min",
            MAP["Cashier"]: "Cashier_Time_Min",
            MAP["Visit"]: "Visit_Duration_Total_Min",
            MAP["Items"]: "Items_Purchased",
            MAP["Spend"]: "Total_Spend",
            MAP["Discount"]: "Discount_Used",
            MAP["Repeat"]: "Repeat_Visitor",
            MAP["Age"]: "Age_Group",
            MAP["Family"]: "Family_Status",
            MAP["Resident"]: "Resident_Type",
            MAP["Nationality"]: "Nationality",
            MAP["Gender"]: "Gender",
            MAP["Time"]: "Time_Band",
            MAP["Day"]: "Day_of_Week",
            MAP["Section"]: "Section",
        }

        agg = {}
        for full_name, val in imps.items():
            matched = None
            for raw, nice in group_map.items():
                if full_name.endswith(raw) or f"__{raw}_" in full_name or full_name.split(":")[0].endswith(raw):
                    matched = nice; break
            if matched is None:
                matched = full_name
            agg[matched] = agg.get(matched, 0) + val

        drv = pd.Series(agg).sort_values(ascending=False).head(8)
        st.bar_chart(drv)

    # ========== Save Scenario ==========
    with tab_save:
        st.subheader("Save Scenario")
        snap = cust.copy()
        snap["Predicted_Type"] = pred
        snap["Confidence"] = round(conf, 4)
        snap["Saved_At"] = dt.datetime.now().isoformat(timespec="seconds")
        st.dataframe(snap, use_container_width=True)
        st.download_button(
            "Download this scenario (CSV)",
            data=snap.to_csv(index=False).encode("utf-8"),
            file_name="customer_scenario.csv",
            mime="text/csv",
        )

# ====================== FOOTER ======================
st.markdown("---")
st.caption("Trains a small Random Forest in-app (no pickles). Tabs: Results • What-If • Heatmap • Drivers • Save.")
