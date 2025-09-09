import io
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Dubai Retail Intelligence", layout="wide")
st.title("Dubai Retail Intelligence — Clean Operational Insights")
st.caption("Demographics • Footfall • Dwell Time • Spend • Repeat Behavior → Practical, store-ready dashboards")

# ---------------- Data Loading ----------------
st.sidebar.header("Data")
upl = st.sidebar.file_uploader("Upload CSV (recommended: regenerated_retail_dataset.csv)", type=["csv"])

# Minimal Dubai-flavored sample (used only if no upload)
def sample_bytes():
    rows = [
        # Section, Day, Time, Nationality, Age, Family, Resident, Repeat, Repeat_Freq, Items, Spend, Dwell, Cashier, Visit_Duration, Preferred_Item_Bucket
        ["Women’s Wear","Friday","Night","India","26-35","Family","Resident","Yes",3,4,320,22,6,70,"Dresses"],
        ["Men's Wear","Saturday","Afternoon","UAE","36-50","Couple","Resident","No",0,3,210,18,4,55,"Formalwear"],
        ["Accessories","Thursday","Evening","Saudi Arabia","36-50","Family","Tourist","Yes",2,5,380,28,7,85,"Bags"],
        ["Kids","Sunday","Morning","Egypt","26-35","Couple","Resident","Yes",5,2,120,14,3,35,"Kidswear"],
        ["Women’s Wear","Thursday","Evening","Philippines","26-35","Family","Resident","Yes",4,3,150,16,4,40,"Dresses"],
        ["Accessories","Tuesday","Afternoon","India","18-25","Bachelor","Tourist","No",0,0,0,8,2,18,"Accessories"],
        ["Women’s Wear","Saturday","Evening","UAE","36-50","Family","Resident","Yes",1,4,260,20,5,60,"Dresses"],
        ["Men's Wear","Monday","Evening","Pakistan","36-50","Couple","Resident","Yes",2,3,225,19,4,52,"Formalwear"],
        ["Women’s Wear","Friday","Night","UK","51+","Family","Tourist","Yes",1,5,410,30,8,90,"Premium"],
        ["Kids","Wednesday","Morning","India","18-25","Bachelor","Tourist","No",0,1,45,10,2,22,"Snacks"],
    ]
    cols = ["Section","Day_of_Week","Time_Band","Nationality","Age_Group","Family_Status","Resident_Type",
            "Repeat_Visitor","Repeat_Frequency","Items_Purchased","Total_Spend","Dwell_Aisle_Min",
            "Cashier_Time_Min","Visit_Duration_Total_Min","Preferred_Item_Bucket"]
    df = pd.DataFrame(rows, columns=cols)
    bio = io.BytesIO(); df.to_csv(bio, index=False); return bio.getvalue()

if upl:
    df = pd.read_csv(upl)
else:
    st.info("No file uploaded—showing demo with a small Dubai sample. Upload your enriched dataset for full power.")
    df = pd.read_csv(io.BytesIO(sample_bytes()))

# --------- Soft column mapping (works even if names slightly differ) ----------
def pick(col_candidates):
    for c in col_candidates:
        if c in df.columns: return c
    return None

COLS = {
    "Section"       : pick(["Section","Aisle_Section","Shop_Section"]),
    "Day_of_Week"   : pick(["Day_of_Week","Visit_Day","Day"]),
    "Time_Band"     : pick(["Time_Band","TimeBand"]),
    "Nationality"   : pick(["Nationality","Country"]),
    "Age_Group"     : pick(["Age_Group","AgeBucket"]),
    "Gender"        : pick(["Gender"]),  # optional
    "Family_Status" : pick(["Family_Status"]),
    "Resident_Type" : pick(["Resident_Type"]),
    "Repeat_Visitor": pick(["Repeat_Visitor"]),
    "Repeat_Frequency": pick(["Repeat_Frequency","Repeat_Count"]),
    "Items_Purchased": pick(["Items_Purchased","Items"]),
    "Total_Spend"   : pick(["Total_Spend","Bill_Amount"]),
    "Dwell_Aisle_Min": pick(["Dwell_Aisle_Min","Dwell_Time_Min"]),
    "Cashier_Time_Min": pick(["Cashier_Time_Min","Checkout_Time_Min"]),
    "Visit_Duration_Total_Min": pick(["Visit_Duration_Total_Min","Total_Visit_Min"]),
    "Preferred_Item_Bucket": pick(["Preferred_Item_Bucket","Item_Bucket"])
}

# guard for required minimum set
required_min = ["Section","Day_of_Week","Time_Band","Nationality","Age_Group","Family_Status",
                "Resident_Type","Items_Purchased","Total_Spend","Dwell_Aisle_Min","Cashier_Time_Min"]
missing = [k for k in required_min if COLS[k] is None]
if missing:
    st.error(f"Your data is missing required columns (or names differ too much): {missing}")
    st.stop()

# ---------- Filters (interactive, simple) ----------
st.sidebar.header("Filters")
countries = ["All"] + sorted(df[COLS["Nationality"]].dropna().unique().tolist())
days      = ["All"] + ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
times     = ["All"] + ["Morning","Afternoon","Evening","Night"]
sections  = ["All"] + sorted(df[COLS["Section"]].dropna().unique().tolist())

f_country = st.sidebar.selectbox("Nationality", countries, index=0)
f_day     = st.sidebar.selectbox("Day of Week", days, index=0)
f_time    = st.sidebar.selectbox("Time Band", times, index=0)
f_section = st.sidebar.selectbox("Section", sections, index=0)

F = df.copy()
if f_country != "All": F = F[F[COLS["Nationality"]]==f_country]
if f_day != "All":     F = F[F[COLS["Day_of_Week"]]==f_day]
if f_time != "All":    F = F[F[COLS["Time_Band"]]==f_time]
if f_section != "All": F = F[F[COLS["Section"]]==f_section]

if F.empty:
    st.warning("No rows match current filters.")
    st.stop()

# ---------- KPI Cards ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Visitors (rows)", f"{len(F):,}")
k2.metric("Total Billing", f"{F[COLS['Total_Spend']].sum():,.0f}")
k3.metric("Avg Dwell (min)", f"{F[COLS['Dwell_Aisle_Min']].mean():.1f}")
k4.metric("Avg Cashier (min)", f"{F[COLS['Cashier_Time_Min']].mean():.1f}")

st.markdown("---")

# ---------- Row 1: Demographics & Footfall ----------
cA, cB, cC = st.columns(3)

with cA:
    st.subheader("Top Nationalities")
    top_nat = F[COLS["Nationality"]].value_counts().head(5)
    st.bar_chart(top_nat)

with cB:
    st.subheader("Footfall by Section")
    sec_counts = F[COLS["Section"]].value_counts().sort_values(ascending=False)
    st.bar_chart(sec_counts)

with cC:
    st.subheader("Peak Day & Time")
    day_counts = F[COLS["Day_of_Week"]].value_counts().reindex(
        ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]).dropna()
    time_counts = F[COLS["Time_Band"]].value_counts().reindex(["Morning","Afternoon","Evening","Night"]).dropna()
    st.write("By Day")
    st.bar_chart(day_counts)
    st.write("By Time")
    st.bar_chart(time_counts)

st.markdown("---")

# ---------- Row 2: Dwell Analytics & Spend Linked ----------
dA, dB, dC = st.columns(3)

with dA:
    st.subheader("Avg Dwell by Section")
    dwell_sec = F.groupby(COLS["Section"])[COLS["Dwell_Aisle_Min"]].mean().sort_values(ascending=False).round(1)
    st.bar_chart(dwell_sec)

with dB:
    st.subheader("Avg Cashier Time")
    cashier_avg = F[COLS["Cashier_Time_Min"]].mean()
    st.write(f"Average cashier time (min): **{cashier_avg:.1f}**")
    # show per section too
    cash_sec = F.groupby(COLS["Section"])[COLS["Cashier_Time_Min"]].mean().sort_values(ascending=False).round(1)
    st.bar_chart(cash_sec)

with dC:
    st.subheader("Spend linked to Section & Time")
    spend_sec = F.groupby(COLS["Section"])[COLS["Total_Spend"]].sum().sort_values(ascending=False)
    st.write("Total Spend by Section")
    st.bar_chart(spend_sec)
    st.write("Avg Spend by Time Band")
    spend_time = F.groupby(COLS["Time_Band"])[COLS["Total_Spend"]].mean().reindex(["Morning","Afternoon","Evening","Night"]).dropna().round(0)
    st.bar_chart(spend_time)

st.markdown("---")

# ---------- Row 3: Who buys what (Families vs Bachelors) ----------
wA, wB, wC = st.columns(3)

with wA:
    st.subheader("Age Groups")
    st.bar_chart(F[COLS["Age_Group"]].value_counts())

with wB:
    st.subheader("Families vs Bachelors — Items")
    fam_mask = F[COLS["Family_Status"]].isin(["Family","Couple"])
    items_by_group = pd.Series({
        "Families": F[fam_mask][COLS["Items_Purchased"]].mean() if fam_mask.any() else 0,
        "Bachelors": F[~fam_mask][COLS["Items_Purchased"]].mean() if (~fam_mask).any() else 0
    }).round(1)
    st.bar_chart(items_by_group)

with wC:
    st.subheader("Popular Buckets (if present)")
    if COLS["Preferred_Item_Bucket"] and COLS["Preferred_Item_Bucket"] in F.columns:
        top_buckets = F[COLS["Preferred_Item_Bucket"]].value_counts().head(5)
        st.bar_chart(top_buckets)
    else:
        st.write("No item bucket column found.")

st.markdown("---")

# ---------- Row 4: Repeat Behavior ----------
rA, rB, rC = st.columns(3)

with rA:
    st.subheader("Repeat Visitors")
    if COLS["Repeat_Visitor"] in F.columns:
        repeat_counts = F[COLS["Repeat_Visitor"]].value_counts()
        st.bar_chart(repeat_counts)
    else:
        st.write("No repeat column found.")

with rB:
    st.subheader("Repeat Frequency (avg)")
    if COLS["Repeat_Frequency"] and COLS["Repeat_Frequency"] in F.columns:
        st.write(f"Average repeat frequency: **{F[COLS['Repeat_Frequency']].astype(float).mean():.1f}** visits")
    else:
        st.write("Repeat frequency not available.")

with rC:
    st.subheader("Common Repeat Sections")
    if COLS["Repeat_Visitor"] in F.columns:
        repeats = F[F[COLS["Repeat_Visitor"]].astype(str).str.lower().isin(["yes","true","1"])]
        if not repeats.empty:
            st.bar_chart(repeats[COLS["Section"]].value_counts().head(5))
        else:
            st.write("No repeat visits in current filter.")
    else:
        st.write("No repeat column found.")

st.markdown("---")

# ---------- Manager Narrative (fills XXX / YYY / AAA / BBB / ZZZ / CCC / KKK) ----------
st.subheader("One-click Manager Narrative")
# Helper picks
top_nat_name = F[COLS["Nationality"]].value_counts().idxmax()
peak_day_name = F[COLS["Day_of_Week"]].value_counts().idxmax()
peak_time_name = F[COLS["Time_Band"]].value_counts().idxmax()
crowd_section  = F[COLS["Section"]].value_counts().idxmax()

people_count = len(F)
total_billing = F[COLS["Total_Spend"]].sum()

# "XXX billed KKK amount": interpret as number of bills (non-zero spend) and amount spent by those
billed_mask = F[COLS["Total_Spend"]] > 0
billed_num  = int(billed_mask.sum())
billed_amt  = float(F.loc[billed_mask, COLS["Total_Spend"]].sum())

# Family vs bachelors product tendencies
fam_mask = F[COLS["Family_Status"]].isin(["Family","Couple"])
families_items = float(F.loc[fam_mask, COLS["Items_Purchased"]].mean()) if fam_mask.any() else 0
bachelors_items = float(F.loc[~fam_mask, COLS["Items_Purchased"]].mean()) if (~fam_mask).any() else 0

popular_family_section = (F.loc[fam_mask, COLS["Section"]].value_counts().idxmax()
                          if fam_mask.any() else "N/A")
popular_bachelor_section = (F.loc[~fam_mask, COLS["Section"]].value_counts().idxmax()
                            if (~fam_mask).any() else "N/A")

# Repeat visitors
if COLS["Repeat_Visitor"] in F.columns:
    repeat_mask = F[COLS["Repeat_Visitor"]].astype(str).str.lower().isin(["yes","true","1"])
    repeat_num  = int(repeat_mask.sum())
else:
    repeat_num = 0

if COLS["Repeat_Frequency"] and COLS["Repeat_Frequency"] in F.columns:
    repeat_freq = F.loc[repeat_mask, COLS["Repeat_Frequency"]].astype(float).mean() if repeat_num>0 else 0.0
else:
    repeat_freq = 0.0

st.write(
    f"1. There were **{people_count}** people from **{top_nat_name}** visiting mostly on **{peak_day_name}** "
    f"and **{peak_time_name}**. Most of them crowded the **{crowd_section}** section and total billing was "
    f"**{total_billing:,.0f}**. Out of these, **{billed_num}** billed **{billed_amt:,.0f}**. "
    f"The remainder were family members or window shoppers."
)

st.write(
    f"2. The **{people_count}** people belonged to age groups like **"
    f"{', '.join(F[COLS['Age_Group']].value_counts().index.tolist()[:3])}**. "
    f"Families bought on average **{families_items:.1f}** items (often in **{popular_family_section}**), "
    f"whereas bachelors bought **{bachelors_items:.1f}** items (often in **{popular_bachelor_section}**). "
    f"Repeat visitors: **{repeat_num}**, visiting about **{repeat_freq:.1f}** times on average."
)

st.markdown("---")
st.caption("Tip: Use the sidebar to filter by nationality, day, time, or section. All KPIs, charts, and the narrative update instantly.")
