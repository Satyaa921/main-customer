import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =============== PAGE ===============
st.set_page_config(page_title="Dubai Retail — Store Heatmaps", layout="wide")
st.title("Dubai Retail — Store Heatmaps")
st.caption("Footfall • Spend • Dwell • Cashier • Conversion • Items • Days • Nationalities")

st.markdown("---")

# =============== DATA LOAD (CSV optional) ===============
st.sidebar.header("Data")
upl = st.sidebar.file_uploader("Upload CSV of recent visits (optional)", type=["csv"])

def demo_logs_bytes():
    rows = [
        # dwell, cashier, visit, items, spend, disc, repeat, age, fam, resident, nationality, gender, time, day, section
        [ 8,2,18,0,  0,"No","No","18-25","Bachelor","Tourist","India","Male","Afternoon","Tuesday","Accessories"],
        [10,3,24,1, 35,"Yes","No","18-25","Bachelor","Resident","UAE","Male","Evening","Monday","Men's Wear"],
        [16,4,38,2,110,"Yes","Yes","26-35","Family","Resident","Philippines","Female","Evening","Thursday","Women’s Wear"],
        [18,5,42,3,140,"No","Yes","26-35","Family","Resident","India","Female","Night","Friday","Women’s Wear"],
        [20,4,55,3,220,"Yes","No","36-50","Couple","Resident","Pakistan","Male","Afternoon","Saturday","Men's Wear"],
        [22,6,60,4,260,"No","Yes","36-50","Couple","Resident","UAE","Male","Evening","Saturday","Women’s Wear"],
        [28,7,85,4,350,"Yes","Yes","36-50","Family","Tourist","Saudi Arabia","Female","Night","Thursday","Women’s Wear"],
        [30,8,90,5,420,"Yes","Yes","51+","Family","Tourist","UK","Female","Night","Friday","Accessories"],
        [14,3,32,2,120,"No","Yes","26-35","Couple","Resident","Egypt","Male","Morning","Sunday","Kids"],
        [25,4,58,3,230,"Yes","Yes","36-50","Family","Resident","India","Female","Evening","Sunday","Women’s Wear"],
        [ 6,1,15,0,  0,"No","No","18-25","Bachelor","Tourist","Pakistan","Male","Morning","Wednesday","Kids"],
        [12,2,28,1, 60,"Yes","No","26-35","Couple","Resident","Philippines","Female","Afternoon","Tuesday","Accessories"],
        [22,5,62,3,240,"Yes","Yes","36-50","Couple","Resident","UAE","Female","Evening","Friday","Men's Wear"],
        [26,6,76,4,320,"No","Yes","36-50","Family","Resident","India","Male","Evening","Thursday","Women’s Wear"],
    ]
    cols = ["Dwell_Aisle_Min","Cashier_Time_Min","Visit_Duration_Total_Min","Items_Purchased","Total_Spend",
            "Discount_Used","Repeat_Visitor","Age_Group","Family_Status","Resident_Type","Nationality",
            "Gender","Time_Band","Day_of_Week","Section"]
    df = pd.DataFrame(rows, columns=cols)
    buf = io.BytesIO(); df.to_csv(buf, index=False); return buf.getvalue()

if upl:
    logs = pd.read_csv(upl)
else:
    st.info("No file uploaded — using a small Dubai demo log. Upload your CSV to power these heatmaps with real data.")
    logs = pd.read_csv(io.BytesIO(demo_logs_bytes()))

# =============== FLEXIBLE COLUMN MAP ===============
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
    "Nationality": pick(logs, ["Nationality","Country"]),
    "Time": pick(logs, ["Time_Band","TimeBand"]),
    "Day": pick(logs, ["Day_of_Week","Day"]),
    "Section": pick(logs, ["Section","Aisle_Section"]),
}

REQUIRED = ["Dwell","Cashier","Visit","Items","Spend","Nationality","Time","Day","Section"]
missing = [k for k in REQUIRED if MAP[k] is None]
if missing:
    st.error(f"Your data is missing required columns (or names differ too much): {missing}")
    st.stop()

# =============== FILTERS ===============
st.sidebar.header("Filters")
days_order = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
time_order = ["Morning","Afternoon","Evening","Night"]

countries = ["All"] + sorted(logs[MAP["Nationality"]].dropna().unique().tolist())
sections  = ["All"] + sorted(logs[MAP["Section"]].dropna().unique().tolist())
days      = ["All"] + days_order
times     = ["All"] + time_order

f_country = st.sidebar.selectbox("Nationality", countries, index=0)
f_section = st.sidebar.selectbox("Section", sections, index=0)
f_day     = st.sidebar.selectbox("Day of Week", days, index=0)
f_time    = st.sidebar.selectbox("Time Band", times, index=0)

F = logs.copy()
if f_country != "All": F = F[F[MAP["Nationality"]]==f_country]
if f_section != "All": F = F[F[MAP["Section"]]==f_section]
if f_day != "All":     F = F[F[MAP["Day"]]==f_day]
if f_time != "All":    F = F[F[MAP["Time"]]==f_time]

if F.empty:
    st.warning("No rows match current filters.")
    st.stop()

# =============== KPIs ===============
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows", f"{len(F):,}")
k2.metric("Total Billing", f"{F[MAP['Spend']].sum():,.0f}")
k3.metric("Avg Dwell (min)", f"{F[MAP['Dwell']].mean():.1f}")
k4.metric("Avg Cashier (min)", f"{F[MAP['Cashier']].mean():.1f}")
k5.metric("Avg Items", f"{F[MAP['Items']].mean():.1f}")

st.markdown("---")

# =============== Heatmap Helper ===============
def draw_heatmap(df_matrix, title, cmap="Blues", fmt="{:.0f}"):
    if df_matrix.empty:
        st.info(f"No data for: {title}")
        return
    # ensure ordering of columns/rows looks nice
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(df_matrix.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(df_matrix.shape[1])); ax.set_xticklabels(df_matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(df_matrix.shape[0])); ax.set_yticklabels(df_matrix.index)
    ax.set_title(title)
    # annotate cells (light touch for readability)
    for i in range(df_matrix.shape[0]):
        for j in range(df_matrix.shape[1]):
            val = df_matrix.values[i, j]
            ax.text(j, i, (fmt.format(val) if pd.notna(val) else ""), ha="center", va="center", fontsize=8, color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    st.pyplot(fig)

# =============== HEATMAPS ===============
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Footfall (Section × Time)",
    "Average Spend (Section × Time)",
    "Average Dwell (Section × Time)",
    "Average Cashier (Section × Time)",
    "Conversion Rate (Section × Time)",
    "Items per Visit (Section × Time)",
    "Footfall (Section × Day)",
    "Nationalities (Section × Country)"
])

# 1) Footfall count (Section × Time)
with tab1:
    grid = F.pivot_table(index=MAP["Section"], columns=MAP["Time"], values=MAP["Items"], aggfunc="count").fillna(0)
    # order axis
    grid = grid.reindex(columns=[t for t in time_order if t in grid.columns])
    draw_heatmap(grid.astype(int), "Footfall (visit count) — Section × Time", cmap="Blues", fmt="{:.0f}")

# 2) Average Spend (Section × Time)
with tab2:
    grid = F.pivot_table(index=MAP["Section"], columns=MAP["Time"], values=MAP["Spend"], aggfunc="mean")
    grid = grid.reindex(columns=[t for t in time_order if t in grid.columns])
    draw_heatmap(grid.fillna(0), "Average Spend — Section × Time", cmap="Greens", fmt="{:.0f}")

# 3) Average Dwell (Section × Time)
with tab3:
    grid = F.pivot_table(index=MAP["Section"], columns=MAP["Time"], values=MAP["Dwell"], aggfunc="mean")
    grid = grid.reindex(columns=[t for t in time_order if t in grid.columns])
    draw_heatmap(grid.fillna(0), "Average Aisle Dwell (minutes) — Section × Time", cmap="Oranges", fmt="{:.1f}")

# 4) Average Cashier (Section × Time)
with tab4:
    grid = F.pivot_table(index=MAP["Section"], columns=MAP["Time"], values=MAP["Cashier"], aggfunc="mean")
    grid = grid.reindex(columns=[t for t in time_order if t in grid.columns])
    draw_heatmap(grid.fillna(0), "Average Cashier Time (minutes) — Section × Time", cmap="Purples", fmt="{:.1f}")

# 5) Conversion Rate (Section × Time): share of visits with spend > 0
with tab5:
    def conv_rate(df):
        grp = df.groupby([MAP["Section"], MAP["Time"]])
        num = grp.apply(lambda d: (d[MAP["Spend"]] > 0).mean() if len(d) else np.nan)
        return num.unstack().reindex(columns=[t for t in time_order if t in num.unstack().columns])
    grid = conv_rate(F) * 100
    draw_heatmap(grid.fillna(0), "Conversion Rate % — Section × Time", cmap="Reds", fmt="{:.0f}")

# 6) Items per Visit (Section × Time)
with tab6:
    grid = F.pivot_table(index=MAP["Section"], columns=MAP["Time"], values=MAP["Items"], aggfunc="mean")
    grid = grid.reindex(columns=[t for t in time_order if t in grid.columns])
    draw_heatmap(grid.fillna(0), "Average Items per Visit — Section × Time", cmap="BuPu", fmt="{:.1f}")

# 7) Footfall (Section × Day)
with tab7:
    grid = F.pivot_table(index=MAP["Section"], columns=MAP["Day"], values=MAP["Items"], aggfunc="count").fillna(0)
    grid = grid.reindex(columns=[d for d in days_order if d in grid.columns])
    draw_heatmap(grid.astype(int), "Footfall (visit count) — Section × Day", cmap="Blues", fmt="{:.0f}")

# 8) Nationalities (Section × Country) — top N countries
with tab8:
    topN = st.slider("Top N countries", 3, 12, 6, step=1)
    top_countries = F[MAP["Nationality"]].value_counts().head(topN).index.tolist()
    F2 = F[F[MAP["Nationality"]].isin(top_countries)]
    grid = F2.pivot_table(index=MAP["Section"], columns=MAP["Nationality"], values=MAP["Items"], aggfunc="count").fillna(0)
    draw_heatmap(grid.astype(int), f"Footfall (count) — Section × Top {topN} Nationalities", cmap="Greens", fmt="{:.0f}")

st.markdown("---")
st.caption("Tip: Use the sidebar filters (Country/Section/Day/Time). All heatmaps update instantly. Align staffing and promotions to high footfall, high spend, and long-dwell cells.")
