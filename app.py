"""
AI-Powered Orbital Traffic Management System — Streamlit Dashboard

Loads the artifacts produced by orbital_traffic_model.ipynb:
    - risk_model.pkl               (trained Random Forest)
    - orbital_data_processed.csv   (cleaned + feature-engineered catalog)
    - band_counts.pkl              (active-object counts per orbital band)

Run with:  streamlit run app.py
"""

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Orbital Traffic Management System", layout="wide")

# ----------------------------------------------------------------------
# Load artifacts (cached so they only load once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("risk_model.pkl")
    data = pd.read_csv("orbital_data_processed.csv")
    band_counts = joblib.load("band_counts.pkl")
    return model, data, band_counts


model, data, band_counts = load_artifacts()
BANDS = ["LEO", "MEO", "GEO", "HEO"]
band_counts = band_counts.reindex(BANDS).fillna(0)

# Average risk score per band, computed once from the processed dataset —
# used by the recommendation engine and the sustainability score.
active_data = data[data["IS_ACTIVE"]]
band_avg_risk = active_data.groupby("ORBITAL_BAND")["RISK_SCORE"].mean().reindex(BANDS)


def classify_band(altitude):
    if altitude < 2000:
        return "LEO"
    elif altitude < 35586:
        return "MEO"
    elif altitude < 35986:
        return "GEO"
    return "HEO"


# ----------------------------------------------------------------------
# Sidebar — mission input
# ----------------------------------------------------------------------
st.sidebar.header("🛰️ Proposed Orbit Parameters")
perigee = st.sidebar.slider("Perigee (km)", 150, 40000, 550, step=10)
apogee = st.sidebar.slider("Apogee (km)", 150, 42000, 560, step=10)
inclination = st.sidebar.slider("Inclination (degrees)", 0.0, 180.0, 53.0, step=0.5)

mean_altitude = (perigee + apogee) / 2
selected_band = classify_band(mean_altitude)

input_df = pd.DataFrame([{"PERIGEE": perigee, "APOGEE": apogee, "INCLINATION": inclination}])
predicted_risk = model.predict(input_df)[0]
proba = dict(zip(model.classes_, model.predict_proba(input_df)[0]))

st.title("AI-Powered Orbital Traffic Management System")
st.caption("Decision-support dashboard for satellite mission planning & orbital safety")

# ----------------------------------------------------------------------
# Row 1 — Risk assessment + Orbit summary
# ----------------------------------------------------------------------
col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("Risk Assessment")
    risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
    st.metric("Predicted Risk Level", f"{risk_colors.get(predicted_risk, '')} {predicted_risk}")
    st.write(f"**Orbital region:** {selected_band}  |  **Mean altitude:** {mean_altitude:,.0f} km")
    proba_df = pd.DataFrame({"Risk Level": list(proba.keys()), "Confidence": list(proba.values())})
    st.bar_chart(proba_df.set_index("Risk Level"))

with col2:
    st.subheader("Orbital Congestion Analysis")
    congestion_df = pd.DataFrame({"Region": BANDS, "Active Objects": band_counts.values})
    st.bar_chart(congestion_df.set_index("Region"))
    region_share = band_counts[selected_band] / band_counts.sum() * 100
    st.write(
        f"**{selected_band}** currently holds **{int(band_counts[selected_band]):,}** active "
        f"objects (**{region_share:.1f}%** of all tracked active traffic)."
    )

st.divider()

# ----------------------------------------------------------------------
# Row 2 — Recommendation Engine + Launch Window + Sustainability
# ----------------------------------------------------------------------
col3, col4, col5 = st.columns(3)

with col3:
    st.subheader("Orbit Recommendation Engine")
    st.write("Average risk score by region (lower is better):")
    rec_df = band_avg_risk.round(3).rename("Avg Risk Score").reset_index()
    rec_df.columns = ["Region", "Avg Risk Score"]
    st.dataframe(rec_df, hide_index=True, use_container_width=True)
    best_band = band_avg_risk.idxmin()
    if selected_band == best_band:
        st.success(f"Your selected region ({selected_band}) is already the lowest-risk option.")
    else:
        st.info(f"Consider **{best_band}** — it has a lower average risk than {selected_band}.")

with col4:
    st.subheader("Launch Window Planning")
    congestion_score_selected = band_counts[selected_band] / band_counts.max()
    if congestion_score_selected < 0.33:
        window_msg = "🟢 Good window — this region is relatively uncongested right now."
    elif congestion_score_selected < 0.7:
        window_msg = "🟡 Moderate window — some coordination with existing traffic advised."
    else:
        window_msg = "🔴 Busy window — this region is heavily trafficked; extra scheduling care advised."
    st.write(window_msg)
    st.caption("Simplified assessment based on current regional congestion, not live conjunction data.")

with col5:
    st.subheader("Orbital Sustainability Score")
    sustainability = 100 - (congestion_score_selected * 50 + band_avg_risk[selected_band] * 50)
    sustainability = max(0, min(100, sustainability))
    st.metric("Sustainability Score", f"{sustainability:.0f} / 100")
    st.progress(int(sustainability))
    st.caption("Combines regional congestion and average collision risk.")

st.divider()

# ----------------------------------------------------------------------
# Row 3 — Debris Priority
# ----------------------------------------------------------------------
st.subheader("Debris Priority")
debris = active_data[active_data["OBJECT_TYPE"] == "DEB"].copy()
debris_band = debris[debris["ORBITAL_BAND"] == selected_band]
debris_band = debris_band.sort_values("RISK_SCORE", ascending=False).head(15)

if len(debris_band) > 0:
    display_cols = ["OBJECT_NAME", "PERIGEE", "APOGEE", "INCLINATION", "RCS", "RISK_LEVEL"]
    st.write(f"Top tracked debris objects in **{selected_band}**, ranked by priority:")
    st.dataframe(debris_band[display_cols], hide_index=True, use_container_width=True)
else:
    st.write(f"No tracked debris currently catalogued in {selected_band}.")

st.divider()

# ----------------------------------------------------------------------
# Orbit Copilot — rule-based Q&A over the computed results
# ----------------------------------------------------------------------
st.subheader("🤖 Orbit Copilot")
question = st.selectbox(
    "Ask about the selected orbit:",
    [
        "Is this orbit safe?",
        "Why is this orbit at this risk level?",
        "Why is this region congested?",
        "What should I do next?",
    ],
)

if question == "Is this orbit safe?":
    answer = (
        f"The predicted risk level for this orbit is **{predicted_risk}**, with "
        f"{proba[predicted_risk]*100:.0f}% model confidence. "
        + ("This is a comparatively safe choice." if predicted_risk in ("Low", "Medium")
           else "Extra caution and closer conjunction screening are recommended.")
    )
elif question == "Why is this orbit at this risk level?":
    answer = (
        f"This orbit sits in **{selected_band}**, which holds "
        f"{int(band_counts[selected_band]):,} active tracked objects "
        f"(congestion score {congestion_score_selected:.2f} out of 1.0). "
        "The model learned this pattern from historical Perigee, Apogee, and Inclination data, "
        "without being told the region directly."
    )
elif question == "Why is this region congested?":
    answer = (
        f"**{selected_band}** accounts for {region_share:.1f}% of all currently active tracked "
        "objects. Regions closer to Earth (like LEO) are cheaper to reach and host most "
        "communication and Earth-observation constellations, which drives up traffic there."
    )
else:
    answer = (
        f"Given a **{predicted_risk}** predicted risk and a sustainability score of "
        f"{sustainability:.0f}/100, consider comparing **{best_band}** as an alternative region, "
        "reviewing the debris priority list above for nearby high-priority objects, and "
        "re-checking the launch window closer to the planned date."
    )

st.write(answer)

st.divider()
st.caption(
    "Decision-support tool for preliminary mission planning — not a substitute for official "
    "satellite tracking or collision-avoidance systems."
)
