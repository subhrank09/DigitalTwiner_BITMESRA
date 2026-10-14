"""
GlucoTwin - clinician dashboard (interactive prototype).

Run:   streamlit run app/dashboard.py
Needs: python scripts/run_pipeline.py   (creates data/ and artifacts/)
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glucotwin.plots import RISK_COLOR, plot_cgm_forecast, plot_drivers, plot_vitals, plot_whatif
from glucotwin.twin import VirtualPatient

st.set_page_config(page_title="GlucoTwin", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner="Loading synthetic cohort and model...")
def load():
    ehr = pd.read_csv(ROOT / "data" / "synthetic" / "ehr_with_latent_truth.csv")
    series = pd.read_csv(ROOT / "data" / "synthetic" / "wearables_5min.csv.gz", parse_dates=["ts"])
    model = joblib.load(ROOT / "artifacts" / "glucoforecaster.joblib")
    return ehr, series, model


try:
    ehr, series, model = load()
except FileNotFoundError:
    st.error("Run `python scripts/run_pipeline.py` first to create the synthetic data and trained model.")
    st.stop()

st.title("GlucoTwin  |  Virtual Patient Console")
st.caption("SYNTHETIC patients and data. Decision-support prototype - not a medical device, not for clinical use.")

with st.sidebar:
    st.header("Patient")
    pid = st.selectbox("Select virtual patient", ehr["patient_id"].tolist(), index=0)
    idx = int(ehr.index[ehr["patient_id"] == pid][0])
    row = ehr.iloc[idx]
    ts = series[series["patient_id"] == pid].reset_index(drop=True)
    st.markdown(f"**{row['age']} y · {row['sex']} · BMI {row['bmi']}**")
    st.metric("HbA1c", f"{row['hba1c']}%")
    st.metric("Fasting glucose", f"{int(row['fasting_glucose'])} mg/dL")
    st.write(f"Diagnoses (ICD-10): `{row['diagnoses_icd10']}`")
    st.write(f"Metformin: {'yes' if row['on_metformin'] else 'no'} · Sulfonylurea: {'yes' if row['on_sulfonylurea'] else 'no'}")
    st.write(f"TCF7L2 risk alleles: {row['tcf7l2_risk_alleles']} · Family history: {'yes' if row['family_history'] else 'no'}")
    st.write(f"eGFR {int(row['egfr'])} · LDL {int(row['ldl'])} · SBP {int(row['sbp'])}")

twin = VirtualPatient(row, ts, model, sim_seed=42 * 1000 + idx)

n_days = len(ts) // 288
c1, c2 = st.columns([3, 1])
day = c1.slider("Day", 3, n_days - 1, min(10, n_days - 1))
hour = c2.slider("Time of day (h)", 0.0, 23.75, 12.0, 0.25)
now = int(day * 288 + hour * 12)
fc = twin.forecast(now)

left, right = st.columns([3, 1.2])
with left:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    plot_cgm_forecast(ax, twin, now)
    ax.set_title(f"Live CGM + 2-hour forecast (now {fc.timestamp:%a %H:%M})", loc="left", fontsize=11)
    fig.tight_layout()
    st.pyplot(fig)
with right:
    color = RISK_COLOR[fc.risk_level]
    st.markdown(f"<div style='background:{color};color:white;padding:10px 16px;border-radius:10px;font-weight:700;"
                f"font-size:22px;text-align:center'>{fc.risk_level} RISK</div>", unsafe_allow_html=True)
    st.metric("Chance of >180 mg/dL within 2 h", f"{fc.probability * 100:.0f}%")
    st.metric("Glucose now", f"{fc.glucose_now:.0f} mg/dL")
    st.metric("Expected 2 h peak", f"{fc.predicted_peak:.0f} mg/dL")
    st.caption(f"Alert threshold {model.threshold * 100:.0f}% (tuned for ~80% sensitivity on validation patients)")

a, b = st.columns(2)
with a:
    st.subheader("Why? Forecast drivers")
    fig, ax = plt.subplots(figsize=(5.2, 3))
    plot_drivers(ax, fc)
    fig.tight_layout()
    st.pyplot(fig)
with b:
    st.subheader("Wearable stream")
    fig, ax = plt.subplots(figsize=(5.2, 3))
    plot_vitals(ax, twin, now)
    fig.tight_layout()
    st.pyplot(fig)

st.divider()
st.subheader("What-if simulator (counter-factual)")
w1, w2, w3 = st.columns(3)
walk = w1.checkbox("15-min walk after each main meal", True)
sleep = w2.slider("Minimum nightly sleep (h)", 0.0, 9.0, 0.0, 0.5, help="0 = patient's usual sleep")
carbs = w3.slider("Carbohydrate load vs usual", 0.6, 1.2, 1.0, 0.05)
iv = {"post_meal_walk": walk, "carb_scale": carbs}
if sleep > 0:
    iv["sleep_min_hours"] = sleep
res = twin.what_if(iv)
m1, m2, m3 = st.columns(3)
bs, it = res["baseline"], res["intervention"]
m1.metric("Time above 180 mg/dL", f"{it['time_above_180_pct']:.1f}%", f"{it['time_above_180_pct'] - bs['time_above_180_pct']:+.1f} pts", delta_color="inverse")
m2.metric("Time in range (70-180)", f"{it['time_in_range_pct']:.1f}%", f"{it['time_in_range_pct'] - bs['time_in_range_pct']:+.1f} pts")
m3.metric("Mean daily peak", f"{it['peak_daily_avg']:.0f} mg/dL", f"{it['peak_daily_avg'] - bs['peak_daily_avg']:+.0f}", delta_color="inverse")
fig, ax = plt.subplots(figsize=(10, 3))
plot_whatif(ax, res, day=day)
fig.tight_layout()
st.pyplot(fig)
st.caption("What-if runs the mechanistic layer of the twin with identical meals and noise, so differences are due only to the change.")
