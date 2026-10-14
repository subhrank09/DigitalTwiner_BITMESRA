"""Draws docs/architecture.pdf (+ .png): the GlucoTwin system architecture."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
TEAL, DARK, AMBER, LIGHT, GREY = "#0F766E", "#134E4A", "#D97706", "#CCFBF1", "#6B7280"

fig, ax = plt.subplots(figsize=(14, 7.6))
ax.set_xlim(0, 14); ax.set_ylim(0, 7.6); ax.axis("off")


def box(x, y, w, h, title, lines=(), fc="white", ec=TEAL, tc=DARK, tsize=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.6))
    ax.text(x + 0.15, y + h - 0.25, title, fontsize=tsize, fontweight="bold", color=tc, va="center")
    for k, ln in enumerate(lines):
        ax.text(x + 0.15, y + h - 0.6 - 0.27 * k, ln, fontsize=8.6, color="#374151", va="center")


def arrow(x1, y1, x2, y2, label=None, color=DARK):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, lw=1.8, color=color))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.17, label, fontsize=8, color=color, ha="center")


ax.text(0.2, 7.25, "GlucoTwin - Digital Twin architecture (Type 2 Diabetes, 2-hour glucose-excursion forecast)",
        fontsize=14, fontweight="bold", color=DARK)

# Column headers
for x, t in [(0.2, "1  DATA STREAMS (synthetic)"), (4.35, "2  FUSION"), (7.45, "3  TWIN CORE"), (11.1, "4  CLINICIAN VIEW")]:
    ax.text(x, 6.75, t, fontsize=9.5, color=GREY, fontweight="bold")

# 1 data
box(0.2, 4.55, 3.7, 1.95, "Static / historical stream", ["Synthetic EHR (Synthea-style)", "Demographics, BMI, HbA1c, fasting glucose",
    "BP, LDL, TG, eGFR, ICD-10 diagnoses", "TCF7L2 genetic risk alleles, family history", "Medication: metformin, sulfonylurea"])
box(0.2, 2.25, 3.7, 2.0, "Dynamic / real-time stream", ["Simulated CGM glucose (5-min)", "Heart rate and HRV (RMSSD)",
    "Steps / activity", "Sleep stages (awake, light, deep, REM)", "Mimics Apple Health / Google Fit / CGM feeds"])
box(0.2, 0.55, 3.7, 1.35, "Patient behaviour (hidden)", ["Indian meal patterns, stress events,", "sleep debt, walking habits drive physiology"],
    fc="#F3F4F6", ec="#9CA3AF", tc=GREY)

# 2 fusion
box(4.35, 3.0, 2.75, 3.5, "Feature fusion", ["Leak-free rolling windows:", "  glucose level, slope, variability",
    "  24 h time-in-range", "  same clock-time, prior days", "  steps, HR excess, HRV drop", "  24 h sleep quality",
    "EHR encoder (16 fields)", "", "= 44-feature fused vector", "  sampled every 15 min"], fc=LIGHT)
arrow(3.9, 5.5, 4.35, 5.0); arrow(3.9, 3.25, 4.35, 3.9)

# 3 core
box(7.45, 4.45, 3.35, 2.05, "Forecast model (ML)", ["Gradient-boosted trees", "P(glucose > 180 mg/dL in next 2 h)",
    "Isotonic calibration + alert threshold", "Expected 2 h peak (regressor)", "Group-level explanations"], fc=LIGHT)
box(7.45, 2.4, 3.35, 1.8, "Mechanistic simulator", ["Meal response, insulin resistance,", "sleep debt, stress, exercise",
    "Counter-factual 'what-if' runs", "with paired meals and noise"], fc="#FEF3C7", ec=AMBER)
box(7.45, 0.55, 3.35, 1.6, "Twin state (VirtualPatient)", ["EHR + rolling wearable history", "forecast(t)  ·  what_if(change)"], fc="white")
arrow(7.1, 5.2, 7.45, 5.45); arrow(7.1, 3.8, 7.45, 3.3, color=AMBER)
arrow(9.1, 2.15, 9.1, 2.4, color=GREY); arrow(9.1, 4.2, 9.1, 4.45, color=GREY)

# 4 UI
box(11.1, 3.0, 2.7, 3.5, "Doctor dashboard", ["Streamlit app", "", "Risk level + probability", "Live CGM + 2 h window",
    "Why? (driver bars)", "HR / HRV / steps panel", "What-if simulator", "Patient EHR card"], fc="#F0FDFA")
arrow(10.8, 5.5, 11.1, 5.4); arrow(10.8, 3.3, 11.1, 3.9, color=AMBER)
box(11.1, 0.55, 2.7, 2.15, "Alert", ["Early warning up to", "2 h ahead of the", "excursion, so the", "patient / care team", "can act"], fc="white", ec="#DC2626", tc="#DC2626")
arrow(12.45, 3.0, 12.45, 2.7, color="#DC2626")

# bottom governance bar
ax.add_patch(FancyBboxPatch((3.0, 0.1), 9.0, 0.3, boxstyle="round,pad=0.01,rounding_size=0.08", fc=DARK, ec=DARK))
ax.text(7.5, 0.25, "Synthetic data only  ·  patient-level train/val/test split  ·  DPDP-safe  ·  MIT licence", color="white",
        fontsize=9, ha="center", va="center")
fig.savefig(ROOT / "docs" / "architecture.pdf", bbox_inches="tight")
fig.savefig(ROOT / "docs" / "figures" / "architecture.png", dpi=170, bbox_inches="tight")
print("saved docs/architecture.pdf")
