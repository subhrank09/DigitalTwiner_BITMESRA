"""Matplotlib helpers shared by the report figures and the Streamlit clinician dashboard."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .twin import GROUP_PLAIN_ENGLISH, Forecast, VirtualPatient

TEAL, DARK, AMBER, RED, GREY, LIGHT = "#0F766E", "#134E4A", "#D97706", "#DC2626", "#6B7280", "#CCFBF1"
RISK_COLOR = {"LOW": TEAL, "MODERATE": AMBER, "HIGH": RED}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)


def plot_cgm_forecast(ax, twin: VirtualPatient, i: int, hours_back: int = 10):
    """CGM trace up to 'now' + the model's risk timeline underneath (as a shaded band) + 2 h horizon."""
    lo = max(0, i - hours_back * 12)
    hi = min(len(twin.series), i + 25)
    seg = twin.series.iloc[lo:hi]
    past = seg[seg.index <= i]
    fut = seg[seg.index >= i]
    ax.axhspan(70, 180, color=LIGHT, alpha=0.55, lw=0)
    ax.axhline(180, color=RED, lw=1, ls="--")
    ax.plot(past["ts"], past["glucose"], color=DARK, lw=2, label="CGM (observed)")
    ax.plot(fut["ts"], fut["glucose"], color=GREY, lw=1.4, ls=":", label="what actually happened next")
    fr = twin.forecast_range(lo, i + 1, 3)
    bad = fr[fr["alert"]]
    ax.scatter(bad["ts"], bad["glucose"], s=16, color=RED, zorder=5, label="twin alert issued")
    now = twin.series.loc[i, "ts"]
    ax.axvline(now, color=DARK, lw=1)
    ax.axvspan(now, now + pd.Timedelta(hours=2), color=AMBER, alpha=0.12, lw=0)
    ax.set_ylabel("Glucose (mg/dL)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylim(50, max(260, float(seg["glucose"].max()) + 15))
    ax.text(now + pd.Timedelta(minutes=6), ax.get_ylim()[0] + 6, "2 h forecast window", fontsize=8, color=AMBER)
    ax.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    _style(ax)


def plot_drivers(ax, fc: Forecast):
    items = sorted(fc.drivers.items(), key=lambda kv: kv[1])
    short = {"EHR (static)": "Clinical profile", "CGM history": "Glucose level + trend", "Activity": "Recent movement",
             "HRV / stress": "HRV (stress)", "Sleep": "Last 24 h sleep", "Time of day": "Time of day (meals)"}
    names = [short.get(k, k) for k, _ in items]
    vals = [v * 100 for _, v in items]
    ax.barh(names, vals, color=[RED if v > 0 else TEAL for v in vals])
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_xlabel("Effect on risk (percentage points)")
    ax.tick_params(axis="y", labelsize=8)
    _style(ax)


def plot_vitals(ax, twin: VirtualPatient, i: int, hours_back: int = 10):
    lo = max(0, i - hours_back * 12)
    seg = twin.series.iloc[lo:i + 1]
    ax.plot(seg["ts"], seg["hr"].rolling(3, min_periods=1).mean(), color=RED, lw=1.3, label="Heart rate (bpm)")
    ax2 = ax.twinx()
    ax2.plot(seg["ts"], seg["hrv_rmssd"].rolling(6, min_periods=1).mean(), color=TEAL, lw=1.3, label="HRV RMSSD (ms)")
    ax.bar(seg["ts"], seg["steps"] / 4, width=0.003, color=GREY, alpha=0.5, label="Steps (scaled)")
    ax.set_ylabel("bpm", fontsize=8)
    ax2.set_ylabel("ms", fontsize=8)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.tick_params(axis="x", labelsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="upper left", ncol=3)
    for s in ("top",):
        ax.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)


def plot_whatif(ax, res: dict, day: int = 3):
    a, b = day * 288, (day + 1) * 288
    base, alt = res["baseline_series"].iloc[a:b], res["intervention_series"].iloc[a:b]
    ax.axhspan(70, 180, color=LIGHT, alpha=0.55, lw=0)
    ax.axhline(180, color=RED, lw=1, ls="--")
    ax.plot(base["ts"], base["glucose"], color=GREY, lw=1.6, label="Current behaviour")
    ax.plot(alt["ts"], alt["glucose"], color=TEAL, lw=1.8, label="With the change")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_ylabel("Glucose (mg/dL)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax)


def clinician_dashboard(twin: VirtualPatient, i: int, whatif: dict, whatif_label: str, path: str):
    """Composite 'conceptual UI' image of the clinician view (used in the slides and README)."""
    fc = twin.forecast(i)
    e = twin.ehr
    fig = plt.figure(figsize=(15, 8.4), facecolor="white")
    gs = fig.add_gridspec(3, 12, height_ratios=[0.55, 2.3, 2.0], hspace=0.6, wspace=1.6,
                          left=0.045, right=0.985, top=0.95, bottom=0.12)
    # header
    hax = fig.add_subplot(gs[0, :]); hax.axis("off")
    hax.text(0, 0.72, "GlucoTwin  |  Virtual Patient Console", fontsize=17, fontweight="bold", color=DARK, va="center")
    hax.text(0, 0.15, f"Patient {twin.patient_id}   ·   {e['age']} y {e['sex']}   ·   BMI {e['bmi']}   ·   HbA1c {e['hba1c']}%   ·   "
                      f"Fasting {int(e['fasting_glucose'])} mg/dL   ·   Dx {e['diagnoses_icd10']}   ·   "
                      f"Metformin {'yes' if e['on_metformin'] else 'no'}   ·   TCF7L2 risk alleles {e['tcf7l2_risk_alleles']}",
             fontsize=9.5, color=GREY, va="center")
    hax.text(1, 0.5, f"{fc.risk_level} RISK", fontsize=16, fontweight="bold", color="white", ha="right", va="center",
             bbox=dict(boxstyle="round,pad=0.45", fc=RISK_COLOR[fc.risk_level], ec="none"))
    # CGM + forecast
    ax1 = fig.add_subplot(gs[1, :8])
    plot_cgm_forecast(ax1, twin, i)
    ax1.set_title(f"Live CGM and 2-hour excursion forecast  (now {fc.timestamp:%a %H:%M})", loc="left", fontsize=11, color=DARK)
    # forecast card
    cax = fig.add_subplot(gs[1, 8:]); cax.axis("off")
    cax.add_patch(plt.Rectangle((0, 0), 1, 1, fc="#F0FDFA", ec="#99F6E4", lw=1, transform=cax.transAxes))
    cax.text(0.06, 0.88, "Forecast for next 2 hours", fontsize=11, fontweight="bold", color=DARK)
    cax.text(0.06, 0.64, f"{fc.probability * 100:.0f}%", fontsize=34, fontweight="bold", color=RISK_COLOR[fc.risk_level])
    cax.text(0.06, 0.52, "chance glucose exceeds 180 mg/dL", fontsize=9.5, color=GREY)
    cax.text(0.06, 0.34, f"Glucose now: {fc.glucose_now:.0f} mg/dL", fontsize=10.5, color=DARK)
    cax.text(0.06, 0.24, f"Expected 2 h peak: {fc.predicted_peak:.0f} mg/dL", fontsize=10.5, color=DARK)
    cax.text(0.06, 0.10, f"Alert threshold: {twin.model.threshold * 100:.0f}%   ·   Alert: {'YES' if fc.alert else 'no'}",
             fontsize=9.5, color=GREY)
    # drivers
    ax2 = fig.add_subplot(gs[2, 2:5])
    plot_drivers(ax2, fc)
    ax2.set_title("Why? (what is driving this forecast)", loc="left", fontsize=11, color=DARK)
    # vitals
    ax3 = fig.add_subplot(gs[2, 5:8])
    plot_vitals(ax3, twin, i)
    ax3.set_title("Wearable stream (HR / HRV / steps)", loc="left", fontsize=11, color=DARK)
    # what-if
    ax4 = fig.add_subplot(gs[2, 9:])
    plot_whatif(ax4, whatif, day=i // 288)
    b, a = whatif["baseline"], whatif["intervention"]
    ax4.set_title(f"What-if: {whatif_label}", loc="left", fontsize=11, color=DARK)
    ax4.text(0.99, 0.04, f"Time >180: {b['time_above_180_pct']:.1f}% → {a['time_above_180_pct']:.1f}%",
             transform=ax4.transAxes, ha="right", fontsize=9, color=DARK,
             bbox=dict(boxstyle="round", fc="white", ec="#99F6E4"))
    fig.text(0.045, 0.015, "SYNTHETIC patient and data. Decision-support prototype only; not a medical device and not for clinical use.",
             fontsize=8, color=GREY)
    fig.savefig(path, dpi=160)
    plt.close(fig)
