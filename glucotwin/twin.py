"""
The Digital Twin: a living virtual replica of ONE patient.

State        = static EHR record  +  rolling wearable/CGM history
Forecast     = ML model -> P(excursion > 180 mg/dL within 2 h), expected peak, risk level, drivers
What-if      = mechanistic simulator re-run with a modified behaviour (walk after meals, sleep, carbs)

Hybrid design: data-driven forecasting (fast, patient-specific) + mechanistic simulation
(lets a doctor ask counter-factual questions that pure ML cannot answer).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_GROUPS, ehr_vector, wearable_features
from .model import GlucoForecaster
from .synthetic import BINS_PER_DAY, simulate_patient

GROUP_PLAIN_ENGLISH = {
    "EHR (static)": "Clinical profile (HbA1c, BMI, genetics, medicines)",
    "CGM history": "Recent glucose level and trend",
    "Activity": "Recent steps / movement",
    "HRV / stress": "Heart-rate variability (stress signal)",
    "Sleep": "Last 24 h sleep quantity and quality",
    "Time of day": "Time of day (meal-time habits)",
}


@dataclass
class Forecast:
    index: int
    timestamp: pd.Timestamp
    glucose_now: float
    probability: float
    predicted_peak: float
    risk_level: str
    alert: bool
    drivers: dict          # group -> change in probability attributable to that group


class VirtualPatient:
    """Digital twin of one patient."""

    def __init__(self, ehr_row: pd.Series, series: pd.DataFrame, model: GlucoForecaster, sim_seed: int | None = None):
        self.ehr = ehr_row
        self.series = series.reset_index(drop=True)
        self.model = model
        self.sim_seed = sim_seed
        self._feat = wearable_features(self.series)
        for k, v in ehr_vector(ehr_row).items():
            self._feat[k] = v

    @property
    def patient_id(self) -> str:
        return str(self.ehr["patient_id"])

    # ------------------------------------------------------------------ forecasting
    def _row(self, i: int) -> pd.DataFrame:
        return self._feat.iloc[[i]]

    def forecast(self, i: int, explain: bool = True) -> Forecast:
        row = self._row(i)
        p = float(self.model.predict_proba(row)[0])
        peak = float(self.model.predict_peak(row)[0])
        thr = self.model.threshold
        level = "HIGH" if p >= thr else "MODERATE" if p >= thr / 2 else "LOW"
        drivers = {}
        if explain:
            for gname, cols in FEATURE_GROUPS.items():
                cols = [c for c in cols if c in self.model.features]
                if not cols:
                    continue
                alt = row.copy()
                alt[cols] = self.model.medians[cols].values          # "replace with a typical patient"
                drivers[gname] = p - float(self.model.predict_proba(alt)[0])
        g_now = float(self.series.loc[i, "glucose"])
        return Forecast(i, self.series.loc[i, "ts"], g_now, p, peak, level, p >= thr, drivers)

    def forecast_range(self, start: int, end: int, step: int = 3) -> pd.DataFrame:
        idx = np.arange(start, end, step)
        rows = self._feat.iloc[idx]
        out = pd.DataFrame({"index": idx, "ts": self.series.loc[idx, "ts"].values,
                            "glucose": self.series.loc[idx, "glucose"].values,
                            "p": self.model.predict_proba(rows), "peak_pred": self.model.predict_peak(rows)})
        out["alert"] = out["p"] >= self.model.threshold
        return out

    # ------------------------------------------------------------------ what-if
    def what_if(self, interventions: dict, n_days: int | None = None) -> dict:
        """
        Counter-factual simulation: same meals, same noise, different behaviour.
        (PoC note: the mechanistic layer re-uses the synthetic physiology engine, parameterised by the twin's
        latent state. In a real deployment these parameters would be *estimated* from the patient's data.)
        """
        if self.sim_seed is None:
            raise ValueError("what_if requires the simulation seed of the synthetic patient")
        n_days = n_days or int(len(self.series) / BINS_PER_DAY)
        base, _ = simulate_patient(self.ehr, n_days, self.sim_seed)
        alt, _ = simulate_patient(self.ehr, n_days, self.sim_seed, interventions)
        # skip first day (warm-up) when summarising
        s = slice(BINS_PER_DAY, None)

        def summ(d):
            g = d["glucose"].values[s]
            return {"mean": float(g.mean()), "peak_daily_avg": float(pd.Series(g).groupby(np.arange(len(g)) // BINS_PER_DAY).max().mean()),
                    "time_above_180_pct": float((g > 180).mean() * 100),
                    "time_in_range_pct": float(((g >= 70) & (g <= 180)).mean() * 100)}
        return {"baseline": summ(base), "intervention": summ(alt), "baseline_series": base, "intervention_series": alt}
