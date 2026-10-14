"""
Feature engineering: fuse the static EHR stream with the dynamic wearable/CGM stream.

For every forecast time ``t`` (every 15 min) we build features using ONLY data up to ``t``
(no leakage) and a label that looks 2 hours ahead:

    y = 1  if  max(glucose in (t, t+120 min]) > 180 mg/dL      (hyperglycaemic excursion)
    eligible only if glucose(t) <= 160 mg/dL                    (we forecast *new* excursions,
                                                                 not ones already under way)
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .synthetic import BINS_PER_DAY

HORIZON_BINS = 24          # 120 minutes
THRESHOLD = 180.0          # mg/dL, ADA "time above range" level
ELIGIBLE_MAX = 160.0
STRIDE = 3                 # sample every 15 minutes

EHR_FEATURES = ["age", "sex_male", "bmi", "hba1c", "fasting_glucose", "sbp", "ldl", "triglycerides", "egfr",
                "years_since_dx", "family_history", "tcf7l2_risk_alleles", "diet_veg", "on_metformin",
                "on_sulfonylurea", "prior_cvd"]
CGM_FEATURES = ["g_now", "g_d15", "g_d30", "g_d60", "g_mean30", "g_mean60", "g_std60", "g_max120", "g_min120",
                "g24_mean", "g24_std", "g24_cv", "g24_above180", "g24_tir", "prev_days_fwd_max_mean",
                "prev_days_fwd_max_max"]
ACTIVITY_FEATURES = ["steps30", "steps60", "steps120", "hr_excess15"]
HRV_FEATURES = ["hrv_drop60", "hrv_mean60"]
SLEEP_FEATURES = ["sleep_h24", "deep_frac24", "rem_frac24", "awake_bins24"]
TIME_FEATURES = ["hour_sin", "hour_cos"]

FEATURE_GROUPS = {
    "EHR (static)": EHR_FEATURES,
    "CGM history": CGM_FEATURES,
    "Activity": ACTIVITY_FEATURES,
    "HRV / stress": HRV_FEATURES,
    "Sleep": SLEEP_FEATURES,
    "Time of day": TIME_FEATURES,
}
ALL_FEATURES = [c for g in FEATURE_GROUPS.values() for c in g]


def ehr_vector(row: pd.Series) -> dict:
    """Encode one EHR row into numeric features."""
    return {
        "age": row["age"], "sex_male": int(row["sex"] == "M"), "bmi": row["bmi"], "hba1c": row["hba1c"],
        "fasting_glucose": row["fasting_glucose"], "sbp": row["sbp"], "ldl": row["ldl"],
        "triglycerides": row["triglycerides"], "egfr": row["egfr"], "years_since_dx": row["years_since_dx"],
        "family_history": row["family_history"], "tcf7l2_risk_alleles": row["tcf7l2_risk_alleles"],
        "diet_veg": int(row["diet"] == "veg"), "on_metformin": row["on_metformin"],
        "on_sulfonylurea": row["on_sulfonylurea"], "prior_cvd": row["prior_cvd"],
    }


def wearable_features(ts: pd.DataFrame) -> pd.DataFrame:
    """Rolling, leak-free wearable/CGM features for every 5-minute bin of one patient."""
    g = ts["glucose"].astype(float)
    f = pd.DataFrame(index=ts.index)
    f["g_now"] = g
    for name, k in [("g_d15", 3), ("g_d30", 6), ("g_d60", 12)]:
        f[name] = g - g.shift(k)
    f["g_mean30"] = g.rolling(6, min_periods=1).mean()
    f["g_mean60"] = g.rolling(12, min_periods=1).mean()
    f["g_std60"] = g.rolling(12, min_periods=2).std()
    f["g_max120"] = g.rolling(24, min_periods=1).max()
    f["g_min120"] = g.rolling(24, min_periods=1).min()
    r24 = g.rolling(BINS_PER_DAY, min_periods=BINS_PER_DAY // 2)
    f["g24_mean"] = r24.mean()
    f["g24_std"] = r24.std()
    f["g24_cv"] = f["g24_std"] / f["g24_mean"]
    f["g24_above180"] = (g > THRESHOLD).astype(float).rolling(BINS_PER_DAY, min_periods=BINS_PER_DAY // 2).mean()
    f["g24_tir"] = ((g >= 70) & (g <= 180)).astype(float).rolling(BINS_PER_DAY, min_periods=BINS_PER_DAY // 2).mean()

    # "What happened in the 2h after this same clock time on previous days?"  (patient habit signature)
    fwd_max = pd.concat([g.shift(-k) for k in range(1, HORIZON_BINS + 1)], axis=1).max(axis=1).values
    n = len(g)
    stacks = []
    for d in range(1, 8):
        idx = np.arange(n) - d * BINS_PER_DAY
        vals = np.where(idx >= 0, fwd_max[np.clip(idx, 0, n - 1)], np.nan)
        stacks.append(vals)
    stacks = np.vstack(stacks)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # early rows have no history yet -> NaN
        f["prev_days_fwd_max_mean"] = np.nanmean(stacks, axis=0)
        f["prev_days_fwd_max_max"] = np.nanmax(stacks, axis=0)

    steps = ts["steps"].astype(float)
    f["steps30"] = steps.rolling(6, min_periods=1).sum()
    f["steps60"] = steps.rolling(12, min_periods=1).sum()
    f["steps120"] = steps.rolling(24, min_periods=1).sum()
    hr = ts["hr"].astype(float)
    rest = hr.rolling(BINS_PER_DAY, min_periods=BINS_PER_DAY // 2).quantile(0.10)
    f["hr_excess15"] = hr.rolling(3, min_periods=1).mean() - rest

    hrv = ts["hrv_rmssd"].astype(float)
    awake_hrv = hrv.where(ts["sleep_stage"] < 0)
    hrv_ref = awake_hrv.rolling(BINS_PER_DAY, min_periods=24).mean().ffill()
    f["hrv_mean60"] = hrv.rolling(12, min_periods=1).mean()
    f["hrv_drop60"] = f["hrv_mean60"] - hrv_ref

    st = ts["sleep_stage"]
    asleep = (st >= 0).astype(float)
    bins24 = asleep.rolling(BINS_PER_DAY, min_periods=1).sum()
    f["sleep_h24"] = bins24 * 5 / 60
    denom = bins24.replace(0, np.nan)
    f["deep_frac24"] = (st == 2).astype(float).rolling(BINS_PER_DAY, min_periods=1).sum() / denom
    f["rem_frac24"] = (st == 3).astype(float).rolling(BINS_PER_DAY, min_periods=1).sum() / denom
    f["awake_bins24"] = (st == 0).astype(float).rolling(BINS_PER_DAY, min_periods=1).sum()

    h = ((ts["ts"].dt.hour * 60 + ts["ts"].dt.minute) / 60.0).values
    f["hour_sin"] = np.sin(2 * np.pi * h / 24)
    f["hour_cos"] = np.cos(2 * np.pi * h / 24)
    return f


def build_dataset(ehr: pd.DataFrame, series: pd.DataFrame, burn_in_days: int = 2) -> pd.DataFrame:
    """Build the supervised dataset (one row per patient per 15-min forecast time)."""
    rows = []
    ehr_idx = ehr.set_index("patient_id")
    for pid, ts in series.groupby("patient_id", sort=False):
        ts = ts.reset_index(drop=True)
        f = wearable_features(ts)
        g = ts["glucose"].values.astype(float)
        n = len(g)
        fwd = np.full(n, np.nan)
        for k in range(1, HORIZON_BINS + 1):
            shifted = np.full(n, -np.inf)
            shifted[: n - k] = g[k:]
            fwd = np.fmax(np.where(np.isnan(fwd), -np.inf, fwd), shifted)
        fwd[n - HORIZON_BINS:] = np.nan
        idx = np.arange(burn_in_days * BINS_PER_DAY, n - HORIZON_BINS, STRIDE)
        sub = f.iloc[idx].copy()
        sub.insert(0, "patient_id", pid)
        sub.insert(1, "ts", ts["ts"].values[idx])
        for k, v in ehr_vector(ehr_idx.loc[pid]).items():
            sub[k] = v
        sub["fwd_max120"] = fwd[idx]
        sub["y"] = (sub["fwd_max120"] > THRESHOLD).astype(int)
        rows.append(sub)
    df = pd.concat(rows, ignore_index=True)
    df["eligible"] = df["g_now"] <= ELIGIBLE_MAX
    return df
