"""
Synthetic data engine (EHR + wearable streams) for GlucoTwin.

Everything here is SYNTHETIC. No real patient data is used (DPDP Act / HIPAA safe).

Two layers:
1. ``make_ehr``         - Synthea-style static patient records (demographics, labs, genetics, meds).
2. ``simulate_patient`` - a compact mechanistic simulator that produces 5-minute CGM glucose,
                          heart rate, HRV (RMSSD), step count and sleep stages for one patient.

The simulator is also the "physics engine" used by the twin for what-if scenarios
(e.g. "what if this patient walked 15 minutes after every main meal?").

Physiological ideas encoded (all simplified, literature-inspired):
* Fasting level is anchored to HbA1c via the ADA eAG formula (eAG = 28.7*HbA1c - 46.7).
* Post-meal excursion scales with carbohydrate load and an insulin-resistance (IR) latent variable
  derived from BMI, HbA1c, TCF7L2 risk alleles and family history.
* Short sleep (<7 h) raises next-day IR; psychological stress raises glucose and lowers HRV.
* Walking after a meal blunts the excursion; steps also lower glucose directly.
* Indian context: rice / roti / idli heavy meals, high carbohydrate share, vegetarian vs non-vegetarian.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BIN_MIN = 5
BINS_PER_DAY = 288
START = pd.Timestamp("2026-01-05 00:00")


def estimated_avg_glucose(hba1c: float) -> float:
    """ADA formula linking HbA1c (%) to estimated average glucose (mg/dL)."""
    return 28.7 * hba1c - 46.7


# ----------------------------------------------------------------------------------------------
# 1. Static EHR
# ----------------------------------------------------------------------------------------------
def make_ehr(n_patients: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generate Synthea-style static EHR records for a T2D / pre-diabetic Indian cohort."""
    rng = np.random.default_rng(seed)
    n = n_patients
    age = np.clip(rng.normal(52, 10, n), 28, 80).round().astype(int)
    sex = rng.choice(["M", "F"], n, p=[0.55, 0.45])
    bmi = np.clip(rng.normal(27.5, 4.2, n), 18.5, 42).round(1)
    alleles = rng.binomial(2, 0.30, n)  # TCF7L2 rs7903146 T-allele count (risk allele)
    fam = rng.binomial(1, 0.5, n)
    years = np.clip(rng.exponential(5, n), 0, 25).round(1)
    hba1c = np.clip(6.8 + 0.045 * (bmi - 27) + 0.05 * years + 0.2 * alleles + rng.normal(0, 0.8, n), 5.6, 11.0).round(1)
    eag = 28.7 * hba1c - 46.7
    fasting = np.clip(0.5 * eag + 50 + rng.normal(0, 8, n), 85, 220).round()
    sbp = np.clip(112 + 0.5 * (age - 50) + 0.6 * (bmi - 25) + rng.normal(0, 10, n), 100, 185).round()
    dbp = np.clip(0.55 * sbp + 10 + rng.normal(0, 5, n), 60, 110).round()
    ldl = np.clip(rng.normal(110, 30, n), 50, 220).round()
    tg = np.clip(rng.normal(165, 55, n) + 6 * (bmi - 27), 60, 450).round()
    egfr = np.clip(100 - 0.6 * (age - 40) - 1.0 * years + rng.normal(0, 8, n), 25, 120).round()
    metformin = ((hba1c > 6.4) & (rng.random(n) < 0.85)).astype(int)
    sulfonylurea = ((hba1c > 7.5) & (rng.random(n) < 0.40)).astype(int)
    p_cvd = 1 / (1 + np.exp(-(-3.2 + 0.04 * (age - 50) + 0.15 * (sbp - 130) / 10 + 0.25 * years / 5)))
    prior_cvd = (rng.random(n) < p_cvd).astype(int)
    hypertension = ((sbp >= 140) | (rng.random(n) < 0.25)).astype(int)
    diet = rng.choice(["veg", "non_veg"], n, p=[0.40, 0.60])
    activity = rng.choice(["low", "moderate", "high"], n, p=[0.45, 0.40, 0.15])

    # Hidden latent variables that drive the simulator but are NOT part of the public EHR
    ir = np.clip(0.45 + 0.03 * (bmi - 25) + 0.12 * (hba1c - 6.5) + 0.12 * alleles + 0.05 * fam
                 + rng.normal(0, 0.08, n), 0.3, 1.6)
    habit_map = {"low": 0.15, "moderate": 0.40, "high": 0.65}
    walk_habit = np.clip(np.array([habit_map[a] for a in activity]) + rng.normal(0, 0.1, n), 0.02, 0.95)
    hr_adj = {"low": 0, "moderate": -3, "high": -6}
    rest_hr = np.clip(64 + 0.3 * (bmi - 25) + np.array([hr_adj[a] for a in activity]) + rng.normal(0, 5, n), 50, 95)

    def dx(i):
        codes = ["E11.9"]  # Type 2 diabetes mellitus without complications
        if hypertension[i]:
            codes.append("I10")
        if prior_cvd[i]:
            codes.append("I25.1")
        if egfr[i] < 60:
            codes.append("N18.3")
        if bmi[i] >= 30:
            codes.append("E66.9")
        return ";".join(codes)

    return pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age": age, "sex": sex, "bmi": bmi, "hba1c": hba1c, "fasting_glucose": fasting,
        "sbp": sbp, "dbp": dbp, "ldl": ldl, "triglycerides": tg, "egfr": egfr,
        "years_since_dx": years, "family_history": fam, "tcf7l2_risk_alleles": alleles,
        "diet": diet, "activity_level": activity, "on_metformin": metformin,
        "on_sulfonylurea": sulfonylurea, "prior_cvd": prior_cvd, "hypertension": hypertension,
        "diagnoses_icd10": [dx(i) for i in range(n)],
        "latent_ir": ir.round(3), "latent_walk_habit": walk_habit.round(3), "latent_rest_hr": rest_hr.round(1),
    })


# ----------------------------------------------------------------------------------------------
# 2. Wearable / CGM simulator
# ----------------------------------------------------------------------------------------------
def _meal_curve(tau_min: np.ndarray, tp: float) -> np.ndarray:
    """Normalised post-meal glucose rise (peak value 1 at tau = tp)."""
    x = np.clip(tau_min, 0, None) / tp
    return x ** 2 * np.exp(2 * (1 - x))


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(x, 0, 0.0))
    idx = np.arange(1, len(x) + 1)
    return c[idx] - c[np.maximum(idx - w, 0)]


def simulate_patient(ehr: pd.Series, n_days: int = 14, seed: int = 0,
                     interventions: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate ``n_days`` of 5-minute wearable data for one patient.

    interventions (all optional; random draws are made *before* they are applied so the
    baseline and counter-factual runs share identical meals/noise -> clean what-if deltas):
        post_meal_walk : bool   add a ~15 min walk 20-35 min after breakfast/lunch/dinner
        sleep_min_hours: float  enforce a minimum nightly sleep duration
        carb_scale     : float  multiply all meal carbohydrate loads
    Returns (timeseries_df, meals_df).
    """
    iv = interventions or {}
    rng = np.random.default_rng(seed)
    N = n_days * BINS_PER_DAY
    t = np.arange(N)
    hour = (t % BINS_PER_DAY) * BIN_MIN / 60.0
    day = t // BINS_PER_DAY
    ir0 = float(ehr["latent_ir"])
    veg = 1.10 if ehr["diet"] == "veg" else 1.0

    # ---------------- randomness drawn up-front ----------------
    bed_h = rng.normal(22.75, 0.9, n_days + 1)
    sleep_dur = np.clip(rng.normal(6.6, 1.1, n_days + 1), 4.0, 9.0)
    if "sleep_min_hours" in iv:
        sleep_dur = np.maximum(sleep_dur, iv["sleep_min_hours"])
    meal_t = {"breakfast": rng.normal(8.0, 0.6, n_days), "lunch": rng.normal(13.25, 0.6, n_days),
              "snack": rng.normal(17.0, 0.6, n_days), "dinner": rng.normal(20.5, 0.7, n_days)}
    meal_c = {"breakfast": rng.uniform(30, 75, n_days) * veg, "lunch": rng.uniform(55, 110, n_days) * veg,
              "snack": rng.uniform(10, 35, n_days), "dinner": rng.uniform(45, 95, n_days) * veg}
    snack_on = rng.random(n_days) < 0.5
    meal_tp = rng.uniform(45, 70, (n_days, 4))
    n_bouts = rng.integers(2, 7, n_days)
    bout_start = rng.uniform(7, 21, (n_days, 6))
    bout_len = rng.integers(2, 9, (n_days, 6))
    bout_int = rng.uniform(250, 520, (n_days, 6))
    eve_walk = rng.random(n_days) < float(ehr["latent_walk_habit"])
    eve_len = rng.integers(4, 7, n_days)
    n_stress = np.minimum(rng.poisson(1.0, n_days), 4)
    st_start = rng.uniform(9, 20, (n_days, 4))
    st_len = rng.integers(6, 25, (n_days, 4))
    st_int = rng.uniform(0.3, 1.0, (n_days, 4))
    daily_off = rng.normal(0, 5, n_days)
    noise_g = rng.normal(0, 2.0, N)
    noise_hr = rng.normal(0, 2.0, N)
    noise_hrv = rng.normal(0, 3.0, N)
    fid = rng.poisson(6, N)
    stage_u = rng.random((n_days + 1, 140))

    # ---------------- sleep ----------------
    in_bed = np.zeros(N, bool)
    stage = np.full(N, -1, int)
    for k in range(n_days + 1):
        s = int(round(((k - 1) * 24 + bed_h[k]) * 12))
        e = s + int(round(sleep_dur[k] * 12))
        for c, a in enumerate(range(s, e, 3)):
            b = min(a + 3, e)
            if b <= 0 or a >= N:
                continue
            prog = (a - s) / max(e - s, 1)
            phase = ((a - s) * 5 % 90) / 90
            u = stage_u[k, min(c, 139)]
            p_deep = 0.30 * (1 - prog) * (phase < 0.45)
            p_rem = 0.05 + 0.25 * prog * (phase > 0.55)
            st = 2 if u < p_deep else 3 if u < p_deep + p_rem else 0 if u > 0.975 else 1
            lo, hi = max(a, 0), min(b, N)
            stage[lo:hi] = st
            in_bed[lo:hi] = True
    sleep_dur_day = sleep_dur[:n_days]          # night ending the morning of day d
    ir_day = ir0 * (1 + 0.10 * np.maximum(0, 7 - sleep_dur_day))

    # ---------------- steps ----------------
    steps = fid.astype(float) * np.where((hour >= 7) & (hour <= 22), 1.0, 0.0)
    steps[in_bed] = 0
    for d in range(n_days):
        base_i = d * BINS_PER_DAY
        for j in range(n_bouts[d]):
            s = int(bout_start[d, j] * 12) + base_i
            steps[s:s + bout_len[d, j]] += bout_int[d, j]
        if eve_walk[d]:
            s = int((meal_t["dinner"][d] + 0.4) * 12) + base_i
            steps[s:s + eve_len[d]] += 380
    if iv.get("post_meal_walk"):
        for d in range(n_days):
            for m in ("breakfast", "lunch", "dinner"):
                s = int(meal_t[m][d] * 12) + 4 + d * BINS_PER_DAY
                steps[s:s + 3] += 430
    steps = np.clip(steps, 0, None)[:N]
    steps[in_bed] = 0

    # ---------------- stress ----------------
    stress = np.zeros(N)
    for d in range(n_days):
        for j in range(n_stress[d]):
            s = int(st_start[d, j] * 12) + d * BINS_PER_DAY
            L = int(st_len[d, j])
            shape = st_int[d, j] * np.sin(np.linspace(0.2, np.pi - 0.2, L))
            stress[s:s + L] = np.maximum(stress[s:s + L], shape[:len(stress[s:s + L])])
    stress = stress[:N]

    # ---------------- meals -> glucose ----------------
    med = (0.85 if ehr["on_metformin"] else 1.0) * (0.90 if ehr["on_sulfonylurea"] else 1.0)
    carb_scale = iv.get("carb_scale", 1.0)
    g_meal = np.zeros(N + 60)
    meal_rows = []
    names = ["breakfast", "lunch", "snack", "dinner"]
    for d in range(n_days):
        for mi, m in enumerate(names):
            if m == "snack" and not snack_on[d]:
                continue
            t0 = int(meal_t[m][d] * 12) + d * BINS_PER_DAY
            if t0 >= N - 2:
                continue
            carbs = meal_c[m][d] * carb_scale
            walk = steps[t0 + 2:t0 + 10].sum()
            walk_f = 1 - 0.30 * min(walk / 1500.0, 1.0)
            amp = 0.95 * carbs * (0.35 + ir_day[d]) * med * walk_f
            tau = np.arange(60) * BIN_MIN
            g_meal[t0:t0 + 60] += amp * _meal_curve(tau, meal_tp[d, mi])
            meal_rows.append({"ts": START + pd.Timedelta(minutes=t0 * BIN_MIN), "meal": m, "carbs_g": round(carbs, 1)})
    g_meal = g_meal[:N]

    # ---------------- glucose ----------------
    fasting = float(ehr["fasting_glucose"])
    base = fasting * (1 + 0.22 * (ir_day / ir0 - 1)) + daily_off
    dawn = 12 * np.exp(-(((hour - 6.0) / 1.3) ** 2)) * (ir_day[day] / ir0)
    steps30 = _rolling_sum(steps, 6)
    ex_eff = -np.minimum(0.010 * steps30, 45)
    stress_eff = 28 * stress * (0.5 + ir_day[day])
    e = np.zeros(N)
    for i in range(1, N):
        e[i] = 0.85 * e[i - 1] + noise_g[i]
    g = np.clip(base[day] + dawn + g_meal + ex_eff + stress_eff + e, 55, 400)

    # ---------------- heart rate / HRV ----------------
    rest = float(ehr["latent_rest_hr"])
    hr = rest + 0.085 * steps + 10 * stress - 7 * in_bed - 3 * (stage == 2) + 0.03 * g_meal + noise_hr
    hrv_base = float(np.clip(48 - 0.35 * (ehr["age"] - 40) - 8 * (ir0 - 0.6), 15, 60))
    sleep_penalty = 1 - 0.04 * np.maximum(0, 7 - sleep_dur_day)[day]
    hrv = hrv_base * sleep_penalty * (1 + 0.25 * in_bed) - 14 * stress * (hrv_base / 40) - 0.012 * steps + noise_hrv
    hr = np.clip(hr, 40, 190).round()
    hrv = np.clip(hrv, 8, 90).round(1)

    ts = START + pd.to_timedelta(t * BIN_MIN, unit="m")
    df = pd.DataFrame({"ts": ts, "glucose": g.round().astype(int), "hr": hr.astype(int), "hrv_rmssd": hrv,
                       "steps": steps.round().astype(int), "sleep_stage": stage})
    return df, pd.DataFrame(meal_rows)


def generate_cohort(n_patients: int = 150, n_days: int = 14, seed: int = 42):
    """Return (ehr_df, timeseries_df_all_patients, meals_df_all_patients)."""
    ehr = make_ehr(n_patients, seed)
    series, meals = [], []
    for i, row in ehr.iterrows():
        ts, ml = simulate_patient(row, n_days=n_days, seed=seed * 1000 + i)
        ts.insert(0, "patient_id", row["patient_id"])
        ml.insert(0, "patient_id", row["patient_id"])
        series.append(ts)
        meals.append(ml)
    return ehr, pd.concat(series, ignore_index=True), pd.concat(meals, ignore_index=True)
