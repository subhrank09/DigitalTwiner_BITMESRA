"""Fast sanity tests.  Run:  pytest -q"""
import numpy as np
import pandas as pd

from glucotwin.features import ALL_FEATURES, build_dataset
from glucotwin.model import GlucoForecaster, split_patients
from glucotwin.synthetic import BINS_PER_DAY, generate_cohort, make_ehr, simulate_patient
from glucotwin.twin import VirtualPatient


def _small():
    ehr, series, _ = generate_cohort(12, 8, seed=3)
    return ehr, series, build_dataset(ehr, series)


def test_ehr_is_synthetic_and_plausible():
    ehr = make_ehr(50, seed=1)
    assert ehr["patient_id"].is_unique
    assert ehr["hba1c"].between(5.6, 11).all()
    assert ehr["diagnoses_icd10"].str.contains("E11.9").all()


def test_simulator_shapes_and_ranges():
    ehr = make_ehr(2, seed=1)
    ts, meals = simulate_patient(ehr.iloc[0], n_days=3, seed=5)
    assert len(ts) == 3 * BINS_PER_DAY
    assert ts["glucose"].between(55, 400).all()
    assert set(ts["sleep_stage"].unique()) <= {-1, 0, 1, 2, 3}
    assert len(meals) >= 6


def test_what_if_is_paired_and_walk_helps():
    ehr = make_ehr(3, seed=2)
    row = ehr.iloc[0]
    base, _ = simulate_patient(row, 6, seed=9)
    same, _ = simulate_patient(row, 6, seed=9)
    walk, _ = simulate_patient(row, 6, seed=9, interventions={"post_meal_walk": True})
    assert (base["glucose"].values == same["glucose"].values).all()           # deterministic
    assert walk["glucose"].mean() < base["glucose"].mean()                     # walking lowers glucose


def test_no_leakage_features_use_only_past():
    """Changing FUTURE glucose must not change features at time t."""
    ehr, series, _ = _small()
    pid = ehr["patient_id"].iloc[0]
    ts = series[series.patient_id == pid].reset_index(drop=True)
    from glucotwin.features import wearable_features
    f1 = wearable_features(ts)
    t = 4 * BINS_PER_DAY
    ts2 = ts.copy()
    ts2.loc[t + 1:, "glucose"] = 999
    f2 = wearable_features(ts2)
    cols = [c for c in f1.columns if c not in ("prev_days_fwd_max_mean", "prev_days_fwd_max_max")]
    pd.testing.assert_series_equal(f1.loc[t, cols], f2.loc[t, cols])
    assert f1.loc[t, "prev_days_fwd_max_mean"] == f2.loc[t, "prev_days_fwd_max_mean"]


def test_patient_level_split_has_no_overlap():
    _, _, df = _small()
    a, b, c = split_patients(df)
    assert not (set(a) & set(b)) and not (set(a) & set(c)) and not (set(b) & set(c))


def test_train_forecast_and_explain():
    ehr, series, df = _small()
    el = df[df.eligible]
    a, b, _ = split_patients(el)
    m = GlucoForecaster(ALL_FEATURES).fit(el[el.patient_id.isin(a)], el[el.patient_id.isin(b)])
    i = ehr.index[0]
    twin = VirtualPatient(ehr.iloc[i], series[series.patient_id == ehr.patient_id[i]], m, sim_seed=3 * 1000 + i)
    fc = twin.forecast(5 * BINS_PER_DAY + 150)
    assert 0 <= fc.probability <= 1 and fc.risk_level in {"LOW", "MODERATE", "HIGH"}
    assert set(fc.drivers) == {"EHR (static)", "CGM history", "Activity", "HRV / stress", "Sleep", "Time of day"}
    wi = twin.what_if({"post_meal_walk": True})
    assert wi["intervention"]["mean"] <= wi["baseline"]["mean"]
