"""
Forecasting model: gradient-boosted trees on the fused EHR + wearable feature set.

* Patient-level train / validation / test split (no patient appears in two splits).
* Classifier  -> P(glucose > 180 mg/dL at any time in the next 2 h)
* Regressor   -> expected peak glucose over the next 2 h (shown on the clinician dashboard)
* Isotonic calibration on the validation split so probabilities are interpretable.
* Ablations (EHR-only / CGM-only / wearable-only / fused) prove that *fusion* adds value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error,
                             roc_auc_score)

from .features import (ACTIVITY_FEATURES, ALL_FEATURES, CGM_FEATURES, EHR_FEATURES, FEATURE_GROUPS,
                       HRV_FEATURES, SLEEP_FEATURES, TIME_FEATURES)

FEATURE_SETS = {
    "EHR only": EHR_FEATURES,
    "CGM only": CGM_FEATURES + TIME_FEATURES,
    "Wearables only (no EHR)": CGM_FEATURES + ACTIVITY_FEATURES + HRV_FEATURES + SLEEP_FEATURES + TIME_FEATURES,
    "Fused EHR + wearables (GlucoTwin)": ALL_FEATURES,
}


# Cold start = brand-new patient: no personal CGM history yet (no 24 h stats, no "same time yesterday" signature)
_COLD_CGM = ["g_now", "g_d15", "g_d30", "g_d60", "g_mean30", "g_mean60", "g_std60", "g_max120", "g_min120"]
COLD_START_SETS = {
    "Cold start: wearables only": _COLD_CGM + ACTIVITY_FEATURES + HRV_FEATURES + SLEEP_FEATURES + TIME_FEATURES,
    "Cold start: wearables + EHR (fused)": EHR_FEATURES + _COLD_CGM + ACTIVITY_FEATURES + HRV_FEATURES
                                          + SLEEP_FEATURES + TIME_FEATURES,
}


def split_patients(df: pd.DataFrame, seed: int = 7, frac=(0.70, 0.15, 0.15)):
    """Patient-level split -> (train_ids, val_ids, test_ids)."""
    ids = np.array(df["patient_id"].unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n = len(ids)
    a, b = int(frac[0] * n), int((frac[0] + frac[1]) * n)
    return ids[:a], ids[a:b], ids[b:]


def _clf(seed=0):
    return HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=seed)


def _metrics(y, p, thr):
    pred = p >= thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "auroc": float(roc_auc_score(y, p)), "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)), "threshold": float(thr),
        "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
        "precision": tp / max(tp + fp, 1), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def pick_threshold(y, p, target_sensitivity=0.80):
    """Highest threshold that still reaches the target sensitivity on validation data."""
    best = 0.05
    for thr in np.linspace(0.02, 0.95, 94):
        sens = ((p >= thr) & (y == 1)).sum() / max((y == 1).sum(), 1)
        if sens >= target_sensitivity:
            best = thr
    return float(best)


class GlucoForecaster:
    """Bundle of classifier + calibrator + regressor + reference values for explanations."""

    def __init__(self, features=None, seed=0):
        self.features = list(features or ALL_FEATURES)
        self.seed = seed
        self.clf = _clf(seed)
        self.reg = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
                                                 min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
        self.cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.threshold = 0.3
        self.medians = None

    def fit(self, train: pd.DataFrame, val: pd.DataFrame):
        self.clf.fit(train[self.features], train["y"])
        self.cal.fit(self.clf.predict_proba(val[self.features])[:, 1], val["y"])
        self.reg.fit(train[self.features], train["fwd_max120"])
        self.medians = train[self.features].median()
        self.threshold = pick_threshold(val["y"].values, self.predict_proba(val))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.cal.predict(self.clf.predict_proba(df[self.features])[:, 1])

    def predict_peak(self, df: pd.DataFrame) -> np.ndarray:
        return self.reg.predict(df[self.features])


def run_experiment(df: pd.DataFrame, seed: int = 7, target_sensitivity: float = 0.80):
    """Train all variants; return (models, results dict, test dataframe with predictions)."""
    el = df[df["eligible"]].copy()
    tr_ids, va_ids, te_ids = split_patients(el, seed)
    tr, va, te = (el[el.patient_id.isin(s)].copy() for s in (tr_ids, va_ids, te_ids))

    results, models = {"n_rows": {"train": len(tr), "val": len(va), "test": len(te)},
                        "n_patients": {"train": len(tr_ids), "val": len(va_ids), "test": len(te_ids)},
                        "prevalence_test": float(te["y"].mean()), "ablation": {}}, {}

    for name, feats in FEATURE_SETS.items():
        m = GlucoForecaster(feats, seed).fit(tr, va)
        p = m.predict_proba(te)
        m.threshold = pick_threshold(va["y"].values, m.predict_proba(va), target_sensitivity)
        results["ablation"][name] = _metrics(te["y"].values, p, m.threshold)
        models[name] = m

    results["cold_start_ablation"] = {}
    for name, feats in COLD_START_SETS.items():
        m = GlucoForecaster(feats, seed).fit(tr, va)
        m.threshold = pick_threshold(va["y"].values, m.predict_proba(va), target_sensitivity)
        results["cold_start_ablation"][name] = _metrics(te["y"].values, m.predict_proba(te), m.threshold)

    # naive baselines
    rule = ((te["g_now"] > 140) & (te["g_d30"] > 10)).astype(float).values
    results["baseline_rule"] = {"description": "alert if glucose > 140 mg/dL and rising > 10 mg/dL per 30 min",
                                **{k: v for k, v in _metrics(te["y"].values, rule * 0.9 + 0.05, 0.5).items()
                                   if k in ("sensitivity", "specificity", "precision")}}
    lr = LogisticRegression(max_iter=500).fit(tr[["g_now", "g_d30"]], tr["y"])
    p_lr = lr.predict_proba(te[["g_now", "g_d30"]])[:, 1]
    thr_lr = pick_threshold(va["y"].values, lr.predict_proba(va[["g_now", "g_d30"]])[:, 1], target_sensitivity)
    results["baseline_logreg_current_glucose"] = _metrics(te["y"].values, p_lr, thr_lr)

    full = models["Fused EHR + wearables (GlucoTwin)"]
    te["p"] = full.predict_proba(te)
    te["peak_pred"] = full.predict_peak(te)
    results["regression"] = {"peak_mae_mgdl": float(mean_absolute_error(te["fwd_max120"], te["peak_pred"]))}
    results["alerts"] = alert_burden(te, full.threshold)
    return models, results, te


def alert_burden(te: pd.DataFrame, thr: float, cooldown_min: int = 60) -> dict:
    """Episode-level view: how many alerts per patient-day, and how many true excursions were pre-warned."""
    out_alerts, days, episodes, warned = 0, 0.0, 0, 0
    for pid, g in te.groupby("patient_id"):
        g = g.sort_values("ts")
        days += (g["ts"].max() - g["ts"].min()).total_seconds() / 86400
        last = None
        for ts_, p in zip(g["ts"], g["p"]):
            if p >= thr and (last is None or (ts_ - last).total_seconds() / 60 >= cooldown_min):
                out_alerts += 1
                last = ts_
        # an "episode" = contiguous run of y==1 rows; warned if any row in run had p >= thr
        run = (g["y"].diff().fillna(g["y"]) != 0).cumsum()
        for _, rg in g.groupby(run):
            if rg["y"].iloc[0] == 1:
                episodes += 1
                warned += int((rg["p"] >= thr).any())
    return {"alerts_per_patient_day": out_alerts / max(days, 1e-9), "episodes": episodes,
            "episodes_warned_in_advance": warned, "episode_recall": warned / max(episodes, 1),
            "cooldown_min": cooldown_min}


def group_importance(model: GlucoForecaster, te: pd.DataFrame, n: int = 15000, seed: int = 0) -> pd.DataFrame:
    """Permutation importance by feature *group* (drop in AUROC when the group is shuffled)."""
    rng = np.random.default_rng(seed)
    sub = te.sample(min(n, len(te)), random_state=seed).reset_index(drop=True)
    base = roc_auc_score(sub["y"], model.predict_proba(sub))
    rows = []
    for gname, cols in FEATURE_GROUPS.items():
        cols = [c for c in cols if c in model.features]
        if not cols:
            continue
        drops = []
        for _ in range(3):
            tmp = sub.copy()
            perm = rng.permutation(len(tmp))
            tmp[cols] = tmp[cols].values[perm]
            drops.append(base - roc_auc_score(tmp["y"], model.predict_proba(tmp)))
        rows.append({"group": gname, "auroc_drop": float(np.mean(drops))})
    return pd.DataFrame(rows).sort_values("auroc_drop", ascending=False)


def repeat_splits(df: pd.DataFrame, seeds=(7, 11, 23)) -> dict:
    """Re-run every ablation on several random patient splits -> mean / std of AUROC and AUPRC."""
    acc: dict[str, list] = {}
    for sd in seeds:
        _, res, _ = run_experiment(df, seed=sd)
        for group in ("ablation", "cold_start_ablation"):
            for name, m in res[group].items():
                acc.setdefault(name, []).append((m["auroc"], m["auprc"]))
    return {k: {"auroc_mean": float(np.mean([a for a, _ in v])), "auroc_std": float(np.std([a for a, _ in v])),
                "auprc_mean": float(np.mean([b for _, b in v])), "auprc_std": float(np.std([b for _, b in v])),
                "n_splits": len(v)} for k, v in acc.items()}
