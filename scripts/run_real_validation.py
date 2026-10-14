"""Real-data validation of GlucoTwin on CGMacros.

Place at scripts/run_real_validation.py. Run from the repo root:
    python scripts/run_real_validation.py

Experiment A: train on synthetic, test on real (sim-to-real gap).
Experiment B: train and test on real, grouped CV by participant (3 repeats x 5 folds),
              with CGM-only features and CGM + the 7 EHR fields CGMacros shares.
Only CGM-derived + time-of-day (+ optional EHR subset) features are used; HRV, sleep and
steps do not exist in CGMacros and are never fed to the model on real data.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from glucotwin import features as F
from glucotwin import real_adapter as ra

EHR7 = ["age", "sex_male", "bmi", "hba1c", "fasting_glucose", "ldl", "triglycerides"]
LABELS = ("y", "label", "event", "target", "excursion", "y_event")


# ----------------------------------------------------------------- helpers
def build(ehr, series):
    try:
        return F.build_dataset(ehr, series)
    except Exception as e1:  # retry with string timestamps in case the code expects them
        s2 = series.copy()
        s2["ts"] = pd.to_datetime(s2["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        try:
            return F.build_dataset(ehr, s2)
        except Exception:
            raise e1


def find_label(ds):
    for c in LABELS:
        if c in ds.columns:
            return c
    for c in ds.columns:
        if c in F.ALL_FEATURES:
            continue
        try:
            vals = {float(v) for v in pd.unique(ds[c].dropna())}
        except (TypeError, ValueError):
            continue
        if vals and vals <= {0.0, 1.0}:
            print(f"[info] using '{c}' as the label column")
            return c
    raise SystemExit(f"Cannot find label column. Columns are: {list(ds.columns)}")


def require(ds, cols, name):
    miss = [c for c in cols if c not in ds.columns]
    if miss:
        raise SystemExit(f"{name}: dataset lacks feature columns {miss}. Columns: {list(ds.columns)}")


def new_model(seed=0):
    # Stand-in for glucotwin/model.py; match its hyper-parameters if they differ.
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=seed)


def point(y, p):
    y = np.asarray(y)
    if y.sum() < 10 or (1 - y).sum() < 10:
        return {"auroc": np.nan, "auprc": np.nan, "prev": float(y.mean()), "n": int(len(y)), "n_pos": int(y.sum())}
    prev = float(y.mean())
    ap = float(average_precision_score(y, p))
    return {"auroc": float(roc_auc_score(y, p)), "auprc": ap, "prev": prev,
            "auprc_lift": ap / prev, "n": int(len(y)), "n_pos": int(y.sum())}


def boot_ci(y, p, groups, n_boot, seed=0):
    """95% CI of AUROC, resampling whole participants."""
    rng = np.random.default_rng(seed)
    ids = pd.unique(groups)
    idx = {k: np.where(groups == k)[0] for k in ids}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids))
        ii = np.concatenate([idx[k] for k in pick])
        yy = y[ii]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, p[ii]))
    if len(vals) < 20:
        return [np.nan, np.nan]
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def report_block(name, df, y, p, n_boot):
    """Metrics overall and per clinical group, with participant-bootstrap CI."""
    out = {}
    for grp in ["all", "T2D", "prediabetes", "healthy"]:
        m = np.ones(len(df), bool) if grp == "all" else (df["group"].values == grp)
        if m.sum() == 0:
            continue
        r = point(y[m], p[m])
        r["n_participants"] = int(df.loc[m, "base_id"].nunique())
        # a bootstrap over fewer than 5 participants is meaningless -> leave the CI empty
        ok_ci = (not np.isnan(r["auroc"])) and r["n_participants"] >= 5
        r["auroc_ci95"] = boot_ci(y[m], p[m], df.loc[m, "base_id"].values, n_boot) if ok_ci else [np.nan, np.nan]
        out[grp] = r
        ci = r["auroc_ci95"]
        print(f"  {name:<28} {grp:<12} participants={r['n_participants']:<3} n={r['n']:<7} prev={r['prev']:.3f} "
              f"AUROC={r['auroc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] AUPRC={r['auprc']:.3f} (lift {r.get('auprc_lift', float('nan')):.1f}x)")
    return out


# ----------------------------------------------------------------- data
def load_synthetic():
    ehr = pd.read_csv("data/synthetic/ehr.csv")
    series = pd.read_csv("data/synthetic/wearables_5min.csv.gz", parse_dates=["ts"])
    ds = build(ehr, series)
    return ds


def load_real(root, cgm_source, min_days):
    series, ehr, meta, rep = ra.make_series(root, cgm_source, min_days)
    print(rep.to_string(index=False))
    ds = build(ehr, series)
    ds["base_id"] = ds["patient_id"].astype(str).str.rsplit("_s", n=1).str[0]
    ds = ds.merge(meta[["base_id", "group"]].drop_duplicates("base_id"), on="base_id", how="left")
    ds["group"] = ds["group"].fillna("unknown")
    return ds


# ----------------------------------------------------------------- experiments
def experiment_A(syn, real, feats, ylab, n_boot):
    print("\n== Experiment A: train on synthetic -> test on real ==")
    res = {}
    # like-for-like synthetic reference (patient-level 80/20 split, same features and model)
    rng = np.random.default_rng(0)
    pids = np.array(syn["patient_id"].unique(), dtype=object)
    rng.shuffle(pids)
    te = set(pids[: max(1, len(pids) // 5)])
    is_te = syn["patient_id"].isin(te).values
    m = new_model().fit(syn.loc[~is_te, feats], syn.loc[~is_te, ylab])
    ref = point(syn.loc[is_te, ylab].values, m.predict_proba(syn.loc[is_te, feats])[:, 1])
    print(f"  synthetic held-out reference    AUROC={ref['auroc']:.3f} AUPRC={ref['auprc']:.3f} prev={ref['prev']:.3f}")
    res["synthetic_reference"] = ref

    m = new_model().fit(syn[feats], syn[ylab])
    p = m.predict_proba(real[feats])[:, 1]
    res["real"] = report_block("synthetic->real", real, real[ylab].values.astype(int), p, n_boot)
    return res


def experiment_B(real, feats, ylab, tag, n_boot, repeats=3, k=5, save_oof=None):
    print(f"\n== Experiment B ({tag}): real data, grouped {k}-fold x {repeats} ==")
    y = real[ylab].values.astype(int)
    base = real["base_id"].values
    ids = np.array(sorted(pd.unique(base)))
    per_rep, oof0 = [], None
    for rep in range(repeats):
        rng = np.random.default_rng(100 + rep)
        order = rng.permutation(ids)
        fold_of = {pid: i % k for i, pid in enumerate(order)}
        folds = np.array([fold_of[b] for b in base])
        oof = np.zeros(len(real))
        for f in range(k):
            tr, te = folds != f, folds == f
            m = new_model(rep).fit(real.loc[tr, feats], y[tr])
            oof[te] = m.predict_proba(real.loc[te, feats])[:, 1]
        per_rep.append(point(y, oof))
        if rep == 0:
            oof0 = oof
    aur = [r["auroc"] for r in per_rep]
    apr = [r["auprc"] for r in per_rep]
    print(f"  {tag}: AUROC {np.mean(aur):.3f} +/- {np.std(aur):.3f} | AUPRC {np.mean(apr):.3f} +/- {np.std(apr):.3f} "
          f"| prevalence {per_rep[0]['prev']:.3f}")
    groups = report_block(f"B {tag} (repeat 0)", real, y, oof0, n_boot)
    if save_oof is not None:
        save_oof[tag] = oof0
    return {"mean_auroc": float(np.mean(aur)), "sd_auroc": float(np.std(aur)),
            "mean_auprc": float(np.mean(apr)), "sd_auprc": float(np.std(apr)),
            "repeat0_by_group": groups}


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/real/cgmacros")
    ap.add_argument("--cgm-source", default="Dexcom GL", help="'Dexcom GL' or 'Libre GL'")
    ap.add_argument("--min-days", type=float, default=3.0)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--out", default="artifacts/real_validation.json")
    ap.add_argument("--oof-out", default="artifacts/real_oof_predictions.csv")
    a = ap.parse_args()

    print("Building synthetic dataset ...")
    syn = load_synthetic()
    ylab = find_label(syn)
    if "eligible" in syn.columns:
        syn = syn[syn["eligible"].astype(bool)]
    print(f"synthetic rows={len(syn)} prevalence={syn[ylab].mean():.3f}")

    print("\nBuilding real dataset (CGMacros) ...")
    real = load_real(a.root, a.cgm_source, a.min_days)
    if "eligible" in real.columns:
        real = real[real["eligible"].astype(bool)]
    real = real.reset_index(drop=True)
    print(f"real rows={len(real)} participants={real['base_id'].nunique()} prevalence={real[ylab].mean():.3f}")
    print(real.drop_duplicates('base_id')['group'].value_counts().to_string())

    cgm = list(F.CGM_FEATURES) + list(F.TIME_FEATURES)
    cgm_ehr = cgm + EHR7
    require(syn, cgm, "synthetic")
    require(real, cgm_ehr, "real")

    results = {"cgm_source": a.cgm_source, "features_cgm": cgm, "ehr7": EHR7}
    results["experiment_A_cgm_only"] = experiment_A(syn, real, cgm, ylab, a.boot)
    oof = {}
    results["experiment_B_cgm_only"] = experiment_B(real, cgm, ylab, "CGM only", a.boot, save_oof=oof)
    results["experiment_B_cgm_plus_ehr7"] = experiment_B(real, cgm_ehr, ylab, "CGM + EHR(7)", a.boot, save_oof=oof)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    keep = [c for c in ["patient_id", "base_id", "group", "ts"] if c in real.columns]
    pred = real[keep].copy()
    pred["y"] = real[ylab].values
    for tag, v in oof.items():
        pred["p_" + tag.replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")] = v
    pred.to_csv(a.oof_out, index=False)
    print(f"\nSaved {a.out} and {a.oof_out}")


if __name__ == "__main__":
    main()