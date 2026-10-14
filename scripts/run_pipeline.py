"""
End-to-end pipeline:  synthetic data -> features -> train/evaluate -> artifacts + figures.

    python scripts/run_pipeline.py            # full run (~5 min on a laptop)
    python scripts/run_pipeline.py --quick    # small cohort, single split (~1 min)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from glucotwin.features import build_dataset
from glucotwin.model import group_importance, repeat_splits, run_experiment
from glucotwin.plots import DARK, GREY, RED, TEAL, AMBER, clinician_dashboard, plot_whatif
from glucotwin.synthetic import generate_cohort
from glucotwin.twin import VirtualPatient


def main(quick: bool):
    n_pat = 40 if quick else 150
    data_dir, art = ROOT / "data" / "synthetic", ROOT / "artifacts"
    fig_dir = ROOT / "docs" / "figures"
    for d in (data_dir, art, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Generating synthetic cohort ({n_pat} patients x 14 days, 5-min resolution)")
    ehr, series, meals = generate_cohort(n_pat, 14, seed=42)
    ehr.drop(columns=[c for c in ehr.columns if c.startswith("latent_")]).to_csv(data_dir / "ehr.csv", index=False)
    ehr.to_csv(data_dir / "ehr_with_latent_truth.csv", index=False)   # needed by the simulator-based what-if
    series.to_csv(data_dir / "wearables_5min.csv.gz", index=False)
    meals.to_csv(data_dir / "meals_groundtruth.csv", index=False)

    print("[2/6] Building fused feature table")
    df = build_dataset(ehr, series)
    el = df[df.eligible]
    print(f"      rows={len(df):,}  eligible={len(el):,}  excursion prevalence={el.y.mean():.3f}")

    print("[3/6] Training + evaluating (patient-level split)")
    models, res, te = run_experiment(df)
    full = models["Fused EHR + wearables (GlucoTwin)"]
    res["dataset"] = {"patients": n_pat, "days": 14, "rows": int(len(df)), "eligible_rows": int(len(el)),
                      "prevalence_eligible": float(el.y.mean()), "time_above_180_pct": float((series.glucose > 180).mean() * 100)}
    gi = group_importance(full, te)
    res["group_importance"] = gi.set_index("group")["auroc_drop"].to_dict()
    if not quick:
        print("[4/6] Repeating over 3 random patient splits for robustness (slow)")
        res["repeated_splits"] = repeat_splits(df)
    joblib.dump(full, art / "glucoforecaster.joblib")
    (art / "metrics.json").write_text(json.dumps(res, indent=2))
    te.to_csv(art / "test_predictions.csv.gz", index=False)

    print("[5/6] Figures")
    # ROC / PR
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    y = te["y"].values
    curves = {"Fused EHR + wearables (GlucoTwin)": (models["Fused EHR + wearables (GlucoTwin)"], TEAL),
              "CGM only": (models["CGM only"], AMBER), "EHR only": (models["EHR only"], GREY)}
    for name, (m, c) in curves.items():
        p = m.predict_proba(te)
        fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        axs[0].plot(fpr, tpr, color=c, lw=2, label=f"{name} ({res['ablation'][name]['auroc']:.3f})")
        axs[1].plot(rc, pr, color=c, lw=2, label=f"{name} ({res['ablation'][name]['auprc']:.3f})")
    axs[0].plot([0, 1], [0, 1], color="#D1D5DB", ls="--"); axs[0].set(xlabel="False-positive rate", ylabel="Sensitivity", title="ROC (held-out patients)")
    axs[1].axhline(y.mean(), color="#D1D5DB", ls="--"); axs[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall (held-out patients)")
    for a in axs:
        a.legend(fontsize=8, frameon=False); a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(fig_dir / "roc_pr.png", dpi=160); plt.close(fig)

    # ablation bars
    names = list(res["ablation"]) + list(res["cold_start_ablation"])
    src = {**res["ablation"], **res["cold_start_ablation"]}
    rep = res.get("repeated_splits", {})
    au = [rep[n]["auroc_mean"] if n in rep else src[n]["auroc"] for n in names]
    er = [rep[n]["auroc_std"] if n in rep else 0 for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    cols = [GREY, AMBER, "#0891B2", TEAL, "#F59E0B", DARK]
    ax.barh(names[::-1], au[::-1], xerr=er[::-1], color=cols[::-1], capsize=3)
    for k, v in enumerate(au[::-1]):
        ax.text(v + er[::-1][k] + 0.006, k, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUROC on held-out patients" + (" (mean ± sd over 3 splits)" if rep else ""))
    ax.grid(axis="x", alpha=.25); fig.tight_layout(); fig.savefig(fig_dir / "ablation.png", dpi=160); plt.close(fig)

    # calibration
    p = full.predict_proba(te)
    fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(4.6, 4.3))
    ax.plot(mp, fr, "o-", color=TEAL); ax.plot([0, 1], [0, 1], "--", color="#D1D5DB")
    ax.set(xlabel="Predicted probability", ylabel="Observed frequency", title="Calibration (held-out patients)"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(fig_dir / "calibration.png", dpi=160); plt.close(fig)

    # group importance
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    g2 = gi.sort_values("auroc_drop")
    ax.barh(g2["group"], g2["auroc_drop"], color=TEAL); ax.set_xlabel("AUROC drop when group is shuffled"); ax.grid(axis="x", alpha=.25)
    fig.tight_layout(); fig.savefig(fig_dir / "group_importance.png", dpi=160); plt.close(fig)

    print("[6/6] Clinician dashboard mock-up from a held-out patient")
    te_ids = te["patient_id"].unique()
    ehr_i = ehr.set_index("patient_id")
    best, best_score = None, -1
    for pid in te_ids:
        g = series.loc[series.patient_id == pid, "glucose"]
        score = -abs((g > 180).mean() - 0.12)
        if score > best_score:
            best, best_score = pid, score
    idx = int(ehr.index[ehr.patient_id == best][0])
    row = ehr.iloc[idx]
    ts = series[series.patient_id == best]
    twin = VirtualPatient(row, ts, full, sim_seed=42 * 1000 + idx)
    # choose a moment ~45 min before a real lunch excursion, on day 10
    d = 10
    lo, hi = d * 288 + 11 * 12, d * 288 + 15 * 12
    fr_ = twin.forecast_range(lo, hi, 1)
    now = int(fr_.loc[fr_["p"].idxmax(), "index"])
    wi = twin.what_if({"post_meal_walk": True})
    clinician_dashboard(twin, now, wi, "walk after meals", str(fig_dir / "clinician_dashboard.png"))
    fig, ax = plt.subplots(figsize=(7.5, 3.6)); plot_whatif(ax, wi, day=d)
    ax.set_title("What-if: 15-min walk after meals (same meals, same day)", loc="left", fontsize=10); fig.tight_layout()
    fig.savefig(fig_dir / "whatif.png", dpi=160); plt.close(fig)
    json.dump({"demo_patient": best, "demo_index": now, "whatif_baseline": wi["baseline"], "whatif_intervention": wi["intervention"]},
              open(art / "demo_case.json", "w"), indent=2)

    print("\nDone. Key results (held-out patients):")
    for k, v in {**res["ablation"], **res["cold_start_ablation"]}.items():
        print(f"  {k:42s} AUROC {v['auroc']:.3f}  AUPRC {v['auprc']:.3f}  sens {v['sensitivity']:.2f}  spec {v['specificity']:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
