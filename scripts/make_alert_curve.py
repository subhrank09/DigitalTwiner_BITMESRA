"""Alert-burden trade-off: episode recall vs alerts per patient-day.

Place at scripts/make_alert_curve.py. Run from the repo root:
    python scripts/make_alert_curve.py

Uses glucotwin.model.alert_burden, so alerts are counted exactly as in metrics.json
(60-minute cool-down; an episode is "warned" if any row in the run had p >= threshold).

Outputs:
    docs/figures/alert_tradeoff.png
    artifacts/alert_sweep_synthetic.csv, artifacts/alert_sweep_real.csv
    a printed table of the best threshold under 1, 2, 3 and 5 alerts per patient-day
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from glucotwin.model import alert_burden

BUDGETS = [1, 2, 3, 5]


def sweep(df, n=45):
    p = df["p"].values.astype(float)
    y = df["y"].values.astype(int)
    grid = np.r_[np.quantile(p, np.linspace(0.40, 0.9995, n)), np.linspace(0.02, 0.98, 25)]
    rows = []
    for t in np.unique(grid):
        r = alert_burden(df, float(t))
        flag = p >= t
        tp = int((flag & (y == 1)).sum())
        rows.append({
            "threshold": float(t),
            "alerts_per_day": float(r["alerts_per_patient_day"]),
            "episode_recall": float(r["episode_recall"]),
            "episodes": int(r["episodes"]),
            "row_sensitivity": tp / max(int(y.sum()), 1),
            "row_precision": tp / max(int(flag.sum()), 1),
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def pick(sw, budget):
    ok = sw[sw["alerts_per_day"] <= budget]
    if ok.empty:
        return None
    return ok.sort_values(["episode_recall", "threshold"], ascending=[False, True]).iloc[0]


def load_synthetic(path, expected_episodes):
    te = pd.read_csv(path)
    te["ts"] = pd.to_datetime(te["ts"])
    cands = []
    if "eligible" in te.columns:
        cands.append(te[te["eligible"].astype(bool)].reset_index(drop=True))
    cands.append(te.reset_index(drop=True))
    for c in cands:
        got = alert_burden(c, 0.5)["episodes"]
        if expected_episodes is None or got == expected_episodes:
            return c
    print(f"WARNING: synthetic episode count ({got}) differs from metrics.json ({expected_episodes}); "
          "the curve may not match your earlier 5.8 alerts/day exactly.")
    return cands[0]


def load_real(path, col):
    d = pd.read_csv(path)
    if "ts" not in d.columns:
        raise SystemExit(f"{path} has no 'ts' column. Re-run run_real_validation.py (new version) first.")
    if col not in d.columns:
        raise SystemExit(f"{path} has no column {col}. Columns: {list(d.columns)}")
    out = pd.DataFrame({"patient_id": d["patient_id"], "ts": pd.to_datetime(d["ts"]),
                        "y": d["y"].astype(int), "p": d[col].astype(float), "eligible": True})
    return out


def show(name, sw):
    print(f"\n{name}: best episode recall within an alert budget")
    print(f"  {'budget/day':<11}{'threshold':<11}{'alerts/day':<12}{'ep. recall':<12}{'row sens.':<11}{'row prec.':<10}")
    for b in BUDGETS:
        r = pick(sw, b)
        if r is None:
            print(f"  {b:<11}no threshold reaches this budget")
            continue
        print(f"  {b:<11}{r.threshold:<11.3f}{r.alerts_per_day:<12.2f}{r.episode_recall:<12.3f}"
              f"{r.row_sensitivity:<11.3f}{r.row_precision:<10.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="artifacts/test_predictions.csv.gz")
    ap.add_argument("--metrics", default="artifacts/metrics.json")
    ap.add_argument("--real", default="artifacts/real_oof_dexcom.csv")
    ap.add_argument("--real-col", default="p_CGM_only")
    ap.add_argument("--fig", default="docs/figures/alert_tradeoff.png")
    a = ap.parse_args()

    metrics = json.load(open(a.metrics)) if Path(a.metrics).exists() else {}
    cur = metrics.get("alerts", {})
    syn = load_synthetic(a.synthetic, cur.get("episodes"))
    real = load_real(a.real, a.real_col)

    sw_syn, sw_real = sweep(syn), sweep(real)
    Path("artifacts").mkdir(exist_ok=True)
    sw_syn.to_csv("artifacts/alert_sweep_synthetic.csv", index=False)
    sw_real.to_csv("artifacts/alert_sweep_real.csv", index=False)

    show("SYNTHETIC test patients (calibrated model)", sw_syn)
    show("REAL CGMacros participants (cross-validated, CGM-only model)", sw_real)
    if cur:
        print(f"\nCurrent operating point in metrics.json: {cur['alerts_per_patient_day']:.2f} alerts/day, "
              f"episode recall {cur['episode_recall']:.3f}")
    print("\nNote: operating points are picked on the same predictions they are reported on, "
          "so treat them as illustrative until chosen on separate data.")

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for sw, label, c in [(sw_syn, "Synthetic test patients", "#1f77b4"),
                         (sw_real, "Real participants (CGMacros)", "#d62728")]:
        s = sw[sw["alerts_per_day"] <= 10].sort_values("alerts_per_day")
        ax.plot(s["alerts_per_day"], s["episode_recall"], lw=2, label=label, color=c)
    ax.axvspan(1, 2, color="gray", alpha=0.15, label="Target: 1-2 alerts/day")
    if cur:
        ax.scatter([cur["alerts_per_patient_day"]], [cur["episode_recall"]], color="black", zorder=5,
                   label="Current setting (synthetic)")
    ax.set_xlabel("Alerts per patient-day (60-min cool-down)")
    ax.set_ylabel("Fraction of excursion episodes warned in advance")
    ax.set_title("Alert burden vs. sensitivity")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(a.fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.fig, dpi=160)
    print(f"\nSaved {a.fig}, artifacts/alert_sweep_synthetic.csv, artifacts/alert_sweep_real.csv")


if __name__ == "__main__":
    main()