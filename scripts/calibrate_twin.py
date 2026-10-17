"""Validate twin.calibrate_ir(): recovery of latent IR + effect on what-if deltas."""
import argparse, json, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glucotwin.synthetic import generate_cohort
from glucotwin.twin import VirtualPatient


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="patients to calibrate (first n)")
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--n-whatif", type=int, default=5)
    ap.add_argument("--out", default="artifacts/twin_calibration.json")
    a = ap.parse_args()

    ehr, ts, _ = generate_cohort(150, 14, 42)       # same cohort/seeds as the pipeline
    rows, t0 = [], time.time()
    for i in range(a.n):
        r = ehr.iloc[i]
        s = ts[ts.patient_id == r["patient_id"]]
        vp = VirtualPatient(r, s, None, sim_seed=42 * 1000 + i)
        res = vp.calibrate_ir(days=a.days)
        rows.append((i, res["ir_true"], res["ir_est"]))
        if (i + 1) % 10 == 0:
            print(f"{i+1}/{a.n}  {time.time()-t0:.0f}s", flush=True)

    true = np.array([x[1] for x in rows]); est = np.array([x[2] for x in rows])
    out = {"n": a.n, "days": a.days,
           "corr": float(np.corrcoef(true, est)[0, 1]),
           "mae": float(np.abs(true - est).mean()),
           "mae_predict_mean_baseline": float(np.abs(true - true.mean()).mean()),
           "ir_true_sd": float(true.std())}

    # what-if: true IR vs estimated IR
    wi = []
    for i in range(a.n_whatif):
        r = ehr.iloc[i]
        s = ts[ts.patient_id == r["patient_id"]]
        vp = VirtualPatient(r, s, None, sim_seed=42 * 1000 + i)
        vp.ir_est = rows[i][2]
        d = {}
        for tag, flag in (("true_ir", True), ("est_ir", False)):
            w = vp.what_if({"post_meal_walk": True}, use_true_ir=flag)
            d[tag] = round(w["baseline"]["time_above_180_pct"] - w["intervention"]["time_above_180_pct"], 2)
        wi.append({"patient": r["patient_id"], "ir_true": rows[i][1], "ir_est": rows[i][2],
                   "delta_time_above_180_pct_points": d})
    out["whatif"] = wi

    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()