"""Convert real CGMacros data into GlucoTwin's `series` and `ehr` formats.

Place at glucotwin/real_adapter.py.

series columns match data/synthetic/wearables_5min.csv.gz:
    patient_id, ts, glucose, hr, hrv_rmssd, steps, sleep_stage
Real data has no HRV / steps / sleep, so those are set to neutral placeholders
(NaN / 0 / -1 = awake). They must NEVER be used as model features on real data;
the validation script only uses CGM + time-of-day (+ optional EHR subset).

Each participant is split at gaps > 15 min into continuous segments
(ids like R001_s0) because the feature code works on a regular 5-min grid.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from glucotwin import real_cgmacros as rc

BINS_PER_DAY = 288

# Columns of the synthetic ehr.csv (build_dataset reads these)
SYN_EHR_COLS = [
    "age", "sex", "bmi", "hba1c", "fasting_glucose", "sbp", "dbp", "ldl",
    "triglycerides", "egfr", "years_since_dx", "family_history",
    "tcf7l2_risk_alleles", "diet", "activity_level", "on_metformin",
    "on_sulfonylurea", "prior_cvd", "hypertension", "diagnoses_icd10",
]


def _find(bio, prefix):
    for c in bio.columns:
        if c.strip().lower().startswith(prefix):
            return c
    return None


def make_ehr_real(root="data/real/cgmacros"):
    """Participant-level EHR in the synthetic schema. Fields CGMacros lacks stay NaN."""
    bio = rc.load_ehr(root)
    out = pd.DataFrame({"patient_id": bio["patient_id"].values})
    for col in SYN_EHR_COLS:
        out[col] = np.nan

    def put(dst, prefix, transform=None):
        c = _find(bio, prefix)
        if c is None:
            print(f"WARNING: bio.csv has no column starting with '{prefix}' -> {dst} left empty")
            return
        v = bio[c].values
        out[dst] = transform(v) if transform else v

    put("age", "age")
    put("sex", "gender", lambda v: pd.Series(v).astype(str).str.strip().str.upper().str[0].values)
    put("bmi", "bmi")
    put("hba1c", "a1c")
    put("fasting_glucose", "fasting glu")
    put("ldl", "ldl", lambda v: pd.Series(v, dtype=float).where(lambda s: s < 800).values)  # 800 = error code
    put("triglycerides", "triglycerides")

    # HbA1c unit guard: dictionary says mmol/mol but the range (4.6-8.5) looks like percent
    if out["hba1c"].median() > 20:
        print("HbA1c looks like mmol/mol -> converting to %")
        out["hba1c"] = out["hba1c"] / 10.929 + 2.15
    else:
        print(f"HbA1c treated as percent (median {out['hba1c'].median():.1f})")
    return out


def _group(a1c):
    if pd.isna(a1c):
        return "unknown"
    return "T2D" if a1c > 6.4 else ("prediabetes" if a1c >= 5.7 else "healthy")


def make_series(root="data/real/cgmacros", cgm_source="Dexcom GL", min_days=3.0):
    """Return (series, ehr, meta, report). ehr/meta are keyed by segment id."""
    wear = rc.load_all(root, cgm_source)
    ehr_base = make_ehr_real(root).drop_duplicates("patient_id").set_index("patient_id")
    min_bins = int(min_days * BINS_PER_DAY)

    series_parts, ehr_rows, meta_rows, report = [], [], [], []
    for pid, g in wear.groupby("patient_id"):
        g = g.set_index("timestamp").sort_index().asfreq("5min")
        ok = g["cgm"].notna()
        run = (ok != ok.shift()).cumsum()
        k, kept_bins = 0, 0
        for _, seg in g[ok].groupby(run[ok]):
            seg = seg.copy()
            # a segment is a run of valid glucose rows; require a regular grid and enough length
            if len(seg) < min_bins:
                continue
            sid = f"{pid}_s{k}"
            k += 1
            kept_bins += len(seg)
            hr = seg["hr"].ffill().bfill() if "hr" in seg else pd.Series(np.nan, index=seg.index)
            hr = hr.fillna(hr.median() if hr.notna().any() else 70.0)
            series_parts.append(pd.DataFrame({
                "patient_id": sid,
                "ts": seg.index,
                "glucose": seg["cgm"].round(1).values,
                "hr": hr.round(1).values,
                "hrv_rmssd": np.nan,
                "steps": 0.0,
                "sleep_stage": -1,
            }))
            row = ehr_base.loc[pid].copy() if pid in ehr_base.index else pd.Series(np.nan, index=SYN_EHR_COLS)
            row["patient_id"] = sid
            ehr_rows.append(row)
            a1c = row.get("hba1c", np.nan)
            meta_rows.append({"patient_id": sid, "base_id": pid, "hba1c": a1c, "group": _group(a1c)})
        report.append({"participant": pid, "segments": k, "days_kept": round(kept_bins / BINS_PER_DAY, 1)})

    if not series_parts:
        raise RuntimeError("No segment met the minimum length; lower min_days or check the data")
    series = pd.concat(series_parts, ignore_index=True)
    ehr = pd.DataFrame(ehr_rows)[["patient_id"] + SYN_EHR_COLS].reset_index(drop=True)
    meta = pd.DataFrame(meta_rows)
    return series, ehr, meta, pd.DataFrame(report)


def save(out_dir="data/real", **kw):
    series, ehr, meta, report = make_series(**kw)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    series.to_csv(out / "wearables_5min_real.csv.gz", index=False)
    ehr.to_csv(out / "ehr_real.csv", index=False)
    meta.to_csv(out / "meta_real.csv", index=False)
    return series, ehr, meta, report


if __name__ == "__main__":
    s, e, m, r = save()
    print(r.to_string(index=False))
    print("\nsegments:", s["patient_id"].nunique(), "| participants:", m["base_id"].nunique())
    print(m.drop_duplicates("base_id")["group"].value_counts().to_string())