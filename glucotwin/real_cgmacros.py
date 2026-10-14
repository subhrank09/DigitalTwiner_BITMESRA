"""Load the real CGMacros dataset (PhysioNet) into regular 5-minute frames.

Place at glucotwin/real_cgmacros.py. Expected layout (adjust if yours differs):
    data/real/cgmacros/bio.csv
    data/real/cgmacros/CGMacros-001/CGMacros-001.csv ...

Only columns that exist in CGMacros are used. HRV, sleep and raw steps are NOT
in this dataset, so they are never invented here.
Gaps up to 15 min are interpolated; longer gaps stay NaN (the adapter then
splits the record into continuous segments).
"""
from pathlib import Path
import numpy as np
import pandas as pd

STEP = "5min"
MAX_GAP_STEPS = 3  # 15 minutes


def _col(df, name):
    lookup = {c.strip().lower(): c for c in df.columns}
    return lookup.get(name.lower())


def _fill_short_gaps(s, max_steps=MAX_GAP_STEPS):
    """Interpolate only NaN runs of length <= max_steps; longer runs stay NaN."""
    isna = s.isna()
    run_id = (isna != isna.shift()).cumsum()
    run_len = isna.groupby(run_id).transform("sum")
    filled = s.interpolate(limit_area="inside")
    return s.where(~(isna & (run_len <= max_steps)), filled)


def load_participant(csv_path, pid, cgm_source="Dexcom GL"):
    df = pd.read_csv(csv_path)
    ts_c, cgm_c = _col(df, "Timestamp"), _col(df, cgm_source)
    if ts_c is None or cgm_c is None:
        raise ValueError(f"{csv_path}: missing Timestamp or {cgm_source}")
    df[ts_c] = pd.to_datetime(df[ts_c], errors="coerce")
    df = df.dropna(subset=[ts_c]).set_index(ts_c).sort_index()
    df = df[~df.index.duplicated(keep="first")]

    agg = {cgm_c: "mean"}
    hr_c, mets_c = _col(df, "HR"), _col(df, "METs")
    if hr_c:
        agg[hr_c] = "mean"
    if mets_c:
        agg[mets_c] = "mean"
    out = df[list(agg)].resample(STEP).agg(agg)
    out = out.rename(columns={cgm_c: "cgm", hr_c: "hr", mets_c: "mets"})
    for c in out.columns:
        out[c] = _fill_short_gaps(out[c])
    out.insert(0, "patient_id", pid)
    return out.reset_index().rename(columns={ts_c: "timestamp"})


def load_all(root="data/real/cgmacros", cgm_source="Dexcom GL"):
    root = Path(root)
    frames = []
    for csv in sorted(root.glob("CGMacros-*/CGMacros-*.csv")):
        pid = csv.parent.name.replace("CGMacros-", "R")
        try:
            frames.append(load_participant(csv, pid, cgm_source))
        except ValueError as e:
            print("skipped:", e)
    if not frames:
        raise FileNotFoundError(f"No participant CSVs under {root}")
    return pd.concat(frames, ignore_index=True)


def load_ehr(root="data/real/cgmacros"):
    bio = pd.read_csv(Path(root) / "bio.csv")
    bio.columns = [c.strip() for c in bio.columns]
    id_c = next((c for c in bio.columns if c.lower() in ("subject", "participant", "id")), bio.columns[0])
    bio["patient_id"] = bio[id_c].astype(str).str.extract(r"(\d+)")[0].str.zfill(3).radd("R")
    return bio


def summarise(wear):
    g = wear.groupby("patient_id")["cgm"]
    t = wear.dropna(subset=["cgm"]).groupby("patient_id")["timestamp"]
    return pd.DataFrame({
        "valid_5min": g.count(),
        "days_span": ((t.max() - t.min()).dt.total_seconds() / 86400).round(1),
        "mean_glucose": g.mean().round(1),
        "pct_above_180": (g.apply(lambda s: (s.dropna() > 180).mean()) * 100).round(1),
    })


if __name__ == "__main__":
    w = load_all()
    print(summarise(w).to_string())