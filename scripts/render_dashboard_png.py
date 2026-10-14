"""Re-render docs/figures/clinician_dashboard.png from saved artifacts (no retraining)."""
import json, sys
from pathlib import Path
import joblib, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from glucotwin.plots import clinician_dashboard
from glucotwin.twin import VirtualPatient
ehr = pd.read_csv(ROOT/"data/synthetic/ehr_with_latent_truth.csv")
series = pd.read_csv(ROOT/"data/synthetic/wearables_5min.csv.gz", parse_dates=["ts"])
model = joblib.load(ROOT/"artifacts/glucoforecaster.joblib")
case = json.load(open(ROOT/"artifacts/demo_case.json"))
idx = int(ehr.index[ehr.patient_id == case["demo_patient"]][0])
twin = VirtualPatient(ehr.iloc[idx], series[series.patient_id == case["demo_patient"]], model, sim_seed=42*1000+idx)
wi = twin.what_if({"post_meal_walk": True})
clinician_dashboard(twin, case["demo_index"], wi, "walk after meals", str(ROOT/"docs/figures/clinician_dashboard.png"))
print("ok")
