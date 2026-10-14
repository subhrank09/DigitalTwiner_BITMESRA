# Demo video script (target 22-24 minutes; organisers require at least 20)

Record as a screen-share with the slides (`GlucoTwin_Presentation.pptx`) plus a live terminal / browser. Upload as an **unlisted** YouTube video and paste the link into the README.
Before recording, run `python scripts/run_pipeline.py` once so everything is ready.

| Time | Segment | What to show / say |
|---|---|---|
| 0:00-1:30 | Intro | Team, college, problem in one sentence. Why a Digital Twin, why Type 2 Diabetes in India. |
| 1:30-4:00 | Problem and use case | Slides 2-3. Clinic visits are minutes apart while glucose changes every few minutes. Define the adverse event: glucose above 180 mg/dL within 2 hours. |
| 4:00-7:00 | Architecture | `docs/architecture.pdf`. Walk left to right: two streams, fusion, forecast model, mechanistic simulator, VirtualPatient, dashboard. Explain the hybrid ML + simulator idea. |
| 7:00-10:00 | Data (live) | Open `data/synthetic/ehr.csv` and `wearables_5min.csv.gz`. Explain Synthea-style EHR fields, TCF7L2 alleles, 5-minute CGM / HR / HRV / steps / sleep stages. State clearly that all data is synthetic and why (DPDP / HIPAA). Show a CGM day with meals. |
| 10:00-13:30 | Features and model (code walk) | `glucotwin/features.py` (leak-free windows, "same clock time on previous days"), `model.py` (patient-level split, calibration, threshold). Run `pytest -q` and point at the no-leakage test. |
| 13:30-17:00 | Results | `metrics.json` and the ablation figure. Present the honest story: EHR-only is weak (0.67), CGM history dominates (0.94), fusion adds a small gain, and mainly helps cold start. Show calibration and the alert burden (about 6 per patient-day) and say it is too high for production. |
| 17:00-21:00 | Live dashboard demo | `streamlit run app/dashboard.py`. Pick a patient, move to just before lunch, show risk, forecast, "Why?" drivers. Toggle the post-meal walk, change sleep and carbs, read the time-above-range change. Explain the what-if is simulator-based. |
| 21:00-23:00 | Limitations and responsible use | Synthetic-only validation, simplified physiology, alert fatigue, not a medical device. |
| 23:00-24:00 | Impact and next steps | Real-data validation with ethics approval, personalised parameter estimation, clinician-tuned alerts, integration with ABDM-style records. Thank the organisers. |
