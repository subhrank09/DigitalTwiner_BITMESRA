# Model card: GlucoForecaster

* **Task:** binary forecast, P(glucose > 180 mg/dL at any time in the next 120 min), for forecast times with current glucose ≤ 160 mg/dL. Secondary regressor for expected 2 h peak.
* **Model:** gradient-boosted trees (scikit-learn HistGradientBoosting), isotonic calibration, threshold chosen on validation patients for about 80% sensitivity.
* **Inputs:** 44 features (16 EHR, 16 CGM history, 4 activity, 2 HRV, 4 sleep, 2 time-of-day), computed from data available at forecast time only (unit test checks no leakage).
* **Evaluation:** patient-level splits (70/15/15), 3 repeated splits, ablations, cold-start setting, calibration curve, alert burden.
* **Intended use:** research and education; demonstration of EHR + wearable fusion in a digital twin.
* **Not intended for:** diagnosis, treatment decisions, insulin dosing or any clinical use.
* **Known limitations:** trained and tested only on synthetic data (performance is not clinical evidence); about 6 alerts per patient-day at the chosen threshold; HRV, sleep and steps add little predictive value in this simulation; fixed 180 mg/dL threshold; no subgroup fairness analysis possible with synthetic demographics; what-if results reflect simulator assumptions.
* **Risks if misused:** alert fatigue, false reassurance, over-reliance on simulated counterfactuals.
* **Next steps:** real-data validation under ethics approval, personalised insulin-sensitivity estimation, uncertainty intervals, clinician-tuned alert policy, regulatory pathway assessment.
