# GlucoTwin: a Digital Twin for Type 2 Diabetes glucose-excursion forecasting

> Proof-of-concept for the **Happiest Health "Reimagining and Reforming Healthcare in India" Summit 2026** Digital Twin challenge.
> All data is **synthetic**. This is a research prototype, **not a medical device** and not for clinical use.

## 1. Team details and college / incubator information
| Field | Details |
|---|---|
| Team name | `<TEAM NAME>` |
| College / incubator | `<COLLEGE NAME / INCUBATOR>` |
| Team leader | `<NAME>`, `<EMAIL>`, `<PHONE>` |
| Members (1-4) | `<NAME, program, year>` |

*(Fill in before submitting. Folder / repo name format required by the organisers: `TeamName_CollegeName`.)*

## 2. Project title
**GlucoTwin: fusing EHR and wearable/CGM streams in a virtual patient that warns of glucose excursions 2 hours ahead.**

## 3. Problem statement and healthcare use case
India has an estimated 100+ million people living with diabetes, and most of the harm comes from poorly controlled glucose between clinic visits. A doctor sees a patient for minutes every few months, while the patient's glucose, sleep, activity and stress change every few minutes.

**Condition chosen:** Type 2 Diabetes (T2D).
**Adverse event predicted:** a hyperglycaemic excursion, defined as *glucose rising above 180 mg/dL at any point in the next 120 minutes* (the ADA "time above range" level), for a patient who is currently not already above 160 mg/dL.
**Who uses it:** a diabetologist / care team gets a virtual replica of each patient, an early warning, an explanation of *why*, and a *what-if* tool ("would a 15-minute walk after lunch help this patient?") that pure prediction models cannot offer.

## 4. What the prototype demonstrates (challenge checklist)
| Requirement | Where |
|---|---|
| Static / historical stream: simulated EHR (demographics, diagnoses, labs, genetic markers) | `glucotwin/synthetic.py::make_ehr` (Synthea-style; ICD-10 diagnoses, HbA1c, eGFR, lipids, TCF7L2 risk alleles, medications) |
| Dynamic stream: simulated wearable / CGM time series (HRV, CGM, sleep stages, steps) | `glucotwin/synthetic.py::simulate_patient` (5-minute CGM, HR, HRV-RMSSD, steps, sleep stages) |
| Fusion of both streams in one model | `glucotwin/features.py` (44-feature fused vector) and `glucotwin/model.py` |
| Working algorithm that predicts an adverse event ahead of time | `GlucoForecaster`: P(glucose > 180 mg/dL within 2 h) + expected 2 h peak |
| Conceptual doctor-facing UI | `app/dashboard.py` (interactive Streamlit) and `docs/figures/clinician_dashboard.png` |
| Only synthetic / anonymised data | Fully synthetic; no real patient data is used anywhere |

## 5. Technical stack, AI/ML model and framework
* **Language / libs:** Python 3.10+, NumPy, pandas, scikit-learn, matplotlib, joblib, Streamlit.
* **Model:** `HistGradientBoostingClassifier` (forecast probability) + isotonic calibration on validation patients + `HistGradientBoostingRegressor` (expected 2 h peak). The alert threshold is chosen on validation patients for roughly 80% sensitivity.
* **Digital-twin design (hybrid):** `VirtualPatient` = static EHR + rolling wearable history. It offers `forecast(t)` (data-driven) and `what_if(change)` (mechanistic simulator re-run with paired meals and noise, so differences come only from the change).
* **Features (44, leak-free):** glucose level / slope / variability, 24 h time-in-range, "what happened after this same clock time on previous days", steps, heart-rate excess, HRV drop (stress), 24 h sleep duration and stage fractions, time of day, plus 16 EHR fields.
* **Explainability:** per-forecast driver bars (each feature group replaced by a typical-patient value) and global permutation importance by group.
* **Evaluation protocol:** patient-level train / validation / test split (no patient in two splits), 3 repeated splits, ablations, baselines, calibration, alert burden.

![architecture](docs/figures/architecture.png)

## 6. Results (held-out synthetic patients, 150-patient cohort, 14 days each, 5-minute resolution)
Mean ± sd over 3 random patient-level splits.

| Model | AUROC | AUPRC |
|---|---|---|
| EHR only | 0.674 ± 0.014 | 0.279 ± 0.015 |
| CGM only | 0.942 ± 0.004 | 0.768 ± 0.022 |
| Wearables only (no EHR) | 0.942 ± 0.004 | 0.772 ± 0.021 |
| **Fused EHR + wearables (GlucoTwin)** | **0.942 ± 0.004** | **0.775 ± 0.024** |
| Cold start (no personal history yet): wearables only | 0.935 ± 0.006 | 0.739 ± 0.032 |
| Cold start: wearables + EHR (fused) | 0.940 ± 0.005 | 0.762 ± 0.025 |
| Baseline: logistic regression on current glucose + trend (primary split) | 0.608 | 0.260 |

Event prevalence among eligible forecast times is about 17%. On the primary split the fused model has about 89% sensitivity / 87% specificity / 62% precision at its validation-chosen threshold, a 2 h peak MAE of about 11.5 mg/dL, and it pre-warned 510 of 521 (98%) excursion episodes, at the cost of about 6 alerts per patient-day (with a 60-minute cool-down).

**Honest reading of these numbers**
* Recent glucose history and time-of-day (meal habits) carry most of the signal. EHR-only is weak (AUROC about 0.67): static records alone cannot time an excursion.
* **Fusion gives a small gain once a patient has CGM history** (AUPRC 0.768 to 0.775, within one standard deviation). Its clearest value is in the **cold-start** setting, where adding the EHR lifts AUPRC from 0.739 to 0.762 (about one sd, 3 splits only).
* In this synthetic world, HRV, sleep and steps contribute little to the *forecast* (global importance ≈ 0.001 to 0.006 AUROC) even though they drive the simulated physiology. The wearable signals mainly add explanatory context for the clinician. We report this as measured rather than claiming otherwise.
* About 6 alerts per patient-day is too many for a real product. A higher threshold trades sensitivity for fewer alerts. Tuning the operating point with clinicians is a key next step.
* Because the labels come from our own simulator, **these metrics measure how well the pipeline learns a known process, not real-world clinical accuracy.** They are not evidence of clinical performance.

What-if example (patient `P0030`, same meals and noise): a 15-minute walk after each main meal reduces time above 180 mg/dL from 12.9% to 6.1% and mean daily peak from 231 to 205 mg/dL. This comes from the mechanistic simulator's assumptions, not from clinical evidence.

![dashboard](docs/figures/clinician_dashboard.png)

## 7. How to run
```bash
git clone <this repo> && cd TeamName_CollegeName
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_pipeline.py --quick     # ~1 min: small cohort, single split
python scripts/run_pipeline.py             # full run (~5-8 min): data, 3-split evaluation, figures, model
streamlit run app/dashboard.py             # interactive clinician console
pytest -q                                  # 6 tests incl. a no-future-leakage test
python scripts/make_architecture.py        # regenerates docs/architecture.pdf
```
Outputs: `data/synthetic/` (EHR, 5-min wearables, meal ground truth), `artifacts/` (model, `metrics.json`, test predictions), `docs/figures/`.

## 8. Data sources and privacy
Everything is generated by our own code (`glucotwin/synthetic.py`), so no personal data is processed and the DPDP Act / HIPAA constraints are respected by design. EHR fields follow the structure of Synthea-style records (ICD-10 coded diagnoses, labs, medications). Details in [`docs/DATA_CARD.md`](docs/DATA_CARD.md). Validation on real open datasets is listed as future work, subject to each dataset's licence and access terms.

## 9. Limitations and responsible use
See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md). In short: synthetic-only validation, a simplified physiology model, 180 mg/dL fixed threshold, alert fatigue risk, no regulatory clearance, and the what-if tool is a simulation, not a prescription.

## 10. Open-source licence
MIT, see [`LICENSE`](LICENSE). Third-party libraries keep their own licences (BSD / MIT / Apache-2.0).

## 11. Submission documents
| Item | File |
|---|---|
| Architecture diagram (PDF) | [`docs/architecture.pdf`](docs/architecture.pdf) |
| Presentation (PPT + PDF) | [`docs/GlucoTwin_Presentation.pptx`](docs/GlucoTwin_Presentation.pptx), [`docs/GlucoTwin_Presentation.pdf`](docs/GlucoTwin_Presentation.pdf) |
| Demo video (≥ 20 min) | `<UNLISTED YOUTUBE LINK>` (script: [`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md)) |

## 12. Repository layout
```
glucotwin/   synthetic.py  features.py  model.py  twin.py  plots.py
scripts/     run_pipeline.py  make_architecture.py  render_dashboard_png.py
app/         dashboard.py                      (Streamlit clinician console)
tests/       test_pipeline.py
data/        synthetic/                        (generated)
artifacts/   model, metrics, predictions       (generated)
docs/        architecture.pdf, presentation, figures/, DATA_CARD.md, MODEL_CARD.md, VIDEO_SCRIPT.md
```
