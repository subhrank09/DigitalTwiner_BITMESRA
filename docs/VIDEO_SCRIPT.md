# GlucoTwin: crisp script (Team Digital Twiner)

R = Rohan, S = Subhrank. `[ACTION]` = do on screen. About 12 to 13 min.
Rules: say "synthetic" with every 0.94 number. Never say "clinically proven". Never claim any clinical or expert input.

## 1. Title (R)
"We are Team Digital Twiner, BIT Mesra. This is GlucoTwin: a digital twin for Type 2 Diabetes that warns of a glucose spike up to two hours ahead. It is a research prototype on synthetic data, not a medical device."

## 2. Why a twin (R)
"India has about 101 million people with diabetes (ICMR-INDIAB, Lancet 2023). A doctor sees a patient for minutes every few months, but glucose changes every five minutes. A twin lets the care team act before the spike, not after."

## 3. Use case (R)
"Condition: Type 2 Diabetes. Event: glucose above 180 mg/dL in the next two hours, for a patient currently at or below 160. User: a diabetologist. Output: probability, expected peak, risk level, why, and what-if."

## 4. Architecture (R)
"Left to right: simulated health record plus wearable stream, then fusion into 44 leak-free features, then the twin core: ML forecaster, mechanistic simulator and VirtualPatient state, then the doctor dashboard and alert."

## 5. Data (R)
"150 synthetic patients, 14 days, five-minute resolution. Training uses no real patient data. Real data, CGMacros from PhysioNet, is used only for validation and is not redistributed."

## 6. Hybrid twin (R)
"Three parts. Forecast: gradient-boosted trees, calibrated probability and expected peak. What-if: a simulator that re-runs the same meals and noise with one change. Explain: which input group pushes risk up or down."

## 7. Synthetic results (S)
"Fused AUROC 0.942 on held-out synthetic patients. EHR alone 0.67. Simple baseline 0.61. 510 of 521 episodes warned, 98 percent, at about six alerts per patient-day. Labels come from our own simulator, so this is not clinical accuracy."

## 8. Honest reading (S)
"One: glucose history carries the signal. Two: with history, fusion gives no gain, AUPRC 0.771, 0.769, 0.770. At cold start it moves 0.739 to 0.762, within one standard deviation. Three: HRV, sleep and steps add very little to the forecast."

## 9. Real data (S)
"CGMacros, 45 adults, Dexcom. Trained on synthetic only, tested on real: AUROC 0.826. Trained on real: 0.827. So 0.94 drops to about 0.83. EHR fields gave no gain on real data, 0.817. One US site, 45 people, not clinical evidence. Libre gave 0.89 to 0.90 but may be interpolated, so Dexcom is primary."

## 10. Alert burden (S)
"At about two alerts a day, real participants get 50.5 percent of episodes warned. One a day, 31.5. Five a day, 79.4. Thresholds were picked on the same predictions, so this is illustrative. We propose a risk bar plus push alerts at a low budget, to be set with clinicians."

## 11. LIVE DASHBOARD (R introduces, S drives)
[ACTION] Open the running Streamlit app. Patient P0030, time Thursday 13:05.

R: "Now live."
S: "Synthetic patient P0030: 73-year-old male, HbA1c 6.7, Type 2 Diabetes and obesity."
S: "Glucose now 157. The twin says 96 percent chance of going above 180 in two hours. Risk HIGH. Expected peak 229. Alert YES."
S: "The dotted line is what really happened. It peaked near 205. Our average peak error is about 11.5, so single cases differ."
S: "Why: glucose level and trend, about 70 points. Time of day, 16. Sleep and clinical profile, about 6 each."
S: "Wearables: steps and heart rate rise near noon, HRV dips. Context for the clinician."
S: "What-if: a 15-minute walk after meals. Time above 180 drops 12.9 to 6.1 percent. This is the simulator's assumption, not a prescription."
[ACTION] Change patient or time once. Back to slides.

## 12. What-if (R)
"With true insulin resistance, time above 180 goes 12.9 to 6.1 percent, mean daily peak 231 to 205. In real life we would not know the true value, so we estimate it."

## 13. Calibration (R)
"We estimate insulin resistance from five days of CGM. On 150 synthetic patients: correlation 0.88, error 0.094 versus 0.164 for the cohort mean. The walk benefit was underestimated in all five patients checked. This is optimistic, because the same simulator generates and fits the data. Not tested on real patients."

## 14. Limitations (S)
"Synthetic training. Real check is 45 adults at one site. Sim-to-real gap 0.94 to 0.83. No fusion gain on real data. Sensor dependent. Fixed 180 threshold. Alert fatigue risk. What-if is a simulation. Not a medical device, no regulatory clearance."

## 15. Next steps (R + S)
R: "Validate on larger real datasets like ShanghaiT2DM, with ethics approval. Estimate more twin parameters on real patients."
S: "Clinician-tuned two-tier alerts. Uncertainty intervals and sensor checks."
R: "Code is open source, MIT, at github.com/subhrank09/DigitalTwiner_BITMESRA. Thank you."

## Q&A one-liners
- Why synthetic? Privacy and access. We then checked on real open data.
- Why 0.83 on real? The simulator is simpler than real life. We report the gap.
- Does fusion help? Not with history. Small cold-start gain in simulation. None on real data.
- Is 98 percent real? No, simulation, at 5.8 alerts a day. Real data: about half at 2 a day.
- Medical device? No. Research prototype.
