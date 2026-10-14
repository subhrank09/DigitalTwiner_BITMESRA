# Data card (all data synthetic)

**Cohort:** 150 virtual adults with T2D / pre-diabetes profiles, 14 days of data each (5-minute resolution, 4,032 points per patient).

**Static EHR (`data/synthetic/ehr.csv`)**: age, sex, BMI, HbA1c, fasting glucose, SBP/DBP, LDL, triglycerides, eGFR, years since diagnosis, family history, TCF7L2 rs7903146 risk-allele count, diet (veg / non-veg), activity level, metformin, sulfonylurea, prior CVD, hypertension, ICD-10 diagnoses. Fasting glucose is anchored to HbA1c via the ADA eAG formula.

**Dynamic streams (`wearables_5min.csv.gz`)**: CGM glucose (mg/dL), heart rate, HRV (RMSSD, ms), steps per 5 min, sleep stage (-1 not in bed, 0 awake, 1 light, 2 deep, 3 REM).

**Hidden ground truth** (`ehr_with_latent_truth.csv`, `meals_groundtruth.csv`): the latent insulin-resistance value, walking habit, resting HR and the meal log. These are NOT used as model features. The latent file is used only by the what-if simulator.

**Generation assumptions:** Indian-style meals (breakfast ~08:00, lunch ~13:15, optional snack, dinner ~20:30; 30 to 110 g carbohydrate), post-meal excursion proportional to carbohydrate and insulin resistance, sleep under 7 h raising next-day resistance, stress events raising glucose and lowering HRV, steps lowering glucose and blunting post-meal peaks, metformin / sulfonylurea damping meal response.

**Known gaps:** simplified physiology; no insulin dosing, illness, alcohol, menstrual cycle or sensor drift; no real-world noise structure; population parameters are plausible but not epidemiologically fitted. Outcomes depend on our own assumptions.

**Real-data validation plan (future):** evaluate on open wearable + CGM datasets and anonymised clinical databases (for example MIMIC-IV or Synthea-generated cohorts) after checking each licence and access agreement.
