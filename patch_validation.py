"""One-off patch: adds --oof-out and saves timestamps in scripts/run_real_validation.py.
Run from the repo root:  python3 patch_validation.py
Safe to run twice (it checks first).
"""
p = "scripts/run_real_validation.py"
s = open(p).read()

if "--oof-out" in s:
    print("Already patched, nothing to do.")
    raise SystemExit

edits = [
    ('    ap.add_argument("--out", default="artifacts/real_validation.json")\n',
     '    ap.add_argument("--out", default="artifacts/real_validation.json")\n'
     '    ap.add_argument("--oof-out", default="artifacts/real_oof_predictions.csv")\n'),
    ('    pred = real[["patient_id", "base_id", "group"]].copy()\n',
     '    keep = [c for c in ["patient_id", "base_id", "group", "ts"] if c in real.columns]\n'
     '    pred = real[keep].copy()\n'),
    ('pred.to_csv("artifacts/real_oof_predictions.csv", index=False)',
     'pred.to_csv(a.oof_out, index=False)'),
    ('Saved {a.out} and artifacts/real_oof_predictions.csv',
     'Saved {a.out} and {a.oof_out}'),
]
for old, new in edits:
    if old not in s:
        raise SystemExit(f"Could not find this text in {p}, so nothing was changed:\n{old}")
    s = s.replace(old, new)

open(p, "w").write(s)
print("Patched OK. You can now use --oof-out.")
