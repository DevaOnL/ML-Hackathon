import json

# Fold structure from one current cv file
cv = json.load(open('experiments/final_five/current/cv/gaussian_nb.json'))
print("gaussian_nb cv keys:", sorted(cv.keys()))
folds = cv.get('folds') or cv.get('fold_runs') or []
if isinstance(folds, list) and folds:
    f0 = folds[0]
    print("fold keys:", sorted(f0.keys()))
    for k, v in sorted(f0.items()):
        s = json.dumps(v)
        print(" ", k, "=", s[:220] + ("..." if len(s) > 220 else ""))
print()
print("objective:", cv.get('objective'), cv.get('objective_name'))
print("reused:", cv.get('reused'))
print("oof file:", cv.get('oof_predictions_file'))
om = cv.get('oof_metrics', {})
print("oof_metrics subset:", {k: om[k] for k in om if k != 'per_trip'})
