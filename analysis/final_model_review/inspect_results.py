import json, sys

d = json.load(open('experiments/final_five/current/results.json'))
print("top keys:", sorted(d.keys()))
print("completed_at:", d.get('completed_at_utc'))
print("families:", d['families'], "count", d['family_count'])
print("selected_model:", d.get('selected_model'), "final_run_name:", d.get('final_run_name'))
for name, cv in d['cv'].items():
    print()
    print("== CV", name, "reused:", cv.get('reused'))
    print("  params:", json.dumps(cv.get('parameters')))
    print("  objective:", cv.get('objective'), "|", cv.get('objective_name'))
    print("  top1_fold_mean:", cv.get('top1_fold_mean'), "std:", cv.get('top1_fold_std'))
    print("  macro_f1_fold_mean:", cv.get('macro_f1_fold_mean'))
    folds = cv.get('folds')
    if folds is not None:
        print("  n_folds:", len(folds))
        for f in folds:
            vm = f.get('val_metrics', {})
            print("   fold", f.get('fold'),
                  "train_trips", len(f.get('train_trip_ids', [])),
                  "val_trips", len(f.get('val_trip_ids', [])),
                  "fold_objective", f.get('objective'),
                  "trip_macro_top1", vm.get('trip_macro_top1_accuracy'))
    rp = cv.get('reuse_provenance')
    if rp:
        print("  reuse:", json.dumps(rp))
    print("  oof_file:", cv.get('oof_predictions_file'))
print()
for k in d:
    if k not in ('cv', 'families'):
        v = d[k]
        if not isinstance(v, dict):
            print("TOP", k, "=", json.dumps(v)[:300])
        else:
            print("TOP", k, ": dict keys", sorted(v.keys())[:20])
