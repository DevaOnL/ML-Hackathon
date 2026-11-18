import json

old = json.load(open('experiments/final_five/cv/gaussian_nb.json'))
new = json.load(open('experiments/final_five/current/cv/gaussian_nb.json'))
print("OLD (first attempt) keys:", sorted(old.keys()))
print("OLD objective:", old.get('objective'))
print("OLD oof top1:", old.get('oof_metrics', {}).get('top1_accuracy'),
      "trip_macro:", old.get('oof_metrics', {}).get('trip_macro_top1_accuracy'),
      "macro_f1:", old.get('oof_metrics', {}).get('macro_f1'),
      "log_loss:", old.get('oof_metrics', {}).get('log_loss'))
print("NEW objective:", new.get('objective'))
print("NEW oof top1:", new.get('oof_metrics', {}).get('top1_accuracy'),
      "trip_macro:", new.get('oof_metrics', {}).get('trip_macro_top1_accuracy'),
      "macro_f1:", new.get('oof_metrics', {}).get('macro_f1'),
      "log_loss:", new.get('oof_metrics', {}).get('log_loss'))
print("OLD completed/elapsed keys present:", [k for k in old if 'time' in k or 'second' in k])
print("OLD has reuse_provenance:", 'reuse_provenance' in old, "| phase:", old.get('phase'))
print("OLD warnings in folds:", [f.get('warnings') for f in old.get('folds', [])][:1])
# HGB old vs new
for n in ('hist_gradient_boosting', 'hist_gradient_boosting_tuned'):
    o = json.load(open(f'experiments/final_five/cv/{n}.json'))
    nn = json.load(open(f'experiments/final_five/current/cv/{n}.json'))
    print(n, "old obj:", o.get('objective'), "new obj:", nn.get('objective'),
          "identical oof_metrics:", o.get('oof_metrics') == nn.get('oof_metrics'))
# reused families old vs new
for n in ('logistic_regression', 'knn_distance_weighted', 'bagging'):
    o = json.load(open(f'experiments/final_five/cv/{n}.json'))
    nn = json.load(open(f'experiments/final_five/current/cv/{n}.json'))
    print(n, "old obj:", o.get('objective'), "new obj:", nn.get('objective'),
          "old has reuse_provenance:", 'reuse_provenance' in o,
          "new reuse:", nn.get('reused'))
import hashlib
for p in ['experiments/final_five/cv/gaussian_nb.json', 'experiments/final_five/current/cv/gaussian_nb.json']:
    print(p, hashlib.sha256(open(p,'rb').read()).hexdigest()[:16])
