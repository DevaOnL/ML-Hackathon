import json

ev = json.load(open('experiments/final_five/current/evaluation/gaussian_nb.json'))
print("gaussian_nb evaluation keys:", sorted(ev.keys()))
for k, v in sorted(ev.items()):
    s = json.dumps(v)
    print(" ", k, "=", s[:260] + ("..." if len(s) > 260 else ""))

print()
tr = json.load(open('experiments/final_five/current/training/gaussian_nb.json'))
print("gaussian_nb training keys:", sorted(tr.keys()))
for k, v in sorted(tr.items()):
    s = json.dumps(v)
    print(" ", k, "=", s[:260] + ("..." if len(s) > 260 else ""))

print()
d = json.load(open('experiments/final_five/current/results.json'))
b = d['bayesian']
print("bayesian keys:", sorted(b.keys()))
for k in sorted(b.keys()):
    if k != 'trials':
        s = json.dumps(b[k])
        print(" ", k, "=", s[:400] + ("..." if len(s) > 400 else ""))
print(" n_trials:", len(b['trials']))
for t in b['trials'][:15]:
    print("  trial:", json.dumps(t)[:300])
print()
mf = d['model_freeze']
print("model_freeze:", json.dumps(mf, indent=1))
print()
print("evaluations keys:", sorted(d['evaluations'].keys()))
eg = d['evaluations']['gaussian_nb']
print("gnb evaluation summary keys:", sorted(eg.keys()))
