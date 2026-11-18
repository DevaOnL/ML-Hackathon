# Phase-Three Independent Review (three-model-first)

Reviewer: Role F successor (task-20). Date: 2026-11-18 (~12:05 +0530).
Scope: read-only audit of the phase-three learning runner, model definitions, decoder,
live-evidence and visualization scripts, plus the frozen caches and phase records.
Writes: only this directory (`analysis/review/phase_three/`). Machine-readable detail:
[defects.json](defects.json). Post-freeze re-run script: [audit_phase_three.py](audit_phase_three.py).

## State at audit time

CV stage completed and selection recorded while this review ran: **bagging selected**
(OOF trip-macro top1: bagging 0.891464 > LR 0.884000 > KNN 0.876786; reviewer's
independent argmax agrees). HPO had just started (bayesian_trials.jsonl created).
No freeze yet. Source hashes at audit: `run_phase_three.py` dac7f23a…,
`phase_models.py` f9500e31…, `phase_three_decoder.py` f52f27fe… (recheck before the
final audit — the writer may edit again). A dry run of the audit script at this state
passed all 13 available checks exactly (three OOF metric recomputations matched recorded
values with max delta 0.00e+00) and correctly reported the five pending stages MISSING.

## Verdict

The learning pipeline is methodologically sound. No correctness defect was found in
estimator construction, weighting, fold hygiene, HPO, or the probability/metric domain.
One previously critical bug is fixed and verified; four open defects are contract/process
issues in the runner, none of which corrupt numbers already produced. All are cheap fixes
owned by the learning agent. **No blocker to continuing cv → select → hpo → freeze**, but
**PH3-D002 and PH3-D004 should be fixed before the freeze is declared final** (D002 taints
the clean-environment evidence the Lead explicitly requested; D004 makes the immutability
claim unenforceable), and D006 needs a Lead decision on report wording before the PDF.

## Defects (summary; full detail in defects.json)

| ID | Severity | Status | Summary |
|----|----------|--------|---------|
| PH3-D001 | critical→0 | **fixed-verified** | Prefit assertion compared empty candidate sets (0) to target misses (578/19). Fixed in dac7f23a; both caches verified: 578/19 zero-positive obs, 0 empty sets. Running CV uses the fixed code. |
| PH3-D002 | medium-high | open | `clean_load_report.json` records the PARENT interpreter's `versions()` under the `.venv-clean` label; child versions never captured; cross-env probability equality not checked (finite/range only). Fix before freeze. |
| PH3-D003 | medium | open | stage_hpo/stage_cv lack existing-result guards; HPO rerun truncates `bayesian_trials.jsonl` and repeats all CV fits. PROTOCOL claims idempotence. |
| PH3-D004 | high | open | stage_freeze unconditionally overwrites `artifacts/phase_three/model.joblib` + freeze records while claiming `immutable_after_freeze: true`. A rerun silently invalidates decoder/live evidence pinned to the sha. Fix before freeze. |
| PH3-D005 | low | open | GP-EI proposal replaced by a random draw on decode collision keeps the model-based label. |
| PH3-D006 | info | clarification | Implemented HPO budget is 8 trials TOTAL (1 prior + 2 seeded + 5 GP-EI), matching the frozen PROTOCOL; the briefing phrase "true GP-EI 8" could mean 8 GP-EI proposals. Lead must pick the wording (or re-freeze PROTOCOL at 11 total). |
| PH3-D007 | low | open | LR selected + winner==defaults produces two student rows from two different fits of identical parameters, unasserted. (Not triggered by the current run: bagging was selected.) |
| PH3-D008 | info | open | LR reuse validation lacks array-identity checks vs the cache (reviewer verified equality externally — passes today). |
| PH3-D009 | info | open | Bank remainder comment says "largest class"; code uses classes[0]. |
| PH3-D010 | low | open | results.json claims stages ran as separate processes even under `--stage all`. |

## Independently verified clean (highlights; full list in defects.json `verified_clean`)

- **Caches**: sha256, fingerprints, shapes (28000/269463/80, 19160/175778/10), misses
  578/19, 0 empty candidate sets, 38 float32 features, y ∈ {0,1}.
- **Selection contract**: exactly the three requested families; TRAIN grouped CV only;
  student data unreachable before selection+freeze gates (verified in code path).
- **Standard models**: sklearn LR(lbfgs)/KNN(distance, fold-local StandardScaler pipeline)/
  Bagging-over-DecisionTree; no custom model, no metric-aware fitting.
- **Trip weighting**: label-free inverse-count-per-obs × inverse-obs-per-trip, mean-1;
  KNN approximates it via bank inclusion probabilities.
- **KNN bank**: 60k budget, fold-local train rows only, class-stratified,
  weight-proportional, deterministic seeds (folds 0–2, final 3). Synthetic audit:
  determinism, fold locality, prior preservation, uniqueness, weight-responsiveness all
  pass; no validation leakage.
- **GP-EI**: real GP surrogate (Matérn 5/2 + WhiteKernel) on measured objectives only,
  analytic EI, xi=0.0003, pool 2048+1024, dedup, seeded.
- **Probability domain**: clip [1e-7, 1-1e-7] (saved OOF spans exactly this), per-obs
  normalization, categorical NLL floor 1e-7 with no reclip, misses kept in denominators.
- **Export**: atomic write, standard sklearn bundle, historical `artifacts/model.joblib`
  untouched, writes confined to `artifacts/phase_three/` + `experiments/phase_three/`.
- **encode/decode**: round-trip and bounds verified for all three search spaces.
- **Sibling scripts**: decoder has proper refuse-overwrite guards and runtime sha checks;
  live-evidence reconciles the running service against the freeze with real network;
  visuals generator writes only its own directory from measured files.

## Test baseline (clean env, 2026-11-18)

`python -m unittest discover -s tests`: **120 OK** ·
`python -B -m unittest discover -s infrastructure -p test_hosting.py`: **28 OK** ·
`node frontend/smoke.mjs`: **PASS** · `node --test tests/frontend_adapter.test.mjs`: **8/8**.

## Final re-run after Role A freeze (exact commands)

```bash
cd /home/d3va/Downloads/ML_Hackathon
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-clean/bin/python -m unittest discover -s tests -q
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-clean/bin/python -B -m unittest discover -s infrastructure -p test_hosting.py
node frontend/smoke.mjs && node --test tests/frontend_adapter.test.mjs
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-clean/bin/python analysis/review/phase_three/audit_phase_three.py
```

The last command is read-only (no training/inference; writes only
`analysis/review/phase_three/audit_report.json`), verifies source-hash drift, cache
integrity, record reconciliation (selection/HPO/freeze), OOF metric recomputation from
saved predictions, probability domain, decoder freeze hash reconciliation, live evidence
presence, and that the historical root model artifact is untouched. It exits nonzero on
any failed check; re-review by hand any check it reports as CHANGED or MISSING.
