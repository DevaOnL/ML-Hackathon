# Frozen evaluator protocol

Preregistered at 2026-11-18T08:47:31.075Z, before this evaluator opened any new input file. The Lead first listed the private evaluator trip names after 08:44 UTC on 2026-11-18. This run owns the first parse and offline scoring of those files.

## Fixed objective and boundaries

Evaluate all ten supplied trips (15, 17, 23, 30, 49, 59, 67, 73, 87 and 94, padded to eight digits) once using the final selected learned model and decoder. This is the final postfreeze evaluation. Selection, fitting, feature changes, calibration, decoder changes and result-driven retries are prohibited. No API, network, service, Git or UI action belongs to this run. A failed CLI attempt keeps its partial outputs and log; it is never rerun or tuned.

The model SHA-256 is c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34, frozen at 2026-11-18T08:26:50.372940+00:00. The decoder file SHA-256 is 10405aa6b6f370e41d4ddcb01207fd1e8f58900fd0395e541ee6bc78146168c5, frozen at 2026-11-18T08:37:21.093991+00:00. Validate these files, frozen metadata/schema, all five model acceptance records, clean-load evidence, independent 111/111 review and actual student CLI verification before the first evaluator input open. Recheck frozen hashes at each stage. The fixed candidate retry widths are 12, 24 and 48; no setting changes after results.

## Input audit before prediction

Use mapmatch.data.discover_trips and the exact per-ID nodes, arcs and track paths. Require the ten distinct declared IDs, reject symlinks and paths outside the trusted root, and validate every graph and full original track before the all-ten CLI starts. Check finite coordinates, projected local coordinates, arc endpoints and finite nondecreasing timestamps. Reports publish counts, duration, non-coordinate ranges, per-extension checksums and structural status. They never publish raw coordinates, absolute raw paths, candidates or arrays. Reference files are neither parsed nor hashed until all predictions have finished and passed verification; the preprediction audit records their hash as pending.

Compare each evaluator track/nodes/arcs checksum against every one of the 80 frozen training IDs and all ten student-development trips. Report matching contents separately from matching IDs and report both partial-file and full-trip matches. An identical ID alone does not establish data overlap. Any full duplicate makes an independent-holdout claim invalid; no claim of an unseen official set follows merely from a folder name. Overlap findings cannot change selection or which ten trips are evaluated.

## One actual prediction run

Use the original .venv with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and PYTHONDONTWRITEBYTECODE=1. The command is fixed:

```sh
.venv/bin/python -u -B -m mapmatch predict --input evaluator_test --output outputs/final_evaluator_test --model artifacts/final_five/model.joblib --config artifacts/final_five/match_config.json
```

Require a new empty output location and preserve the CLI exit status. Record an honest stage plan before launch. Inference must not load references. After exit zero, validate all ten nonempty connected directed routes, a flat predictions-only ten-member ZIP, saved observation count/order/timestamps, actual classifier columns, finite probability audit, learned model loaded and learned emissions for every original observation. Publish route, matched-state and predicted-geometry hashes for the Lead's independent API parity checks. Keep observations CSVs and run-level raw paths private.

## Scoring after all predictions

Only after full prediction verification may the scorer open the private reference routes. Preserve repeated arcs and zero-length arcs. Check exact graph arc IDs; report broken reference joins without repairing or dropping them. Use mapmatch.metrics.score_routes for directed-arc multiset F1, length-weighted occurrence F1, edit similarity, continuity and exact match. Report all ten rows, trip-macro means and contributing counts. Missing or structurally invalid references give null metrics with a reason, never synthetic zeroes.

Derive full-observation proxy labels with mapmatch.features.reference_observation_labels after prediction. Report decoded and saved classifier top-1 accuracy and graph-namespaced macro F1. These nearest-reference geometry labels are proxies, not true GPS road labels. Use backend.scoring helpers when the stored CLI fields suffice. Candidate categorical NLL is null with an explicit reason when the CLI did not save the complete probability arrays. Never rerun inference or manufacture probabilities to fill that field. HMM posterior NLL is unavailable.

## Acceptance and privacy

Acceptance must contain a check for each trip and each required invariant, frozen provenance, the prediction-before-reference chronology, overlap disclosure, and an expected-trips hash dictionary for API reconciliation. Keep the audit, prediction-verification and score stages separate and refuse overwrites. Public JSON contains only fold labels, trip IDs, hashes, counts, timings, scalar metrics and declared limitations. The predictions ZIP contains only the ten predicted route files. The raw evaluator tree, observations, candidate arrays and references remain private. External viva and plagiarism review remain outside this offline evaluation.
