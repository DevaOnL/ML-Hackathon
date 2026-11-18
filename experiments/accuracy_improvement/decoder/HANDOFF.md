# TASK52 decoder handoff

Status: implementation complete; scoped synthetic tests passed. No actual dataset/model training, supplied trip decoding, holdout/student/evaluator input/reference/result access, or live/artifact/report writes were performed by this worker. Only a tiny generated 6-row GaussianNB fixture is fitted inside a test. No improvement claim is made.

## Frozen first-pass science

Eight preregistered configurations live in [preregistered_menu.json](<experiments/accuracy_improvement/decoder/preregistered_menu.json>). The Lead-approved first pass fixes K=12 and radius80 with inherited empty-only radius expansion up to1000; candidate_retry_limit12 prevents hidden K expansion. Every raw GPS row is retained. Fixed physical sigma7.5m; beta2/5/10; classifier weights1/2; geometry weights1/.25; two bounded transition-cost variants use a10m mismatch deadband and optional stationary-only penalties. Constants are not calibrated from any routes. K24/thinning are synthetic-only tests, deferred distribution-policy experiments, not current selection configurations.

The cheap slack means max(abs(network_distance - GPS_chord) - 10m,0)/beta. It is not proactive horizon widening and performs no additional routing. Stationary chord<=3m adds same-edge backward-displacement cost beyond1m deadband, plus2 for an immediate directed reverse. True cycles remain in reconstructed routes. No postprediction repair/deduplication. All decisions were recorded before decoder runs.

## Imports and API

[accuracy_matcher.py](<mapmatch/accuracy_matcher.py>) exports:

- `AccuracyEmissionModel(estimator, metadata)`; metadata must be the common fold envelope excluding estimator. Canonical profiles v1/context/compact/physics only. Current common bundle with base_feature_version and fold_id is supported.
- `load_accuracy_emission_model(path, *, expected_sha256, expected_fold_id, expected_train_trip_ids, expected_validation_trip_ids, expected_protocol_sha256=None)`. Mandatory partition/hash pins; hash checked before trusted joblib unpickle. Legacy released all80 bundles fail. Match protocol_sha256 against the frozen training-protocol byte pin saved in the model/manifest, **not** later protocol bytes changed to authorize decoding.
- `AccuracyMapMatcher(graph, config=None, emission_model=None, transition_policy=None)` reuses v1 Viterbi/retry/reconstruction, overriding transition costs only. No reference imports/calls.
- `AccuracyTransitionPolicy(...)`, `strict_binary_probabilities(estimator,X)` and graph/GPS-only motion diagnostics.

[accuracy_decoder.py](<scripts/accuracy_decoder.py>) exports:

- `decoder_configurations()` -> eight frozen `DecoderConfiguration` values, with `.name`, `.match_config` (alias `.config`), `.transition_policy`.
- `verify_menu_registration()` -> exact preregistration file SHA256 after code/menu consistency check.
- `selection_plan(folds, *, tuning_trip_ids=None)` -> six dicts containing fold_id, trip_id, train_trip_ids, validation_trip_ids, selection_sha256. Input is common.get_folds(common.load_train_dataset()); validates same64 universe/disjoint folds and takes two SHA256(20261118:id)-smallest validation IDs per fold. Pass registered tuning IDs as additional guard. No dataset is opened by this helper.
- `assert_paired_models(new_v1_baseline,candidate)` -> requires identical fold train/validation IDs and training-protocol pin. Both fitted at common float64 precision. Historical all80 in-sample is not paired control.
- `require_decoder_authorization(protocol, *, candidate_manifest_sha256, decoder_menu_sha256)` -> requires separate `protocol.decoder_selection={authorized:true,model_menu_frozen:true,candidate_manifest_sha256:<pin>,decoder_menu_sha256:<pin>}` plus training_authorized true. Lead owns setting this only after model menu freeze. No raw file loader/runner is provided.
- `decode_arrays(graph, full_raw_track, model, configuration, *, trip_id)` -> requires trip held out of model fit and registered config; dense full-track inference with threadpool cap2. **Lead must apply the authorization gate before loading raw trips/calling this helper.**
- `score_decoded_route(graph,track,result,target_route_array)` -> postprediction only, reuses unchanged score_routes, reference_observation_labels and backend classification_metrics on canonical edges. Reports length/edge F1, exact, continuity, time, proxy edge accuracy/macro-F1, extra_arc_count=predicted-common multiplicity. Prediction never changed.
- `aggregate_route_records(records, expected_trip_ids)` -> every selected trip exactly once, failures contribute0 to length/edgeF1 rather than disappearing. Caller retains all detailed successes/failures/worse trials and ranks equal-trip mean lengthF1, then edgeF1, then preregistered menu order.

CLI is intentionally only `python -B -m scripts.accuracy_decoder --show-menu`; no fit or raw decode option.

## Lead run sequence (not executed here)

1. Freeze model candidate manifest and route menu hashes, authorize decoder_selection explicitly. Preserve fitting-protocol hash separately if protocol file changes.
2. Build six-route selection_plan from tuning64 common folds.
3. Load each candidate and directly-refitted v1 baseline fold estimator using identical expected train/validation IDs plus manifest hashes; assert_paired_models.
4. Load only each permitted TRAIN raw graph/full track; decode_arrays before accessing its target. Do not reuse old350-row probability archives as full-track emissions.
5. Score completed predictions with unchanged TRAIN target arrays. Record every failure and all mandatory metrics. Select by lengthF1 then edgeF1; do not select by likelihood, especially across candidate policies.
6. Freeze final decoder/model before Lead-owned single prospective16 confirmation/student comparison. No holdout-selected retuning.

## Verification

`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 .venv/bin/python -B -m unittest tests.test_accuracy_matcher tests.test_emissions tests.test_viterbi_oracle -v`

Final verification: all37 synthetic tests passed in0.215s (exit0), including final None-schema/fold-type assertions; evidence in [synthetic_tests.log](<analysis/accuracy_decoder/synthetic_tests.log>). The TASK52-specific suite has27tests. Coverage includes all4feature profiles, ordering/offsets/fulltrack context, both class columns sum1tol1e-10 before clip, GNB final float64 after custom downcast, schema/fitted/partition/hash rejection, no file/target access during inference, disabled-policy v1 state/cost parity, no cycle deletion, feasible badpath no-Kretry, feasible transition no horizon retry, interval/distance semantics, fixed-Kempty fallback, deterministic sixfold plan, protocol gate, paired baseline and failures denominator. Joblib emitted only an upstream NumPy2.5 shape-assignment DeprecationWarning during synthetic pickle loading.
