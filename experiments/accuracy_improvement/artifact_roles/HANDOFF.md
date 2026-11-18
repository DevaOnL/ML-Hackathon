# Non-OOF accuracy artifact roles

## Scope

Created only [accuracy_artifacts.py](</home/d3va/Downloads/ML_Hackathon/mapmatch/accuracy_artifacts.py>), [test_accuracy_artifacts.py](</home/d3va/Downloads/ML_Hackathon/tests/test_accuracy_artifacts.py>), and this handoff. All other files remained read-only. No live/original model, dataset, raw-data file, cache, student input, evaluator input, or real trained artifact was read or changed. Existing inference/source/test files were read as code only. No training command or full test suite was run.

[accuracy_matcher.py](</home/d3va/Downloads/ML_Hackathon/mapmatch/accuracy_matcher.py>) was read-only. Its existing OOF loader and validator remain restricted to folds 0/1/2; neither is called with fabricated metadata. Both new wrappers subclass AccuracyEmissionModel and override only __post_init__ to run their own role validator.

## Exported loader API

~~~python
from mapmatch.accuracy_artifacts import (
    load_accuracy_confirmation_model,
    load_accuracy_deployment_model,
)

confirmation = load_accuracy_confirmation_model(
    path,
    expected_sha256=model_sha256,
    expected_train_trip_ids=tuning64_ids,
    expected_validation_trip_ids=reserved16_ids,
    expected_protocol_sha256=protocol_sha256,
)

deployment = load_accuracy_deployment_model(
    path,
    expected_sha256=model_sha256,
    expected_protocol_sha256=protocol_sha256,  # optional; default None
)
~~~

Return types are AccuracyConfirmationModel and AccuracyDeploymentModel, both also instances of AccuracyEmissionModel. Role-specific validation helpers are validate_accuracy_confirmation_metadata and validate_accuracy_deployment_metadata. Invalid pins, metadata, roles, partitions, or probability output raise ValueError. Filesystem/unpickling errors propagate normally.

All externally supplied pins must be lowercase 64-hex strings. Caller identity arguments are validated before opening the artifact. A loader opens its one explicitly supplied file, reads it once, hashes the resulting immutable bytes, and only after an exact SHA256 match calls joblib.load on a BytesIO view of those same bytes. Replacing or modifying the pathname after the hash cannot change the deserialized bytes. This holds one artifact-sized byte snapshot in memory. Hashes establish identity, not pickle safety: artifacts and the manifests providing their expected hashes must already be trusted.

## Confirmation contract: existing producer needs no change

The dictionary envelope emitted by [accuracy_confirmation.py:125](</home/d3va/Downloads/ML_Hackathon/scripts/accuracy_confirmation.py#L125>) is accepted unchanged:

- estimator plus the common metadata below.
- fold_id is the integer 3, not a bool or float. The fold alias, when present, must also be integer 3. Fold 3 stays fold 3 in wrapper metadata.
- Exactly 64 unique train_trip_ids and exactly 16 unique validation_trip_ids, disjoint, with every ID matching [A-Za-z0-9_-]{1,100}.
- training_scope is exactly:

~~~text
TRAIN64 fitting, reserved16 one-shot internal confirmation; never historicallyunseen
~~~

- fit_role is not required or synthesized. If present it must be exactly internal_confirmation16. The exact scope plus fold 3 identifies the existing role; no new producer field is necessary.
- Both caller-supplied expected trip populations are mandatory and must themselves satisfy the same cardinality, uniqueness, validity, and disjointness requirements. Comparison is exact by ID population, independent of input order. The expected protocol pin is mandatory and must match metadata.

This is one-shot **internal** reserved TRAIN16 confirmation. Earlier released fits exposed all80 TRAIN historically; the loader does not establish a blind test, historical non-exposure, permission to repeat confirmation, or permission to tune after confirmation.

## Deployment contract

A final refit dictionary must carry the same estimator/common metadata and:

~~~python
{
    "fit_role": "all_train_deployment",
    "fold_id": None,  # may instead be absent
    "fold": None,     # may instead be absent
    "train_trip_ids": all80_unique_train_ids,
    "validation_trip_ids": [],
    "training_scope": "all80 TRAIN final refit after frozen selection; no student/evaluator fitting",
    # public_name and seed may be included, but are optional.
}
~~~

Exactly 80 unique valid training IDs and an explicitly present empty validation list/tuple are required. Any non-None fold value is rejected, including 0, 3, False, and strings. Missing folds stay missing; None folds stay None. No OOF fold is invented. Optional public_name/seed and other metadata are retained without prescribing new semantics.

The model metadata must always contain a valid protocol SHA256, even when the caller omits expected_protocol_sha256. A supplied expected protocol pin must match. This API intentionally has no expected trip manifest argument; it checks the declared 80-ID population structurally, not against an on-disk manifest. The independently trusted artifact pin must bind the intended all80 population.

## Common metadata and estimator checks

- profile must be canonical v1/context/compact/physics.
- base_feature_names and base_feature_version must exactly match the installed immutable 38-column extractor, including feature order.
- feature_names must match the selected profile in order: v1=38, context=70, compact=12, physics=3. feature_version must be the installed v1 version for v1, otherwise pathsnap-candidate-v2.
- family and run_name must match [A-Za-z0-9_-]{1,100}; parameters must be a mapping.
- protocol_sha256 and every source_sha256 value must be lowercase 64-hex strings.
- source_sha256 must have normalized, repository-relative POSIX path keys (no absolute paths, parent traversal, backslashes, drive prefixes, redundant slash/dot segments, trailing slash, or short OOF aliases). It must include all 18 paths already emitted by the current freeze_selection:

~~~text
scripts/accuracy_common.py
scripts/accuracy_trees.py
scripts/accuracy_linear_bayes.py
scripts/accuracy_knn.py
scripts/accuracy_confirmation.py
scripts/accuracy_decoder.py
scripts/accuracy_route_validation.py
mapmatch/accuracy_features.py
mapmatch/accuracy_matcher.py
mapmatch/features.py
mapmatch/learning.py
mapmatch/experiments.py
mapmatch/phase_models.py
mapmatch/data.py
mapmatch/metrics.py
mapmatch/matcher.py
mapmatch/routing.py
mapmatch/candidates.py
~~~

Additional normalized relative dependency paths are allowed. Lead should include the new [accuracy_artifacts.py](</home/d3va/Downloads/ML_Hackathon/mapmatch/accuracy_artifacts.py>) in future frozen source sets when using it for scoring. Source pins are validated for format, path form, and completeness; the loader does not open source files or compare historical hashes with mutable installed files. The trusted model pin and expected protocol pin bind that provenance. Existing producer-side source-freeze verification remains separate.

The estimator must expose callable predict_proba, pass sklearn check_is_fitted, have exactly the two numeric binary classes 0 and 1 (either order, as in the inherited guard), and have integer n_features_in_ equal to the profile width. Loader construction does not call predict_proba. Trip-ID metadata is normalized to sorted tuples and feature_names to the canonical tuple; scopes, real fold fields, optional role fields, and other provenance are not relabeled. The original metadata mapping is not mutated.

## Inherited inference semantics

score_candidates_batch is the exact inherited method: extract one immutable 38-column matrix from the full supplied graph/track and retained observation groups, then apply groupwise v2 transformation once for non-v1 profiles, then one strict estimator prediction. Candidate order and original observation indices remain unchanged. Raw positive probabilities are retained in records; only emissions receive the inherited 1e-7 positive-column clip followed by within-group normalization.

predict_proba is also the exact inherited method. **Its argument must already have the selected profile width.** It does not run the 38-to-v2 transform or interpret group offsets. Inputs and the final Pipeline classifier boundary use float64. Both raw output columns must be finite float64, in [0,1], of shape (rows,2), with row sums within absolute 1e-10 of one before any clipping. Noncanonical probability output is rejected, not repaired.

## Verification

Ran only the owned generated-only test module:

~~~bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -B -m unittest tests.test_accuracy_artifacts -v
~~~

Result: **18 tests passed, 0.117 seconds, exit 0.** Tests cover both roles and all four profiles, real metadata preservation, no OOF weakening, invalid role/schema/partition/provenance rejection, exact mandatory confirmation identity checks, optional deployment protocol binding, hashing before unpickling, pathname replacement, strict probability validation, grouping/order, and inherited method identity. All joblib files were generated inside temporary test directories and removed by TemporaryDirectory. Probability spies never fit. The only actual fit was GaussianNB on six generated rows with 38 columns and generated binary labels, reused for the two role wrappers. No real model, supplied data, cache, or held-out/evaluation input was opened.

A third-party joblib/NumPy deprecation warning about array.shape assignment was emitted during synthetic deserialization; it did not affect results.
