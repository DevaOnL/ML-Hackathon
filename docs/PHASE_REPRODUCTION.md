# Reproducing the phase three inputs

The phase three runner (`experiments/phase_three/run_phase_three.py`) does not
build its own datasets. It loads shared artifacts from earlier runs, most of
which are git-excluded, so a fresh clone cannot run it until those files exist
again. This page documents what is required, which pieces can be rebuilt, and
the one tool that does the rebuilding: `scripts/prepare_phase_three_inputs.py`.

Scope first, because it shapes the tool. The phase three effort covers three
model families: logistic regression, distance-weighted KNN, and bagging. The
preparation tool therefore rebuilds shared inputs with logistic regression
only. It never calls `run_experiments`, never retrains the other families, and
never writes into `experiments/phase_three/` or `artifacts/phase_three/`;
those directories belong to the phase runner and its owner.

## What the runner needs

| File | Git status | How it comes back |
| --- | --- | --- |
| `experiments/cache/train_edee9a8f151dd25e.npz` | excluded | rebuilt by `--stage train` from raw train data |
| `experiments/cache/student_test_240a8fbfc79205b3.npz` | excluded | rebuilt by `--stage student` from raw student data |
| `experiments/cv_folds.json` | included in publication | created by `--stage train` if missing, validated if present |
| `experiments/cv/logistic_regression.json` | included in publication | created by `--stage train` (LR-only grouped CV) |
| `experiments/predictions/logistic_regression_oof.npz` | excluded | rebuilt by `--stage train` via an LR-only mirror |
| `experiments/evaluation/logistic_regression.json` | included in publication | restore from the published repository; never fabricated |
| `experiments/predictions/logistic_regression_student.npz` | excluded | rebuilt by `--stage student` after the freeze gate |
| `artifacts/baselines/logistic_regression.joblib` | included in publication | loaded read-only; refit only with `--allow-lr-fit` |
| `experiments/results.json` | included as historical provenance | restore from the published repository; schema-checked only |

The last three rows are the ones to read twice. The evaluation record and
`results.json` are measured history; the tool verifies them and refuses to
invent substitutes. The baseline joblib is used read-only by default because
the committed artifact reproduces the historical student probabilities to
2.2e-15 under the current runtime.

The [selection record](<../experiments/phase_three/selection.json>) and the
[artifact freeze record](<../artifacts/phase_three/model_freeze.json>) come from
the phase runner itself (selection then freeze stages). They are prerequisites
for `--stage student`, not outputs of this preparation tool.

## The three stages

All commands run from the repository root with the project virtual
environment. Exit codes: 0 success, 1 verification failure, 2 missing inputs.

### check

```bash
.venv/bin/python scripts/prepare_phase_three_inputs.py --stage check
```

Read-only. It hashes the raw train bytes (never parses labels), verifies the
train cache against pinned per-array digests, recomputes the LR OOF metrics
when the committed CV record is present, and lists anything missing. Student
raw data is only counted in this mode; pass `--hash-student-raw` to also hash
the raw student bytes and deep-verify the student cache.

### train

```bash
.venv/bin/python scripts/prepare_phase_three_inputs.py --stage train --data-root <path-to-Student-folder>
```

Rebuilds the train cache, records `experiments/cv_folds.json` if absent, and
produces the LR OOF record: the committed CV JSON is reused as-is, and only
the missing prediction arrays are rebuilt by a logistic-regression-only mirror
of `cross_validate_model`. The rebuilt metrics must agree with the committed
record within 1e-9 or nothing is written. If the CV record itself is missing,
the shared `cross_validate_model` runs once, logistic regression only.

### student

```bash
.venv/bin/python scripts/prepare_phase_three_inputs.py --stage student --data-root <path-to-Student-folder>
```

Runs only after `experiments/phase_three/selection.json` selects one of the
three in-scope families and `model_freeze.json` matches the frozen
`model.joblib` byte for byte. Student data is not touched until both gates
pass. The stage then rebuilds the student cache, loads
`artifacts/baselines/logistic_regression.joblib` read-only, and writes the
student predictions after the recomputed metrics agree with the committed
evaluation record within 1e-9.

`--allow-lr-fit` is the documented opt-out for a missing baseline: it fits a
fresh logistic regression on all 80 train trips and labels the artifact with
`regenerated_by` in its metadata. It is a deliberate regeneration, recorded
as such, not a reconstruction of the historical baseline.

## Current project state

Everything above is already present here. The check run on 18 November 2026
finished with exit 0 and the verdict "every phase-required shared input is
present and verified". All pinned digests matched:

| Pin | Value |
| --- | --- |
| train cache sha256 | `db523c112a6119f16b6f473d6221e3c56b6730ccb2db64dee55433aedbf3ba9e` |
| student cache sha256 | `ae96abb4784501b2b086c9b4aa2f5668e1c9d7f29abda907b7f402118af81b73` |
| cv_folds.json sha256 | `175e9476604746bc2d801c4d2e0eb5a2a5876ecf4f95d730443cd5841d5fd5fa` |
| cv LR record sha256 | `ae4bb5a57b1ef3c22d388d189260c7e93353e6a0887d5c9767434b08fdd4815b` |
| OOF npz sha256 | `937a54e0139b90c999edfb8675eb2afe71a1ffd7d75a61f343c104b7d9bc6432` |
| evaluation LR record sha256 | `9c385f4ea675dce282a887d4c43ead3b0b777fad649456a2917fd7febcc8cb61` |
| student npz sha256 | `31562ba3540e4d7bab0c07d7b5a30dbb05f524ccde4fbb5967b8d041201d3c9b` |
| baseline joblib sha256 | `29157349d52a993d33cae78ed244805a9f087e79b6dc914aaa3b4461c02a58c9` |
| results.json sha256 | `4f308d422263e313b13253e2bca2267dba6a445303fbc7a404d9c44d9d1c1198` |

The recomputed OOF metrics agreed with the committed record within 1e-9. In
this project, `check` is the only stage worth running; `train` and
`student` exist for the fresh-clone case.

## Fresh clone sequence

```bash
# 1. See what is missing and verify whatever is present.
.venv/bin/python scripts/prepare_phase_three_inputs.py --stage check --data-root <path-to-Student-folder>

# 2. Rebuild the train side (cache, folds, LR CV and OOF records).
.venv/bin/python scripts/prepare_phase_three_inputs.py --stage train --data-root <path-to-Student-folder>

# 3. Restore the committed historical records that are not fabricated:
#    experiments/evaluation/logistic_regression.json and experiments/results.json.
#    Copy them from the published repository.

# 4. Run the phase runner through its selection and freeze stages so that
#    selection.json and model_freeze.json exist.

# 5. Rebuild the student cache and LR student predictions.
.venv/bin/python scripts/prepare_phase_three_inputs.py --stage student --data-root <path-to-Student-folder>
```

If the baseline joblib is unavailable, either copy it from the published
repository or pass `--allow-lr-fit` and accept the labeled regeneration.

## Runtime honesty

The original shared run used Python 3.14.6, numpy 2.5.2, scipy 1.18.0,
scikit-learn 1.8.0, and joblib 1.5.3. The pinned environment in this
repository differs in one component: numpy 2.5.5. The tool prints this drift
at startup and pins every array in the cache by shape, dtype, and sha256, so a
mismatch is reported per array rather than as one opaque hash.

What that means in practice: `np.savez_compressed` output is deterministic
within a runtime, but byte-identical npz containers across numpy 2.5.2 and
2.5.5 are unverified. If a rebuilt cache hashes differently, the tool prints
the per-array digests and stops. Compare them with the pinned digests in
`scripts/prepare_phase_three_inputs.py` (`TRAIN_ARRAY_SHA256` and
`STUDENT_ARRAY_SHA256`). Matching arrays under a different container hash
mean the content is right and only the container differs; either rerun under
the original numpy 2.5.2 or accept the new file as a deliberate rerun with a
recorded reason. The tool itself does not decide this for you, and it never
rewrites a pin to make a mismatch disappear.

The cache fingerprint also depends on the shared experiment configuration.
Both pinned fingerprints reproduce with `bayesian_trials=12` (the saved shared
run), not the phase three HPO budget of 8; the tool says so in its startup
notes.

## Tests

`tests/test_phase_input_preparation.py` covers the tool on synthetic fixtures
in temporary directories: fingerprint determinism and sensitivity, the
read-only guarantee of `check`, refusal to overwrite existing files, the
per-array honesty check on a corrupt rebuild, agreement between the LR mirror
and the shared `cross_validate_model`, the student gates (selection, freeze,
sha), the metric-mismatch refusals, and the `--allow-lr-fit` path.

```bash
.venv/bin/python -m unittest tests.test_phase_input_preparation
```

These tests are fixtures and mocks by construction; they say nothing about
the real project data. The live evidence for this repository is the check run
summarized above, and the two are kept separate on purpose.

## Outstanding items

- The all-80 fit, model freeze and all-ten student diagnostics are complete.
  The byte-bound freeze is [model_freeze.json](<../artifacts/phase_three/model_freeze.json>);
  actual artifact and prediction replays passed in
  [model_acceptance.json](<../reports/phase_three/model_acceptance.json>).
- Completed and partial phase outputs now refuse accidental overwrites or
  refits before loading data. Use the frozen artifacts for inference. Rebuild
  inputs with this tool in a separate working copy, and use a deliberately
  separate output tree for a new numerical experiment; do not delete the
  supplied freeze just to make a command run.
- The CV folds, aggregate records and LR baseline are included in the reviewed
  publication scope; row-level caches remain private and must be regenerated
  from authorized local raw data.
- The numpy 2.5.2-to-2.5.5 container question above is open by design; it
  resolves itself the first time someone rebuilds a cache, and the per-array
  digests make the answer unambiguous.
