# Accuracy-improvement release

The new model is opt-in; original final-five weights, configuration, report and prediction archives stay intact. The existing backend entrypoint chooses the accepted accuracy runtime only when `PATHSNAP_MODEL` resolves exactly to the new all-TRAIN model. A missing or rejected acceptance receipt fails closed. No second server is required.

## Evidence

- [Frozen five-family selection](<../experiments/accuracy_improvement/confirmation/frozen_selection.json>) includes the prospective TRAIN64/16 split, model/feature choices and selected dense OOF decoder.
- [Expedited budget amendment](<../experiments/accuracy_improvement/BUDGET_AMENDMENT.json>) discloses the optional second Bagging finalist stopped at the user request, before confirmation.
- [Measured tuning results](<../reports/accuracy_improvement/TUNING_RESULTS.md>) distinguish tuning from confirmation.
- The accuracy addendum reports all five paired internal comparisons, full-route checks and the final historical-student comparison, including declines.

## Same-service activation

After the hash-bound final acceptance gate passes, set all three dedicated environment entries together:

```ini
PATHSNAP_MODEL="artifacts/accuracy_improvement/model.joblib"
PATHSNAP_MODEL_METADATA="artifacts/accuracy_improvement/model_metadata.json"
PATHSNAP_MATCH_CONFIG="artifacts/accuracy_improvement/match_config.json"
```

Validate the existing dedicated launcher, then restart only `path-snap-api.service`. Its existing tunnel and public URL are unchanged. Verify health reports the new model hash, then compare a real request against its saved prediction evidence. Do not activate a candidate rejected by the final gate.

## Rollback

Restore the three `artifacts/final_five/` paths in the dedicated runtime environment and restart the same API service. The original artifacts are never overwritten. Default local CLI and API behavior remains on the original release unless the exact new model path is explicitly selected. The legacy CLI does not silently accept the new role-qualified bundle.

## Limits

The reserved TRAIN16 trips were historically exposed to earlier model releases. Their paired one-shot confirmation is internal evidence, not a blind official test. Student data were used historically for development; their final comparison does not feed new tuning. The already-scored evaluator is not used in this improvement study. Higher top-1 does not imply every secondary metric improves.
