# PathSnap final route-tuned HistGradientBoosting

The final70-feature HistGradientBoosting model and directed HMM were selected using TRAIN-only whole-trip validation. Final comparison uses all ten historically exposed student-development trips and all19,160 GPS rows. This is development/reference-based evaluation, not blind localisation accuracy or an organiser score.

| Metric (equal-trip mean) | Previous final | New final |
|---|---:|---:|
| Directed-route edge F1 |0.9513611361|0.9821150783|
| Length-weighted F1 |0.9794127201|0.9883620932|
| Sequence edit similarity |0.9172446210|0.9721423693|
| Route continuity |1.0000|1.0000|
| Failed trips |0/10|0/10|

Classifier-only pooled observation top1 is0.9161795407 and binary candidate-row Macro F1 is0.9533897605; neither is the route-edgeF1 target. Candidate pools start at24, retry48 only on reachability failure. Wider pools change candidate-relative features and the negative-row mix; K12 CV is labelled separately.

The model has180 boosting rounds, learning_rate0.1,31 leaves,min_samples_leaf30,L2=2. Its70 features add stateless geometry-energy/rank/rival-margin transforms to the base38. Final transitions retain deterministic metre-shortest paths, add5 per actually appended arc and3 for endpoint-opposite low-motion edge switches (GPS chord≤20m). Geometry weight0.25,sigma7.1606459163m,beta5m. All penalties are applied before Viterbi; no prediction/reference repair or deduplication is performed.

All64 TRAIN OOF route comparison: v1control edgeF10.8493933028 vsselected0.9103237774, with the same3 failed trips included aszero in both means. The study never claims all TRAIN roads are solvable. Final student development has no failures. Model/context alone with the earlier fixedK12 decoder was rejected; that experiment remains unchanged. This release is the newly frozen fullpipeline, not a rewritten old result.

Artifacts and sources are pinned in release_manifest.json; acceptance.json binds the unchanged full-ten comparison. Original artifacts/final_five is preserved. Serving or CLI use must load the accepted release through backend.target96_runtime; no Gaussian fallback or reference-file inference. Evaluator scores must not be used to tune this release.
