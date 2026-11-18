# Route-edge-F1 improvement study

Target: raise the equal-trip mean original-directed-route edge F1 over all ten student-development trips from 0.9513611360981356 to at least 0.96. This is not classification accuracy, an independent per-GPS ground-truth score, or an organiser-certified metric.

Selection uses TRAIN trips only. Stage 1 preregisters 12 joint model/decoder settings on 18 deterministic, whole-trip out-of-fold trajectories. Stage 2 is a separate prospectively registered TRAIN precision refinement: 9 configurations on those 18, then two finalists plus the paired v1 control on the remaining 46. The final decision ranks all 64 OOF routes with failures retained as zero. The 16 historically used TRAIN confirmation trips are not relabelled as fresh holdout.

The new arc prior adds a finite cost for the exact number of original arcs Viterbi would append from the deterministic shortest-metre path. It never deletes repeated arcs, repairs routes, changes the scorer, or uses reference geometry during inference. Reversal penalties are finite; reachable real cycles remain possible.

Original production model/config/report and previous failed experiments are preserved. The final all80 artifact and complete decision are frozen before one named full student-development comparison. Student development has historical exposure; it is not a blind test. Evaluator data do not determine any tuning or release decision. A missed target remains a missed target.

Private candidate arrays, labels, per-observation scores and full TRAIN route states stay under ignored private directories. Public JSON contains scalar measurements, configuration, failure counts and hash bindings.
