# Experimental accuracy candidate — NOT APPROVED FOR DEPLOYMENT

This is a genuine HistGradientBoosting classifier refitted on all80 authorized TRAIN trips after the feature/model/decoder selection freeze. It uses70 stateless context features and strict float64 learned probabilities. No student/evaluator data were fitted.

The candidate improved top-1 and route F1 in internal prospective TRAIN16 confirmation, but those trips had historical exposure in earlier releases. Its final historically used student-development comparison had2 failures out of10 under the frozen fixed-K12 decoder. The original live release retains a12→24→48 retry policy. Failed trips count aszero; they are not dropped.

**Acceptance is false. Do not deploy this bundle or replace the original artifacts.** The opt-in backend refuses this release unless an accepted hash-bound receipt is present; the recorded receipt deliberately rejects it. Do not edit the receipt to bypass the gate.

See [measured addendum](<../../reports/accuracy_improvement/ACCURACY_REPORT.md>), [final release decision](<../../reports/accuracy_improvement/release_decision.json>) and [same-service rollback guide](<../../docs/ACCURACY_RELEASE.md>). The original production model remains in [the final-five bundle](<../final_five/model.joblib>).
