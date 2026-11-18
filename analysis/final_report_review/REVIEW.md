# Read-only report-contract audit

## Result and scope

This is not report acceptance. [The machine-readable snapshot](<analysis/final_report_review/audit_snapshot.json>) was taken at 2026-11-18T09:20:23Z against builder SHA-256 `1064b000aebce79cc6f70429297c4ddab350119931a7a7e6b8f458bdf537b797`. Its bytes did not change during that run. The report child was still writing independently, so rerun after its handoff before treating any finding as current.

[The audit program](<analysis/final_report_review/audit_snapshot.py>) imports a pinned in-memory copy of [the builder](<scripts/build_final_report.py>), calls its state/fact/guard functions and reads canonical public JSON and asset hashes. It does not call main, render a PDF, fit a model, infer probabilities, fetch HTTP, read raw evaluator files or access private configuration. All audit writes are under this review directory. A zero audit-process exit means the snapshot was collected, not that the report passed.

## Confirmed canonical evidence

- All 37 declared builder inputs exist and load without JSON errors. [The actual core test record](<reports/final_five/test_results.json#L2-L15>) now has 376/376 passes (325 Python + 29 hosting + 22 Node) and `complete_release_suite=false`: final report-specific tests are not yet included.
- All 201 facts resolve. Earlier `live_requested` and live parity/terminal schema errors were fixed by the active child during this review. Fact resolution does not mean the final release suite or report passed.
- [Delivery verification](<reports/final_five/delivery_verification.json>) has 1,534 boolean leaves under check containers, all true, and no defects. Its top-level `checks` is a dictionary of 11 booleans, not nested check objects. [API integration acceptance](<reports/final_five/api_integration_acceptance.json>) has 12 top-level boolean checks. The builder correctly rejects a top-level false in both documents.
- [Live API evidence](<reports/final_five/live/live_api.json#L1-L11>) reports `all_twenty_requested=true`, 20 attempted and 20 successful requests. It uses root `model_sha256`, `config_file_sha256` and `match_config_hash`; it has no `freeze` object. The model is `c9cc8e9f...`, decoder file hash `10405aa6...` and compact API configuration hash `e326b1af...`. Do not conflate the latter two hashes.
- All six paths in [the browser screenshot list](<reports/final_five/live/browser_evidence.json#L228-L283>) exist and match their recorded hashes. This includes the route-map crop. It is not a missing-path defect.
- [The terminal evidence](<reports/final_five/live/api_terminal_evidence.json>) has three events. Inference start/completion carry the model hash; `http_request` is a transport event and does not. It must be tied by request/trip identity rather than forced to contain an invented hash.
- [The final figure manifest](<visualizations/final_five/figure_manifest.json>) contains 23 figures. [The actual Folium map manifest](<visualizations/final_five/map_manifest.json#L1-L42>) names and hashes the final development trip-9 map and its PNG/PDF/metadata/browser/OCR evidence.

There are false booleans elsewhere in the delivery JSON, including measured `exact_match=false` results. Those are not failed acceptance checks. A recursive all-booleans-true predicate would be wrong.

## Build blockers in the pinned source

| Finding | Concrete location and correction |
| --- | --- |
| Datetime class used as a module | [Import](<scripts/build_final_report.py#L47>) binds the class, while [render callback](<scripts/build_final_report.py#L1410>) and [manifest timestamp](<scripts/build_final_report.py#L1494-L1495>) call `datetime.datetime`/`datetime.timezone`. They raise AttributeError. Use `datetime.now(timezone.utc)` consistently, or change the import consistently. No render was attempted here. |
| CLI definition order | [main](<scripts/build_final_report.py#L1558>) calls `expand_tables`, but [the script entry point](<scripts/build_final_report.py#L1638-L1639>) runs before [that function is defined](<scripts/build_final_report.py#L1662-L1670>). Import-based tests miss this. Move the entry-point block after all definitions and add a subprocess `--check` test. |
| Template scanner sees resolved text | [The guard](<scripts/build_final_report.py#L1076-L1082>) and [main](<scripts/build_final_report.py#L1561>) scan resolved Markdown after tokens have been replaced. Every required fact then appears unused. Scan the original template for token coverage/unknown tokens; scan resolved output separately for pending markers, missing links and stale paths. Preserve fact tokens in the authored template. |
| The 23rd final plot is not registered | [The registry](<scripts/build_final_report.py#L188-L227>) omits `visualizations/final_five/plots/final_five_route_ablation.png`. The existing figure verifier therefore fails even on synthetic Markdown containing every registered image. Add a registry entry and include/hash-check the actual decoder plot. |
| Actual final Folium map is not bound | The registry and 37-input list do not include [the map manifest](<visualizations/final_five/map_manifest.json>) or [the final map image](<visualizations/final_five/maps/final_development_trip_00000009.png>). Add the map manifest, metadata/browser record and image binding. Use its `files` array, not an assumed `figures` array. Include the static PNG and link the HTML; do not put reference-containing map evidence into the frontend. |

## Fail-closed hardening proved by synthetic mutation

These mutations were made only to copied in-memory state. Canonical evidence was not modified.

1. [verify_delivery](<scripts/build_final_report.py#L806-L817>) accepts `checks=[]` when `passed=true`. It also accepts a copied record with `datasets.development.trips.00000000.offline_checks.cli_status=false` while top-level summaries stay true. Require a nonempty dictionary and recursively validate boolean leaves within check containers, including per-trip checks. Do not apply that predicate to metric/diagnostic booleans. The actual evidence shape is valid; these are guard weaknesses, not claims that current delivery failed.
2. [The final-test guard](<scripts/build_final_report.py#L1051-L1057>) accepts `{"failures":[]}` because both missing counts are None. It also accepts the real 376/376 core record despite `complete_release_suite=false`. Final mode must require integer `total>0`, integer `passed==total`, an empty failure list, successful measured suite commands and `complete_release_suite=true`. The Lead will measure report-specific tests and update the record after handoff.
3. [Figure verification](<scripts/build_final_report.py#L719-L746>) treats present live images lacking browser hash bindings as notes, not errors. Clearing the browser screenshot list reproduces this. For final mode, every referenced screenshot must have a matching canonical hash. Preview may retain placeholders or warnings.
4. Requirement rows remain honestly pending in [the matrix](<analysis/final_requirements/requirements_matrix.json>). The final guard currently identifies seven report/publication rows. The Lead should advance them only after the actual report/repository checks; do not change them merely to make this guard green.

## Content checks still required after template handoff

The template did not exist at the snapshot, so the following are unverified, not failed or passed:

- [R-DL-05](<analysis/final_requirements/requirements_matrix.json#L677-L690>): one consolidated table with exactly these nine columns: Model; Algorithm Family; Key Configuration; CV Macro F1-score; Top-1 Accuracy; Route continuity; Student-Test Macro F1-score; Log Loss; Inference Time. Populate all five prescribed families and distinguish the tuned HistGB configuration. Label binary-candidate versus observation/route metrics and time units.
- [R-DL-06](<analysis/final_requirements/requirements_matrix.json#L693-L706>): measured per-model explanation of suitability, strengths/limitations, initial hyperparameters, objective/loss, student results and evidence of over/underfitting. Current registry exposes some precision/recall/time facts only for tuned HistGB; add actual per-family facts where the authored table/text needs them, rather than type values into prose.
- [Epoch histories](<docs/FINAL_REQUIREMENTS.md#L15-L19>): explicit N/A for each pipeline, explaining that this recorded sklearn workflow has no epoch history. The partition-loss and Bayesian-trial plots must not be labelled synthetic epoch curves.
- [Every experimented model](<docs/FINAL_REQUIREMENTS.md#L13>): a separate historical appendix for RandomForest, ExtraTrees and earlier tuned Bagging, using [historical results](<experiments/results.json>). They are not sixth/seventh eligible families in the final five-family selection. Show exact measured configurations and preserve the reused 12-trial versus fresh-fit distinction.
- [Evaluator caveats](<docs/FINAL_COORDINATION.md#L33>): no claim that every metric improved, no independent/geographic-holdout claim, disclose zero exact evaluator routes and the weak trip. Model/classifier proxy accuracy and decoded macro F1 can move in different directions.
- Include actual dark/light and evaluator UI evidence, receiving-terminal request binding and the final Folium map with their canonical identities. Hash/DOM/OCR checks do not constitute visual inspection. No manual visual claim is made by this audit.

## Rerun after the child handoff

~~~bash
.venv/bin/python -B analysis/final_report_review/audit_snapshot.py
~~~

The program writes only [its snapshot](<analysis/final_report_review/audit_snapshot.json>). Compare `source_sha256` and `source_changed_during_run`, then review the actual template and run the Lead-owned final suite and report checks. The synthetic scaffold results in the snapshot test evidence plumbing only; they do not certify authored report content or a PDF.
