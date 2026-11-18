# Pre-fit and pre-confirmation review fixes

The Sol 6.1 MAX read-only protocol reviewer reported two HIGH launch/provenance risks before actual training. Both were corrected before training authorization: exact preregistered fold/menu matching with saved-trial source/protocol/parameter validation, and launch preflight plus provenance-checked completed-trial resume instead of poisoning an output on an unauthorized start. Synthetic safety checks passed before measured fits.

The separate MAX confirmation reviewer reported two further gate risks while tuning fits were running and before any confirmation scoring or selection freeze: transitive scorer/weight/bank/feature source pins were not revalidated after freeze, and an incomplete third promoted full-CV finalist could be omitted. The confirmation runner now checks every frozen trial dependency on entry and exit, pins the inference/metrics/route modules, and requires the completed full-CV set to equal the exact promoted set.

A later MAX dense-route reviewer identified incomplete transitive per-route provenance in the original route-result schema. Measured route files and sources remained untouched. A separate execution attestation verifies the original pre-run candidate manifest, both trial records, every fold model, the complete original decoder dependency map and every dense route result. Final selection freeze requires this attestation and binds its exact hash. It is described as a postexecution integrity audit, not a replacement measurement or independent recreation.

A final integration reviewer found aggregate trip filenames could conceal duplicate embedded trip identities. Before any reserved/student comparison, the Lead added exact expected unique16/10trip identity, model/fit/export/source/configuration bindings and hash-bound per-trip records; export requires all16 unique paired route observations.

Before the irreversible science freeze, final integration review fixes also recomputed the decoder table and winner from the original16predictions per route, rejected duplicate/empty trip bootstrap inputs, recomputed finite export gate deltas, and derived the opt-in runtime model/config paths consistently. The final frozen comparison sources contain these corrections.

These are automated code reviews, not human or organizer approvals. No accuracy gain is inferred from a review. No reserved16 score existed when these fixes were made. The original released sources, artifacts, completed report and live model were not changed.
