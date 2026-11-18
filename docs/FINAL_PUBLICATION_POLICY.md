# Final publication boundaries

Private organizer documents, raw TRAIN/student/evaluator graph and GPS files, reference route files, feature caches, row-level supervision, candidate probability NPZs, observation CSVs, raw diagnostic arrays and host credentials are not published.

Public evaluator evidence is limited to the twelve explicitly named JSON reports produced by the guarded post-freeze scorer: freeze validation, stage plan, preprediction audit, actual prediction attempt/result/verification, reference-scoring start, final input audit, aggregate/per-trip route metrics, parity expectations, final acceptance, and its privacy/test verification. They contain hashes, counts, model/config provenance and aggregate metrics rather than raw inputs or aligned labels. A filename that happens to start with evaluator is not sufficient for publication.

The separate evaluator API contract document is allowed. Predicted route outputs and the flat ten-route submission ZIP are derived outputs, not supplied reference routes. Their private diagnostics and observation arrays remain excluded. Development-only Folium report maps are never served by the API static allowlist.

The exact Git index, not an unreviewed working tree, is scanned before commits. Archives must contain exactly ten uniquely named predicted .route members and pass ZIP CRC checks. Every published archive member must also pass the secret scanner. All final branch blobs receive a second scan before the pull request.
