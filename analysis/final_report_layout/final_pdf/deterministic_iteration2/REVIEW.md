# Deterministic final PDF: automated layout GO

Final PDF: [PathSnap Final Technical Report](<reports/final_five/PathSnap_Final_Technical_Report.pdf>)

SHA-256: `f6bdd47595a4e06295ed23b10db94f22c7bf69484f8bf9476ae15785ce2e65e2`

[The exact-artifact acceptance record](<analysis/final_report_layout/final_pdf/deterministic_iteration2/comparative_acceptance.json>) passes all 14 checks. Its SHA-256 is `dca4321e9ef1da96c304653094473726b4029a5cd0bddf78f48c7f0d363ded65`. This replaces the superseded 1503a65c handover pin while retaining that artifact's complete strict render/OCR record as comparative evidence.

The current PDF has 57 pages. Every page size and all 55 image dimensions/placements match the [earlier fully rasterized strict check](<analysis/final_report_layout/final_pdf/verifier/layout_report.json>) within 0.001 PDF point. Per the Lead's instruction, the 57 page rasters were not duplicated. Current PDF text/spans were checked again for required topics, unresolved markers, preview watermark, bad glyphs, out-of-bounds text, blank pages and overlaps; none failed.

[Fresh current-PDF supplemental checks](<analysis/final_report_layout/final_pdf/deterministic_iteration2/supplemental_layout.json>) confirm 55 of 55 actual embedded raw-pixel asset matches and all 12 source/crop/PDF placement bindings. Base text estimates in enlarged crops remain at least 9.47 pt. There are 62 clickable link annotations and no literal Markdown link syntax. The mandatory nine-column comparison stays on page 14, seven rows at 8.1 pt. [Fresh 300 dpi OCR](<analysis/final_report_layout/final_pdf/deterministic_iteration2/mandatory-table-ocr.txt>) recovers all 29 distinct numeric samples, with mean confidence 86.53.

The actual PDF now says 421 total tests, 421 passed and zero failures. The [release test record](<reports/final_five/test_results.json#L2-L16>) has 370 Python, 29 hosting and 22 Node tests, with `complete_release_suite=true`. The [normal final manifest](<reports/final_five/report_manifest.json#L769-L785>) matches the current PDF, and all 17 guard results, 55 asset hashes and builder/template/Markdown hashes match the current files.

The deployment-export paragraph in actual PDF text retains `c9cc8e9f36838f656a94d07ed69632979df9981ca94a9daf3b52fbce052e4b34`. The compared artifact remains explicitly separate. Current source identities are builder `ae6b72a7455a8105054d36568df46a20db201e9b5db2cd1d6a412e2a5e051c1a`, template `637096eb4bfc5ab73df5181dcd8a776fd1b9376c1cf545df8168177576a86f4f`, and resolved Markdown `909b9bca8f51907000c0ef90bc9d3391dacc7155bef1ebf94c251da0f1b7eddb`.

This GO is for automated layout and artifact binding, not manual visual inspection, scientific certification or organizer approval. No PDF, source, template, original screenshot, model, raw evaluator file or private configuration was modified. No HTTP, fit, inference, browser or service operation ran. The one supplemental background job finished with exit code zero and its output was collected.
