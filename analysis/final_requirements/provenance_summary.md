# Provenance summary: original-requirements audit

Audit run on 18 November 2026 by the requirements-audit subagent (fc571adb). Scope was fixed at delegation: list, read and verify the original organizer documents, and produce this audit. No model work, no web access, no evaluator data.

## What was audited

Six DOCX files in PathSnap_student_materials/. Five are the original organizer documents; the sixth is a synthesis file that was cross-checked only:

| File | Role | SHA-256 (binary) | Non-empty XML paragraphs |
|---|---|---|---|
| Main Guidelines.docx | original | 84279a8748ec6de9707ddbdcf13eadd230339d88d567cee6837c225df48274d2 | 36 |
| Dataset Exploration.docx | original | 87d93a36979cec6e57647506768e2275ffe437ffdf3b175f2344cf68e55109e2 | 90 |
| Models and Training Guidelines.docx | original | e85fae3503eba502485469e41ea69fad5feaf0a6c8b3c55248a9c772580cc3f1 | 146 |
| Deliverables.docx | original | 5cc83562dedf54f1a82c30fc024654c07278a314856ef4f3b35322feaa865e26 | 147 |
| Assessment and Plagiarism Policy.docx | original | 023de0bdfadc5d7bb2faa598b79cfe38c0c660ee4ce9d5bf9b9fa258042b2b53 | 61 |
| PathSnap - Complete Guidelines.docx | synthesis | 4ef3e3392a7c80c4725f80ed75dd106a695efcb53c5fe21cf1c85b4273d2e3d1 | 480 |

## Verification method

Each DOCX is a ZIP container. The audit parsed word/document.xml with Python zipfile and ElementTree, walked the W namespace recursively (body, then p / tbl / tr / tc, including paragraphs nested inside tables), and assembled text from w:t, w:tab and w:br nodes in document order. The result was compared paragraph by paragraph against the existing markdown extracts in analysis/instructions_review/*__xml_text.md, after normalisation: NFKC, whitespace stripping, zero-width character removal, and semicolon-to-pipe folding for the table-row appendix lines. Continuation lines in the extracts were grouped under their P<n> headers, the "# " title lines and the "## All XML table rows" appendix were excluded, and only body paragraph text was compared.

## Results

All six documents matched their existing extracts exactly after normalisation (pairwise_norm_match true for every file, first mismatch none). The five originals concatenated in document order reproduce the synthesis file exactly: 480 of 480 non-empty paragraphs identical, in the same order. The synthesis file therefore adds no requirement beyond the five originals, and citations in the matrix are pinned to the originals.

Locators: P<n> counts non-empty XML paragraphs in document order, matching the numbering in the extracts. Nested-table rows carry path markers like [tc/tr/tbl/body/document]. The matrix quotes the original wording only where exact phrasing matters, for example "at leastFIVE different models", which reproduces the source's own missing space.

## Extracts and their relation to the originals

The reads behind the matrix came from the markdown extracts, which this audit re-derived independently from the binaries and found identical. Where an earlier extract set (analysis/instructions/*.md) had omitted nested-table rows, the set used here includes them; that gap does not affect this audit.

## Images

The DOCX files contain three unique images: two event logos and one Folium map screenshot. The prior tesseract OCR review (analysis/instructions_review/IMAGE_OCR_REVIEW.md) found no instructions hidden in them. This agent did not visually inspect the images and does not support image input, so no claim is made beyond that OCR record.

## Access limits held during the audit

evaluator_test/ was never opened. No data archive was opened. No network request was made. No model was fitted, run or selected. The ten human-reported evaluator trips remain deferred until after the final model and decoder freeze, per the user's standing order.

## Outputs of this audit

- analysis/final_requirements/requirements_matrix.json: 47 requirement rows with source locators, classification, acceptance evidence and status.
- analysis/final_requirements/provenance_summary.md: this file.
- docs/FINAL_REQUIREMENTS.md: the human-readable report.

The organizer documents are private event material. This audit quotes them sparingly, at locator precision, and does not republish their full contents.

## AI provenance

This audit was produced by an AI agent. It claims no human inspection of the sources and no human verification of the artifacts referenced in the status column; those statuses rely on the workspace records named in each row, which the Lead should confirm before the final build.
