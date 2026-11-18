# Final delivery guide

## Deliverables

The final submission consists of the consolidated five-family PDF, source and fitted weights/configuration in the repository, two separately labelled ten-route prediction ZIPs, actual experiment/trial logs, data/feature audit, model-comparison and route-ablation figures, a Folium implementation map, and deployed UI/API evidence. The old three-family PDF remains available as historical work. Do not submit it in place of the final five-family report.

The final primary comparison is exactly Logistic Regression, distance-weighted KNN, Bagging, GaussianNB and HistGradientBoosting. Document the separately preserved RandomForest/ExtraTrees historical experiments in an appendix rather than omit them or count them as current families.

## What must remain private

Obtain the organizer inputs through the authorized process. Do not add raw graph/GPS/reference files, DOCX documents, candidate caches, probability NPZs, aligned observations, private diagnostics or host credentials to the repository. Only the audited aggregate/hash-only evaluator reports and predicted route outputs are public. See the [publication policy](<FINAL_PUBLICATION_POLICY.md>).

## Demo procedure

1. Open [ml.d3va.dev](https://ml.d3va.dev), verify the model-ready badge and select Student development or Evaluator. The ten trips in each section remain separate.
2. Select a trip and reconstruct it. This sends a real POST containing only the selected trip ID. Explain the returned route, GPS observations and aggregate scores. Switch dark/light theme or use keyboard dataset navigation if needed.
3. On the receiving computer show the dedicated API journal filtered to the response request ID. A successful demo requires genuine matching model/configuration hashes and a completed request, not a typed example or a replayed screenshot.
4. If a reference is absent, explain N/A scores; do not substitute zero or route geometry as labels. Keep private references out of the browser and model.

## Human and organizer responsibilities

The user requested the public website in place of the brief’s physical two-laptop demonstration. The implementation also retains the two-laptop instructions if organizers require them. Physical venue setup cannot be certified remotely.

Every participating member must understand the data, proxy labels, feature extraction, five algorithms, TRAIN-only selection/HPO, decoder equations, metrics, reference isolation and deployment. Read the report and run the commands before the viva. AI assistance and GitHub account administration are disclosed; they do not establish independent human authorship or permission under an organizer’s policy. Organizers decide plagiarism clearance and evaluation scheduling. The source material’s before-lunch notice specifies no calendar date.
