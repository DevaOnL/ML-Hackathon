"""Derive graph-qualified observation metrics and raw classifier continuity.

Run AFTER mapmatch.experiments finishes, before declaring final artifact frozen.
No estimator is refitted and no inference uses reference route files. Targets
are the proxy arc IDs already recorded during the permitted experiment.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from mapmatch.data import discover_trips, load_graph
from mapmatch.features import FEATURE_DESCRIPTIONS, FEATURE_NAMES, FEATURE_VERSION


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def observation_metrics(prediction, graphs=None):
    p, offsets = prediction["probabilities"], prediction["offsets"]
    y, trips = prediction["y"], prediction["observation_trip_ids"]
    reference, candidates = prediction["reference_edges"], prediction["candidate_edges"]
    indices = prediction["observation_indices"]
    chosen, present, correct, target_probability, max_probability = [], [], [], [], []
    for lo, hi in zip(offsets[:-1], offsets[1:]):
        available = bool(np.any(y[lo:hi]))
        if hi > lo:
            local = np.clip(p[lo:hi], 1e-7, 1 - 1e-7)
            normalized = local / local.sum()
            row = int(np.argmax(normalized))
            chosen.append(int(candidates[lo + row]))
            correct.append(bool(y[lo + row]))
            max_probability.append(float(normalized[row]))
            target_probability.append(float(normalized[y[lo:hi] == 1].sum()) if available else 0.0)
        else:
            chosen.append(-1); correct.append(False); max_probability.append(0.0); target_probability.append(0.0)
        present.append(available)
    chosen, present, correct = np.asarray(chosen), np.asarray(present), np.asarray(correct)
    target_probability = np.asarray(target_probability)
    true_classes = np.array([str(trip) + ":" + str(int(edge)) for trip, edge in zip(trips, reference)])
    predicted_classes = np.array([str(trip) + ":" + str(int(edge)) for trip, edge in zip(trips, chosen)])
    metrics = {
        "observation_edge_macro_f1": float(f1_score(true_classes, predicted_classes, average="macro", zero_division=0)),
        "observation_edge_macro_precision": float(precision_score(true_classes, predicted_classes, average="macro", zero_division=0)),
        "observation_edge_macro_recall": float(recall_score(true_classes, predicted_classes, average="macro", zero_division=0)),
        "edge_class_namespace": "trip_id:canonical_directed_original_arc_id; class union includes reference-only and prediction-only roads",
        "observation_categorical_log_loss_clipped": float(-np.mean(np.log(np.maximum(target_probability, 1e-7)))),
        "observation_categorical_log_loss_given_candidate": float(-np.mean(np.log(np.maximum(target_probability[present], 1e-7)))) if np.any(present) else None,
        "categorical_log_loss_policy": "Normalize binary positive probabilities within each observation; absent target receives zero mass and finite log score uses floor1e-7. This is an emission log score, not a calibration claim. Misses are included in the unconditional score.",
        "observations": len(chosen), "candidate_miss_count": int((~present).sum()),
        "candidate_miss_rate": float((~present).mean()), "top1_accuracy": float(correct.mean()),
        "per_trip": [],
    }
    all_connected, all_pairs, all_same, all_edge_changes = 0, 0, 0, 0
    for trip in np.unique(trips):
        mask = trips == trip
        rows = np.flatnonzero(mask)
        # Recorded sample arrays should already be increasing; enforce this to
        # avoid calculating continuity in an arbitrary row order.
        rows = rows[np.argsort(indices[rows], kind="stable")]
        selected = chosen[rows]
        item = {"trip_id": str(trip), "observations": len(rows), "top1_accuracy": float(correct[rows].mean()), "candidate_miss_rate": float((~present[rows]).mean()), "observation_edge_macro_f1": float(f1_score(reference[rows], selected, average="macro", zero_division=0)), "observation_categorical_log_loss_clipped": float(-np.log(np.maximum(target_probability[rows], 1e-7)).mean())}
        if graphs is not None:
            graph = graphs[str(trip)]
            valid = selected >= 0
            canonical = np.full(len(selected), -1, dtype=np.int64)
            canonical[valid] = graph.canonical_edges[selected[valid]]
            previous, current = canonical[:-1], canonical[1:]
            both = (previous >= 0) & (current >= 0)
            same = both & (previous == current)
            connected = same.copy()
            connected[both] |= graph.heads[previous[both]] == graph.tails[current[both]]
            pairs = max(len(selected) - 1, 0)
            changes = both & (previous != current)
            change_connected = connected & changes
            item.update({"adjacent_observation_pairs": pairs, "directly_continuous_pairs": int(connected.sum()), "raw_classifier_continuity": float(connected.sum() / pairs) if pairs else None, "same_edge_pairs": int(same.sum()), "edge_change_pairs": int(changes.sum()), "directly_continuous_edge_change_pairs": int(change_connected.sum()), "edge_change_continuity": float(change_connected.sum() / changes.sum()) if changes.any() else None})
            all_connected += int(connected.sum()); all_pairs += pairs; all_same += int(same.sum()); all_edge_changes += int(changes.sum())
        metrics["per_trip"].append(item)
    metrics["trip_macro_observation_edge_f1"] = float(np.mean([v["observation_edge_macro_f1"] for v in metrics["per_trip"]]))
    if graphs is not None:
        metrics.update({"raw_classifier_continuity": all_connected / all_pairs if all_pairs else None, "directly_continuous_pairs": all_connected, "adjacent_observation_pairs": all_pairs, "same_edge_pairs": all_same, "edge_change_pairs": all_edge_changes, "edge_change_continuity": (all_connected - all_same) / all_edge_changes if all_edge_changes else None, "continuity_definition": "Adjacent available observation assignments are continuous iff same canonical edge OR head(previous)==tail(current); no routed connectors. Empty assignments count discontinuous. Includes observed time gaps; on downsampled CV this would not imply every intervening road was predicted."})
    sparse_confusion = Counter((str(t), int(r), int(c)) for t, r, c in zip(trips, reference, chosen))
    return metrics, sparse_confusion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("experiments"))
    parser.add_argument("--data-root", type=Path, default=Path("Student folder"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--no-update-artifact", action="store_true", help="Export evidence without changing an already frozen model bundle or its metadata")
    args = parser.parse_args(argv)
    output = args.output
    results_path = output / "results.json"
    results = json.loads(results_path.read_text())
    graphs = {p.name: load_graph(p) for p in discover_trips(args.data_root / "student_test")}
    supplement = {"target": "route-projection proxy, not independently supplied per-GPS ground truth", "binary_macro_f1_warning": "Candidate binary macro-F1 and graph-qualified observation edge macro-F1 answer different questions and must not share an unqualified label.", "student_test": {}, "train_out_of_fold": {}}
    confusion_path = output / "observation_confusion_sparse.csv"
    with confusion_path.open("w", newline="", encoding="utf-8") as confusion_stream:
        writer = csv.writer(confusion_stream); writer.writerow(["model", "split", "trip_id", "true_proxy_arc_id", "predicted_arc_id", "observation_count"])
        for name in results["evaluations"]:
            with np.load(output / "predictions" / f"{name}_student.npz", allow_pickle=False) as prediction:
                metrics, confusion = observation_metrics(prediction, graphs)
            supplement["student_test"][name] = metrics
            for (trip, truth, predicted), count in sorted(confusion.items()):
                writer.writerow([name, "student_test", trip, truth, predicted, count])
            results["evaluations"][name]["student_observation_metrics"] = metrics
            for row in results["comparison"]:
                if row["model"] == name:
                    row.update({"student_candidate_binary_macro_f1": row["student_macro_f1"], "student_observation_edge_macro_f1": metrics["observation_edge_macro_f1"], "student_trip_macro_observation_edge_f1": metrics["trip_macro_observation_edge_f1"], "student_observation_categorical_log_loss_clipped": metrics["observation_categorical_log_loss_clipped"], "route_continuity": metrics["raw_classifier_continuity"], "student_raw_classifier_continuity": metrics["raw_classifier_continuity"], "student_edge_change_continuity": metrics["edge_change_continuity"]})
        for name in results["cv"]:
            prediction_path = output / "predictions" / f"{name}_oof.npz"
            if not prediction_path.is_file():
                # HPO trials retained scalar/fold metrics but not all OOF arrays.
                supplement["train_out_of_fold"][name] = {"observation_edge_macro_f1": None, "reason": "tuned trial OOF probability array not retained; no unsupported metric is reconstructed from binary confusion counts"}
                continue
            with np.load(prediction_path, allow_pickle=False) as prediction:
                metrics, confusion = observation_metrics(prediction)
            supplement["train_out_of_fold"][name] = metrics
            results["cv"][name]["observation_metrics"] = metrics
            for row in results["comparison"]:
                if row["model"] == name:
                    row["cv_observation_edge_macro_f1"] = metrics["observation_edge_macro_f1"]
                    row["cv_observation_categorical_log_loss_clipped"] = metrics["observation_categorical_log_loss_clipped"]
    # Fill keys consistently for the tuned row without unavailable OOF arrays.
    for row in results["comparison"]:
        row.setdefault("cv_observation_edge_macro_f1", None)
        row.setdefault("cv_observation_categorical_log_loss_clipped", None)
    results["observation_metrics_supplement"] = "sequence_classification.json"
    results["route_continuity"] = "Actual raw classifier same-edge/direct-head-tail continuity is in comparison/evaluations; full routed HMM ablation is separate Lead output."
    save_json(output / "sequence_classification.json", supplement)
    save_json(results_path, results)
    with (output / "model_comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results["comparison"][0])); writer.writeheader(); writer.writerows(results["comparison"])
    # Recovery may freeze the prediction artifact before slow baseline refits.
    if args.no_update_artifact:
        model_path = args.artifacts / "model.joblib"
        print(json.dumps({"supplement": str(output / "sequence_classification.json"), "final_model": results["final_run_name"], "artifact_unchanged": True, "artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}))
        return
    # Add supplementary evaluated metrics to metadata BEFORE final freeze signal.
    model_path = args.artifacts / "model.joblib"
    bundle = joblib.load(model_path)
    final = results["final_run_name"]
    bundle["metadata"]["metrics"]["student_test_observation_metrics"] = supplement["student_test"][final]
    bundle["metadata"]["metrics"]["cv_observation_metrics"] = supplement["train_out_of_fold"][final]
    hardware_path = output / "hardware.json"
    if hardware_path.is_file():
        bundle["metadata"]["hardware"] = json.loads(hardware_path.read_text())
    replay_path = output / "final_oof_replay.json"
    if replay_path.is_file():
        bundle["metadata"]["final_oof_export"] = json.loads(replay_path.read_text())
    bundle["metadata"]["metric_definitions"] = {"candidate_binary_macro_f1": "Fixed two-class macro-F1 over candidate rows at0.5 threshold", "observation_edge_macro_f1": "Macro-F1 over union of trip-qualified canonical directed edge classes", "binary_log_loss": "Raw clipped binary positive probabilities over candidate rows", "observation_categorical_log_loss": "Normalized emission mass on proxy edge, floor1e-7 including misses", "raw_classifier_continuity": "same canonical edge OR head(previous)==tail(current), no connector; every adjacent student-test pair"}
    schema = {"version": FEATURE_VERSION, "dtype": "float32", "row_order": "observation order, then unchanged Candidates.edges/fractions/distances order; IDs persisted separately, never numerical features", "reference_route_used_as_features": False, "features": [{"index": i, "name": name, "description": FEATURE_DESCRIPTIONS[name]} for i, name in enumerate(FEATURE_NAMES)]}
    save_json(args.artifacts / "feature_schema.json", schema)
    joblib.dump(bundle, model_path, compress=3)
    save_json(args.artifacts / "model_metadata.json", bundle["metadata"])
    manifest_path = args.artifacts / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text()); manifest["sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest(); manifest["bytes"] = model_path.stat().st_size
    save_json(manifest_path, manifest)
    print(json.dumps({"supplement": str(output / "sequence_classification.json"), "final_model": final, "final_student_edge_macro_f1": supplement["student_test"][final]["observation_edge_macro_f1"], "final_student_classifier_continuity": supplement["student_test"][final]["raw_classifier_continuity"], "artifact_sha256": manifest["sha256"]}))


if __name__ == "__main__":
    main()
