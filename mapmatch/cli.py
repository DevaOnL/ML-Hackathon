"""Reproducible command-line prediction, explicit scoring and integrity checks."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from .data import discover_trips, load_graph, load_route, load_track, validate_route, write_route
from .matcher import MapMatcher, MatchConfig, MatchingError
from .metrics import score_routes

PROJECT = Path(__file__).resolve().parents[1]


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _config(path: str | None, *, require_frozen: bool = False) -> MatchConfig:
    if path is None:
        default = PROJECT / "artifacts" / "phase_three" / "match_config.json"
        if require_frozen and not default.is_file():
            raise FileNotFoundError("frozen requested-trio decoder configuration is missing; pass an explicit --config or complete decoder calibration")
        path = str(default) if default.is_file() else None
    return MatchConfig(**json.loads(Path(path).read_text(encoding="utf-8"))) if path else MatchConfig()


def _trips(args) -> list[Path]:
    prefixes = discover_trips(args.input)
    if args.trip:
        wanted = set(args.trip)
        available = {p.name for p in prefixes}
        if wanted - available:
            raise ValueError("unknown trip IDs: " + ", ".join(sorted(wanted - available)))
        prefixes = [p for p in prefixes if p.name in wanted]
    return prefixes


def _archive(output: Path, paths: list[Path]) -> None:
    """Flat route archive with stable filenames, bytes, timestamps and ordering."""
    with ZipFile(output / "submission.zip", "w", compression=ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            info = ZipInfo(path.name, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def predict(args) -> int:
    prefixes = _trips(args)
    output = Path(args.output).resolve()
    input_root = Path(args.input).resolve()
    if output == input_root or output.is_relative_to(input_root):
        raise ValueError("output directory must be outside the raw input directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; use a new directory to prevent stale results")
    config = _config(args.config, require_frozen=not args.baseline)
    emission_model = None
    model_hash = None
    if not args.baseline:
        from .learning import load_emission_model
        model_path = Path(args.model)
        emission_model = load_emission_model(model_path)
        model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    _json(output / "run_config.json", {"config": asdict(config), "model_sha256": model_hash, "baseline": args.baseline, "uses_reference_routes": False, "trip_ids": [p.name for p in prefixes]})
    index = []
    routes = []
    failures = []
    for prefix in prefixes:
        start = time.perf_counter()
        try:
            graph = load_graph(prefix)
            track = load_track(prefix.with_suffix(".track"))
            result = MapMatcher(graph, config, emission_model=emission_model).match(track)
            route_path = output / "routes" / (prefix.name + ".route")
            write_route(route_path, result.route)
            routes.append(route_path)
            _json(output / "diagnostics" / (prefix.name + ".json"), result.diagnostics)
            observations = output / "observations" / (prefix.name + ".csv")
            observations.parent.mkdir(parents=True, exist_ok=True)
            with observations.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["observation_index", "timestamp_s", "predicted_arc_id", "fraction", "lateral_distance_m", "classifier_only_arc_id"])
                classifier = result.diagnostics.get("classifier_only_edges")
                for j, (i, edge, fraction, distance) in enumerate(zip(result.observation_indices, result.matched_edges, result.matched_fractions, result.matched_distances)):
                    writer.writerow([int(i), float(track[i, 2]), int(edge), float(fraction), float(distance), classifier[j] if classifier is not None else ""])
            row = {"trip_id": prefix.name, "status": "ok", "route_file": str(route_path.relative_to(output)), "arc_count": len(result.route), "length_m": result.diagnostics["route_length_m"], "elapsed_s": time.perf_counter() - start, "sha256": hashlib.sha256(route_path.read_bytes()).hexdigest()}
            index.append(row)
            print(json.dumps(row), flush=True)
        except (ValueError, OSError, MatchingError) as exc:
            failure = {"trip_id": prefix.name, "status": "failed", "error": str(exc)}
            failures.append(failure)
            print(json.dumps(failure), file=sys.stderr, flush=True)
    with (output / "index.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["trip_id", "status", "route_file", "arc_count", "length_m", "elapsed_s", "sha256"])
        writer.writeheader()
        writer.writerows(index)
    _json(output / "failures.json", failures)
    if failures:
        print("No submission archive created: one or more trips failed.", file=sys.stderr)
        return 1
    _archive(output, routes)
    return 0


def _safe_output_file(output: str, *protected_roots: str) -> Path:
    target = Path(output).resolve()
    for root in protected_roots:
        protected = Path(root).resolve()
        if target == protected or target.is_relative_to(protected):
            raise ValueError("output file must be outside raw input and prediction directories")
    if target.exists():
        raise ValueError("output file already exists; use a new path to avoid overwriting evidence")
    return target


def evaluate(args) -> int:
    output = _safe_output_file(args.output, args.input, args.predictions)
    prefixes = _trips(args)
    if any("evaluator" in part.lower() or part.lower() == "private" for part in Path(args.input).parts) and not args.final_evaluation:
        raise ValueError("reserved evaluator scoring requires --final-evaluation and must not inform development")
    rows = []
    for prefix in prefixes:
        prediction = Path(args.predictions) / "routes" / (prefix.name + ".route")
        if not prediction.is_file():
            prediction = Path(args.predictions) / (prefix.name + ".route")
        graph = load_graph(prefix)
        score = score_routes(graph, load_route(prediction), load_route(prefix.with_suffix(".route")))
        rows.append({"trip_id": prefix.name, **score})
    numeric = ["edge_precision", "edge_recall", "edge_f1", "length_precision", "length_recall", "length_f1", "duplicate_canonical_length_f1", "edit_similarity", "route_continuity"]
    summary = {"trip_count": len(rows), "macro": {key: float(np.mean([r[key] for r in rows])) for key in numeric}, "exact_route_count": sum(r["exact_match"] for r in rows), "connected_prediction_count": sum(r["prediction_connected"] for r in rows), "reference_discontinuities": sum(r["reference_discontinuity_count"] for r in rows), "evaluation_only": True, "official_score": False}
    _json(output, {"summary": summary, "trips": rows})
    print(json.dumps(summary, indent=2))
    return 0 if summary["connected_prediction_count"] == len(rows) else 1


def validate(args) -> int:
    output = _safe_output_file(args.output, args.input)
    rows = []
    failures = []
    for prefix in _trips(args):
        try:
            graph = load_graph(prefix)
            track = load_track(prefix.with_suffix(".track"))
            row = {"trip_id": prefix.name, "nodes": len(graph.nodes), "arcs": len(graph.arcs), "observations": len(track), "timestamp_duration_s": float(track[-1, 2] - track[0, 2]), "valid": True}
            if args.references:
                row["reference"] = validate_route(graph, load_route(prefix.with_suffix(".route")))
                row["valid"] = row["reference"]["valid"]
                if not row["valid"]:
                    failures.append({"trip_id": prefix.name, "error": "invalid or topologically disconnected reference route"})
            rows.append(row)
        except (ValueError, OSError) as exc:
            failures.append({"trip_id": prefix.name, "error": str(exc)})
    _json(output, {"trips": rows, "failures": failures, "reference_labels_read": args.references})
    print(json.dumps({"valid_trip_count": len(rows), "failure_count": len(failures)}))
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="mapmatch", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict", help="fresh inference without opening any reference route")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--trip", action="append")
    p.add_argument("--model", default=str(PROJECT / "artifacts" / "phase_three" / "model.joblib"))
    p.add_argument("--config")
    p.add_argument("--baseline", action="store_true", help="explicit geometry-only diagnostic baseline (not final learned system)")
    p.set_defaults(function=predict)
    e = sub.add_parser("evaluate", help="explicit post-inference reference scoring")
    e.add_argument("--input", required=True)
    e.add_argument("--predictions", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--trip", action="append")
    e.add_argument("--final-evaluation", action="store_true")
    e.set_defaults(function=evaluate)
    v = sub.add_parser("validate", help="parse graph/track integrity, optionally references")
    v.add_argument("--input", required=True)
    v.add_argument("--output", required=True)
    v.add_argument("--trip", action="append")
    v.add_argument("--references", action="store_true")
    v.set_defaults(function=validate)
    args = parser.parse_args(argv)
    try:
        return args.function(args)
    except (ValueError, OSError, MatchingError) as exc:
        print(f"mapmatch: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
