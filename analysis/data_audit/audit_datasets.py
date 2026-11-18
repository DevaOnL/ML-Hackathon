#!/usr/bin/env python3
"""Complete read-only corpus audit; write outputs only to analysis/data_audit.
No learning/prediction. Test routes used for schema and walk integrity only.
"""
from __future__ import annotations
import argparse
import collections
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import zipfile
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

EXT = ("nodes", "arcs", "route", "track")
WIDTH = dict(zip(EXT, (2, 2, 1, 3)))
INT = re.compile(r"[+-]?[0-9]+\Z")
FLOAT = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
R = 6371008.8
Q = (0, .01, .05, .25, .5, .75, .9, .95, .99, 1)


def native(v):
    if isinstance(v, dict):
        return {str(k): native(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [native(x) for x in v]
    if isinstance(v, np.ndarray):
        return native(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def dist(v):
    v = np.asarray(v, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if not len(v):
        return {"count": 0}
    d = {"count": len(v), "sum": float(v.sum()), "mean": float(v.mean()), "std": float(v.std())}
    d.update({f"p{q*100:g}": float(x) for q, x in zip(Q, np.quantile(v, Q))})
    return native(d)


def hav(a, b):
    a, b = np.radians(np.asarray(a)), np.radians(np.asarray(b))
    d = b-a
    h = np.sin(d[..., 1]/2)**2 + np.cos(a[..., 1])*np.cos(b[..., 1])*np.sin(d[..., 0]/2)**2
    return 2*R*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def project(p, lon0, lat0):
    p = np.asarray(p, dtype=float)
    return np.column_stack((np.radians(p[:, 0]-lon0)*R*math.cos(math.radians(lat0)),
                            np.radians(p[:, 1]-lat0)*R))


def nearest_segments(points, start, end):
    """Exact exhaustive segment comparisons in local projection, bounded memory."""
    best = np.full(len(points), np.inf)
    idx_out = np.full(len(points), -1, dtype=int)
    frac_out = np.zeros(len(points))
    if not len(start):
        return best, idx_out, frac_out
    chunk = max(1, min(256, 1_500_000 // max(1, len(start))))
    vec = end-start
    vv = np.einsum("ij,ij->i", vec, vec)
    denom = np.where(vv > 0, vv, 1)
    for lo in range(0, len(points), chunk):
        p = points[lo:lo+chunk]
        w = p[:, None, :]-start[None, :, :]
        f = np.clip(np.einsum("pij,ij->pi", w, vec)/denom[None, :], 0, 1)
        residual = w-f[:, :, None]*vec[None, :, :]
        d2 = np.einsum("pij,pij->pi", residual, residual)
        ix = d2.argmin(axis=1)
        row = np.arange(len(p))
        best[lo:lo+len(p)] = np.sqrt(d2[row, ix])
        idx_out[lo:lo+len(p)] = ix
        frac_out[lo:lo+len(p)] = f[row, ix]
    return best, idx_out, frac_out


def parse_file(p, root, zip_hashes):
    raw = p.read_bytes()  # complete byte read; every physical line parsed below
    digest = hashlib.sha256(raw).hexdigest()
    ext = p.suffix[1:]
    rel = str(p.relative_to(root.parent))
    candidates = [z for z in zip_hashes if z == rel or z.endswith("/"+rel)]
    entry = candidates[0] if len(candidates) == 1 else None
    m = {"path": str(p.resolve()), "relative_path": rel, "split": p.parent.parent.name,
         "trip_id": p.stem, "extension": ext, "bytes": len(raw), "sha256": digest,
         "read_complete": True, "zip_entry": entry,
         "zip_sha256_match": zip_hashes[entry] == digest if entry else None,
         "bom": raw.startswith(b"\xef\xbb\xbf"), "crlf_count": raw.count(b"\r\n"),
         "lf_count": raw.count(b"\n"), "terminal_lf": raw.endswith(b"\n"), "nul_count": raw.count(b"\x00")}
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as e:
        m.update(row_count=None, data_row_count=0, parsing_status="invalid_utf8", parse_errors=[str(e)])
        return m, None
    lines = text.splitlines()
    m.update(row_count=len(lines), blank_rows=sum(not x.strip() for x in lines),
             duplicate_text_rows=len(lines)-len(set(lines)),
             whitespace_field_counts=dict(collections.Counter(len(x.split()) for x in lines)),
             tab_field_counts=dict(collections.Counter(len(x.split("\t")) for x in lines)),
             non_ascii_characters=sum(ord(x) > 127 for x in text))
    lex = INT if ext in ("arcs", "route") else FLOAT
    errors, data, precision = [], [], collections.Counter()
    for number, line in enumerate(lines, 1):
        cells = line.split()
        if len(cells) != WIDTH[ext]:
            errors.append({"line": number, "error": "wrong_field_count", "count": len(cells)})
            continue
        if any(not lex.fullmatch(x) for x in cells):
            errors.append({"line": number, "error": "invalid_numeric_lexeme"})
            continue
        try:
            row = [int(x) for x in cells] if ext in ("arcs", "route") else [float(x) for x in cells]
            if any(not math.isfinite(x) for x in row):
                errors.append({"line": number, "error": "non_finite_numeric"})
                continue
            data.append(row)
            if ext in ("nodes", "track"):
                for cell in cells:
                    precision[len(cell.rsplit(".", 1)[1]) if "." in cell else 0] += 1
        except (ValueError, OverflowError):
            errors.append({"line": number, "error": "numeric_conversion"})
    a = np.asarray(data, dtype=np.int64 if ext in ("arcs", "route") else np.float64).reshape((-1, WIDTH[ext]))
    m.update(data_row_count=len(a), fractional_digit_counts=dict(precision), parse_error_count=len(errors),
             parse_errors=errors[:30], parsing_status="ok" if not errors and len(a) and not m["bom"] and not m["nul_count"] else "invalid_or_empty")
    return m, a


def bounds(a):
    return dict(longitude_min=float(a[:, 0].min()), longitude_max=float(a[:, 0].max()),
                latitude_min=float(a[:, 1].min()), latitude_max=float(a[:, 1].max()))


def audit_trip(files, arrays):
    fm = {x["extension"]: x for x in files}
    split, trip = files[0]["split"], files[0]["trip_id"]
    result = {"split": split, "trip_id": trip, "complete_four_file_set": set(fm) == set(EXT),
              "all_files_parse_ok": all(x["parsing_status"] == "ok" for x in files),
              "test_route_usage": "schema/range/directed-walk validity only; no track-label residuals, learning or prediction" if split == "student_test" else None}
    if not result["complete_four_file_set"] or not result["all_files_parse_ok"]:
        return result, {}
    n, a, r, t = [arrays[x] for x in EXT]
    r = r.ravel()
    N, A = len(n), len(a)
    _, pc = np.unique(n, axis=0, return_counts=True)
    result["nodes"] = {"count": N, "bounds_degrees": bounds(n),
        "invalid_geographic_rows": int(((np.abs(n[:, 0]) > 180) | (np.abs(n[:, 1]) > 90)).sum()),
        "duplicate_coordinate_extra_nodes": int((pc-1).sum()), "duplicate_coordinate_groups": int((pc > 1).sum()),
        "largest_coordinate_group": int(pc.max())}
    av = (a >= 0).all(axis=1) & (a < N).all(axis=1)
    good = a[av]
    ua, ac = np.unique(good, axis=0, return_counts=True)
    enc, rev = ua[:, 0]*N+ua[:, 1], ua[:, 1]*N+ua[:, 0]
    reverse_exists = np.isin(rev, enc)
    indeg, outdeg = np.bincount(good[:, 1], minlength=N), np.bincount(good[:, 0], minlength=N)
    graph = csr_matrix((np.ones(len(ua), dtype=np.int8), (ua[:, 0], ua[:, 1])), shape=(N, N))
    nw, weak = connected_components(graph, directed=True, connection="weak")
    ns, strong = connected_components(graph, directed=True, connection="strong")
    ws, ss = np.bincount(weak), np.bincount(strong)
    lengths = hav(n[good[:, 0]], n[good[:, 1]])
    result["arcs"] = {"count": A, "node_id_min": int(a.min()), "node_id_max": int(a.max()),
        "invalid_endpoint_rows": int((~av).sum()), "self_loops": int((good[:, 0] == good[:, 1]).sum()),
        "unique_directed_pairs": len(ua), "duplicate_directed_pair_extra_arcs": int((ac-1).sum()),
        "duplicate_directed_pair_groups": int((ac > 1).sum()), "unique_pairs_with_reverse": int(reverse_exists.sum()),
        "unique_pairs_without_reverse": int((~reverse_exists).sum()),
        "reciprocal_directed_pair_fraction": float(reverse_exists.mean()),
        "isolated_nodes": int(((indeg+outdeg) == 0).sum()), "nodes_with_zero_indegree": int((indeg == 0).sum()),
        "nodes_with_zero_outdegree": int((outdeg == 0).sum()),
        "nodes_with_edges_but_zero_indegree": int(((indeg == 0) & (outdeg > 0)).sum()),
        "nodes_with_edges_but_zero_outdegree": int(((outdeg == 0) & (indeg > 0)).sum()),
        "indegree": dist(indeg), "outdegree": dist(outdeg), "weak_components": int(nw),
        "weak_component_sizes_desc": sorted(ws.tolist(), reverse=True), "strong_components": int(ns),
        "strong_component_sizes_top10": sorted(ss.tolist(), reverse=True)[:10],
        "largest_weak_fraction": float(ws.max()/N), "largest_strong_fraction": float(ss.max()/N),
        "endpoint_chord_lengths_m": dist(lengths), "zero_chord_length_arcs": int((lengths == 0).sum()),
        "chord_length_over_1km": int((lengths > 1000).sum()), "chord_length_over_10km": int((lengths > 10000).sum())}
    rv, r1v = (r >= 0) & (r < A), (r >= 1) & (r <= A)
    dup = int((r[:-1] == r[1:]).sum())
    unique_r, rc = np.unique(r, return_counts=True)
    rs = {"count": len(r), "id_min": int(r.min()), "id_max": int(r.max()),
          "invalid_zero_based_arc_ids": int((~rv).sum()), "invalid_one_based_arc_ids": int((~r1v).sum()),
          "unique_arc_ids": len(unique_r), "repeated_arc_extra_occurrences": int((rc-1).sum()),
          "repeated_arc_id_count": int((rc > 1).sum()), "consecutive_duplicate_arc_ids": dup,
          "nonconsecutive_repeat_extra_occurrences": int((rc-1).sum())-dup,
          "ids_outside_zero_based_node_range": int(((r < 0) | (r >= N)).sum())}
    for offset in (0, 1):
        rr = r-offset
        valid = (rr >= 0) & (rr < A)
        pv = valid[:-1] & valid[1:]
        prev, nxt = a[rr[:-1][pv]], a[rr[1:][pv]]
        join = prev[:, 1] == nxt[:, 0]
        rs[f"arc_index_offset_{offset}"] = {"valid_rows": int(valid.sum()), "join_comparisons": len(join),
            "directed_joins": int(join.sum()), "broken_joins": int((~join).sum()),
            "broken_join_after_route_line_examples": (np.where(pv)[0][~join]+1)[:20].tolist()}
        valid_node = (rr >= 0) & (rr < N)
        adjacent = valid_node[:-1] & valid_node[1:]
        rs[f"node_index_offset_{offset}"] = {"valid_rows": int(valid_node.sum()),
            "join_comparisons": int(adjacent.sum()),
            "existing_directed_edges": int(np.isin(rr[:-1][adjacent]*N+rr[1:][adjacent], enc).sum())}
    path = a[r] if rv.all() else None
    route_lengths = None
    if path is not None and ((path >= 0) & (path < N)).all():
        route_lengths = hav(n[path[:, 0]], n[path[:, 1]])
        node_path = np.r_[path[0, 0], path[:, 1]]
        rs.update(directed_walk_is_continuous=bool((path[:-1, 1] == path[1:, 0]).all()),
                  path_endpoint_chord_length_m=float(route_lengths.sum()), path_arc_chord_lengths_m=dist(route_lengths),
                  weak_components_touched=int(len(np.unique(weak[node_path]))),
                  strong_components_touched=int(len(np.unique(strong[node_path]))),
                  repeated_nodes_extra_occurrences=int(len(node_path)-len(np.unique(node_path))),
                  zero_chord_length_arc_occurrences=int((route_lengths == 0).sum()))
    result["route_validity"] = rs
    coords, times = t[:, :2], t[:, 2]
    delta, step = np.diff(times), hav(coords[:-1], coords[1:])
    speed = np.divide(step, delta, out=np.full_like(step, np.nan), where=delta > 0)
    _, tc = np.unique(times, return_counts=True)
    nb = result["nodes"]["bounds_degrees"]
    outside = ((coords[:, 0] < nb["longitude_min"]) | (coords[:, 0] > nb["longitude_max"]) |
               (coords[:, 1] < nb["latitude_min"]) | (coords[:, 1] > nb["latitude_max"]))
    ts = {"count": len(t), "bounds_degrees": bounds(coords),
          "invalid_geographic_rows": int(((np.abs(coords[:, 0]) > 180) | (np.abs(coords[:, 1]) > 90)).sum()),
          "time_first": float(times[0]), "time_last": float(times[-1]), "time_min": float(times.min()),
          "time_max": float(times.max()), "time_duration": float(times[-1]-times[0]),
          "time_exactly_1_to_N": bool(np.array_equal(times, np.arange(1, len(t)+1))),
          "noninteger_timestamps": int((times != np.rint(times)).sum()),
          "duplicate_timestamp_extra_rows": int((tc-1).sum()), "nonpositive_time_deltas": int((delta <= 0).sum()),
          "time_delta_not_one": int((delta != 1).sum()), "time_delta": dist(delta),
          "duplicate_coordinate_extra_rows": int(len(coords)-len(np.unique(coords, axis=0))),
          "consecutive_duplicate_coordinates": int((step == 0).sum()),
          "points_outside_node_bbox": int(outside.sum()), "step_lengths_m": dist(step),
          "step_speed_mps_if_time_unit_seconds": dist(speed), "steps_over_50mps_if_seconds": int((speed > 50).sum()),
          "steps_over_100mps_if_seconds": int((speed > 100).sum()), "total_point_to_point_distance_m": float(step.sum())}
    lon0, lat0 = float(np.median(n[:, 0])), float(np.median(n[:, 1]))
    pxy, nxy = project(coords, lon0, lat0), project(n, lon0, lat0)
    nd, ni = cKDTree(nxy).query(pxy, workers=1)
    ts.update(nearest_graph_node_distance_m_local_projection=dist(nd),
              nearest_node_weak_components_count=int(len(np.unique(weak[ni]))),
              projection_reference_lon_lat_degrees=[lon0, lat0])
    residual = np.empty(0)
    if split == "train" and path is not None and route_lengths is not None:
        residual, si, frac = nearest_segments(pxy, nxy[path[:, 0]], nxy[path[:, 1]])
        rs["train_only_track_to_route_segment_distance_m_local_projection"] = dist(residual)
        for cutoff in (20, 50, 100):
            rs[f"train_only_track_points_over_{cutoff}m_from_route_chords"] = int((residual > cutoff).sum())
        rs["train_only_first_point_to_first_route_segment_m"] = float(nearest_segments(pxy[:1], nxy[path[:1, 0]], nxy[path[:1, 1]])[0][0])
        rs["train_only_last_point_to_last_route_segment_m"] = float(nearest_segments(pxy[-1:], nxy[path[-1:, 0]], nxy[path[-1:, 1]])[0][0])
        rs["train_only_first_point_to_route_start_node_m"] = float(hav(coords[0], n[path[0, 0]]))
        rs["train_only_last_point_to_route_end_node_m"] = float(hav(coords[-1], n[path[-1, 1]]))
        rs["train_only_first_point_nearest_route_segment_index"] = int(si[0])
        rs["train_only_last_point_nearest_route_segment_index"] = int(si[-1])
        rs["train_only_route_chord_distance_to_track_step_distance_ratio"] = float(route_lengths.sum()/step.sum()) if step.sum() > 0 else None
    result["track"] = ts
    pool = dict(arc_chord_lengths_m=lengths, track_step_lengths_m=step, track_time_deltas=delta,
                track_speed_mps_if_seconds=speed, nearest_node_distance_m=nd, train_only_route_residual_m=residual)
    return native(result), pool


def flatten(x):
    n, a, r, t = [x.get(k, {}) for k in ("nodes", "arcs", "route_validity", "track")]
    d = {"split": x["split"], "trip_id": x["trip_id"], "parse_ok": x["all_files_parse_ok"],
         "nodes": n.get("count"), "arcs": a.get("count"), "route_arcs": r.get("count"), "track_points": t.get("count")}
    d.update(n.get("bounds_degrees", {}))
    for k in ("weak_components", "strong_components", "isolated_nodes", "unique_pairs_without_reverse", "invalid_endpoint_rows", "duplicate_directed_pair_extra_arcs", "zero_chord_length_arcs"):
        d[k] = a.get(k)
    d.update(duplicate_coordinate_extra_nodes=n.get("duplicate_coordinate_extra_nodes"),
             invalid_route_rows=r.get("invalid_zero_based_arc_ids"),
             route_broken_joins=r.get("arc_index_offset_0", {}).get("broken_joins"),
             route_duplicate_extra_arcs=r.get("repeated_arc_extra_occurrences"),
             route_length_m=r.get("path_endpoint_chord_length_m"), track_length_m=t.get("total_point_to_point_distance_m"),
             time_duration=t.get("time_duration"), timestamps_1_to_N=t.get("time_exactly_1_to_N"),
             track_outside_bbox=t.get("points_outside_node_bbox"),
             track_max_speed_mps_if_seconds=t.get("step_speed_mps_if_time_unit_seconds", {}).get("p100"))
    for label, q in (("median", "p50"), ("p95", "p95"), ("max", "p100")):
        d[f"train_only_route_residual_{label}_m"] = r.get("train_only_track_to_route_segment_distance_m_local_projection", {}).get(q)
    return d


def write_json(path, x):
    path.write_text(json.dumps(native(x), indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def write_csv(path, rows):
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()} for row in rows)


def summary(manifest, trips, pools, archive):
    s = {"created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
         "method": "Every file read completely as bytes, SHA256 hashed, UTF8 decoded, every line numerically/structurally validated, then cross-file integrity and geometry audited.",
         "test_label_policy": "Test routes used only for schema/range/duplicate/directed-walk validity, never track-label residuals, learning, tuning, prediction or scores.",
         "software": dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
         "archive": {k: v for k, v in archive.items() if k != "entries"},
         "file_count": len(manifest), "trip_count": len(trips),
         "bytes_read_from_extracted_files": sum(m["bytes"] for m in manifest),
         "all_360_read_complete": len(manifest) == 360 and all(m["read_complete"] for m in manifest),
         "all_files_parse_ok": all(m["parsing_status"] == "ok" for m in manifest),
         "all_extracted_files_match_zip": all(m["zip_sha256_match"] is True for m in manifest),
         "files_by_extension": dict(collections.Counter(m["extension"] for m in manifest)),
         "row_totals_by_extension": {ext: sum(m["data_row_count"] for m in manifest if m["extension"] == ext) for ext in EXT},
         "newline_and_encoding": dict(all_ascii=all(not m.get("non_ascii_characters") for m in manifest),
               all_terminal_lf=all(m["terminal_lf"] for m in manifest), all_no_bom=all(not m["bom"] for m in manifest),
               total_blank_rows=sum(m.get("blank_rows", 0) for m in manifest),
               total_crlf=sum(m["crlf_count"] for m in manifest), total_nul=sum(m["nul_count"] for m in manifest)),
         "splits": {}}
    checks = {"nodes": ("invalid_geographic_rows", "duplicate_coordinate_extra_nodes"),
              "arcs": ("invalid_endpoint_rows", "duplicate_directed_pair_extra_arcs", "self_loops", "zero_chord_length_arcs", "isolated_nodes", "unique_pairs_without_reverse", "unique_directed_pairs", "nodes_with_zero_indegree", "nodes_with_zero_outdegree"),
              "route_validity": ("invalid_zero_based_arc_ids", "repeated_arc_extra_occurrences", "consecutive_duplicate_arc_ids"),
              "track": ("invalid_geographic_rows", "duplicate_coordinate_extra_rows", "nonpositive_time_deltas", "duplicate_timestamp_extra_rows", "time_delta_not_one", "points_outside_node_bbox", "steps_over_50mps_if_seconds", "steps_over_100mps_if_seconds")}
    for split in sorted(set(x["split"] for x in trips)):
        tt, ff = [t for t in trips if t["split"] == split], [m for m in manifest if m["split"] == split]
        d = dict(trip_count=len(tt), trip_ids=[t["trip_id"] for t in tt], file_count=len(ff), bytes=sum(m["bytes"] for m in ff),
                 row_totals={ext: sum(m["data_row_count"] for m in ff if m["extension"] == ext) for ext in EXT})
        for k in ("nodes", "arcs", "route_arcs", "track_points", "weak_components", "strong_components", "route_length_m", "track_length_m", "time_duration"):
            d[f"per_trip_{k}"] = dist([flatten(t)[k] for t in tt])
        for section, keys in checks.items():
            for k in keys:
                d[f"{section}.{k}"] = sum(t[section][k] for t in tt)
        for offset in (0, 1):
            for k in ("join_comparisons", "directed_joins", "broken_joins"):
                d[f"route_arc_offset_{offset}.{k}"] = sum(t["route_validity"][f"arc_index_offset_{offset}"][k] for t in tt)
        d["all_track_timestamps_1_to_N"] = all(t["track"]["time_exactly_1_to_N"] for t in tt)
        d["trips_with_disconnected_weak_graph"] = [t["trip_id"] for t in tt if t["arcs"]["weak_components"] > 1]
        d["trips_with_route_repeated_arcs"] = [t["trip_id"] for t in tt if t["route_validity"]["repeated_arc_extra_occurrences"] > 0]
        d["trips_with_route_more_than_one_strong_component"] = [t["trip_id"] for t in tt if t["route_validity"]["strong_components_touched"] > 1]
        d["pooled_row_weighted_distributions"] = {k: dist(np.concatenate(v)) for k, v in pools[split].items() if v and any(len(z) for z in v)}
        s["splits"][split] = d
    hashes = collections.defaultdict(list)
    for m in manifest:
        hashes[m["sha256"]].append(m["relative_path"])
    s["byte_identical_file_groups"] = [v for v in hashes.values() if len(v) > 1]
    return native(s)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("Student folder"))
    p.add_argument("--archive", type=Path, default=Path("Student folder.zip"))
    args = p.parse_args()
    root, archive = args.root.resolve(), args.archive.resolve()
    out = Path(__file__).resolve().parent
    if out.name != "data_audit" or out.parent.name != "analysis":
        raise SystemExit("Output script must remain under analysis/data_audit")
    paths = sorted(x for x in root.rglob("*") if x.is_file())
    unexpected = [str(x) for x in paths if x.suffix[1:] not in EXT]
    if unexpected:
        raise SystemExit(f"Unexpected files, not silently skipped: {unexpected}")
    zip_hashes, entries = {}, []
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if not info.is_dir():
                raw = z.read(info)
                h = hashlib.sha256(raw).hexdigest()
                zip_hashes[info.filename] = h
                entries.append(dict(path=info.filename, bytes=len(raw), sha256=h, crc32=f"{info.CRC:08x}"))
    archive_info = dict(path=str(archive), bytes=archive.stat().st_size, sha256=digest,
                        regular_entries=len(entries), entries=entries)
    grouped = collections.defaultdict(list)
    for path in paths:
        grouped[(path.parent.parent.name, path.stem)].append(path)
    manifests, trips, pools = [], [], collections.defaultdict(lambda: collections.defaultdict(list))
    for (split, trip), members in sorted(grouped.items()):
        fm, arrays = [], {}
        for path in members:
            m, arr = parse_file(path, root, zip_hashes)
            manifests.append(m)
            fm.append(m)
            arrays[path.suffix[1:]] = arr
        audited, pool = audit_trip(fm, arrays)
        trips.append(audited)
        for key, val in pool.items():
            pools[split][key].append(val)
        print("audited", split+"/"+trip, {m["extension"]: m["data_row_count"] for m in fm}, flush=True)
    s = summary(manifests, trips, pools, archive_info)
    write_json(out/"manifest.json", manifests)
    write_csv(out/"manifest.csv", manifests)
    write_json(out/"per_trip.json", trips)
    write_csv(out/"per_trip.csv", [flatten(x) for x in trips])
    write_json(out/"summary.json", s)
    write_json(out/"archive_manifest.json", archive_info)
    assert len(manifests) == 360, len(manifests)
    assert len(trips) == 90, len(trips)
    assert all(m["read_complete"] and m["parsing_status"] == "ok" and m["zip_sha256_match"] is True for m in manifests)
    assert all(t["complete_four_file_set"] for t in trips)
    assert all(t["route_validity"]["invalid_zero_based_arc_ids"] == 0 for t in trips)
    assert all(t["route_validity"]["arc_index_offset_0"]["broken_joins"] == 0 for t in trips)
    print("SUCCESS: all 360 complete reads, strict parses, ZIP hashes, route walk gates passed.")
    print(json.dumps(dict(files=s["file_count"], trips=s["trip_count"], rows=s["row_totals_by_extension"])))


if __name__ == "__main__":
    main()
