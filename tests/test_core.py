"""Independent synthetic verification; no supplied test references are read."""
from __future__ import annotations

import builtins
import heapq
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np

from mapmatch.candidates import CandidateIndex
from mapmatch.data import (Graph, discover_trips, integer_array, load_graph,
                           load_route, load_track, validate_route, validate_track, write_route)
from mapmatch.matcher import MapMatcher, MatchConfig, MatchingError
from mapmatch.routing import Router


# Independent simple local conversion for synthetic fixtures, away from poles.
_METERS_PER_DEGREE = math.pi * 6_371_008.8 / 180.0


def metric_graph(points, arcs, latitude=0.0):
    points = np.asarray(points, dtype=float)
    scale = np.array([_METERS_PER_DEGREE * math.cos(math.radians(latitude)),
                      _METERS_PER_DEGREE])
    nodes = points / scale + [8.0, latitude]
    return Graph(nodes, np.asarray(arcs, dtype=np.int64))


def metric_track(points, times=None, latitude=0.0):
    points = np.asarray(points, dtype=float)
    if times is None:
        times = np.arange(1, len(points) + 1, dtype=float)
    scale = np.array([_METERS_PER_DEGREE * math.cos(math.radians(latitude)),
                      _METERS_PER_DEGREE])
    return np.column_stack((points / scale + [8.0, latitude], times))


def write_fixture(root, trip_id="00000001", *, points=None, arcs=None, samples=None):
    if points is None:
        points = [(0, 0), (100, 0), (200, 0)]
    if arcs is None:
        arcs = [(1, 2), (0, 1), (1, 0), (2, 1)]
    if samples is None:
        samples = [(10, 1), (50, -1), (150, 1), (190, 0)]
    graph = metric_graph(points, arcs)
    track = metric_track(samples)
    prefix = Path(root) / trip_id / trip_id
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(prefix.with_suffix(".nodes"), graph.nodes, fmt="%.12f", delimiter="\t")
    np.savetxt(prefix.with_suffix(".arcs"), graph.arcs, fmt="%d", delimiter="\t")
    np.savetxt(prefix.with_suffix(".track"), track, fmt="%.12f", delimiter="\t")
    return prefix


def exhaustive_segments(graph, point, radius):
    """Scalar point/segment oracle independent of the KD-tree implementation."""
    result = {}
    seen = set()
    for edge, (tail, head) in enumerate(graph.arcs):
        pair = int(tail), int(head)
        if pair in seen:
            continue
        seen.add(pair)
        a, b = graph.xy[tail], graph.xy[head]
        dx, dy = float(b[0] - a[0]), float(b[1] - a[1])
        denominator = dx * dx + dy * dy
        fraction = 0.0 if denominator == 0 else max(0.0, min(1.0,
            ((float(point[0]) - a[0]) * dx + (float(point[1]) - a[1]) * dy) / denominator))
        distance = math.hypot(a[0] + fraction * dx - point[0],
                              a[1] + fraction * dy - point[1])
        if distance <= radius + 1e-8:
            result[edge] = fraction, distance
    return result


def independent_distances(graph, source):
    adjacency = [[] for _ in graph.nodes]
    for edge, (tail, head) in enumerate(graph.arcs):
        adjacency[int(tail)].append((int(head), float(graph.edge_lengths[edge])))
    distances = [math.inf] * len(graph.nodes)
    distances[source] = 0.0
    queue = [(0.0, source)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for target, weight in adjacency[node]:
            proposed = distance + weight
            if proposed < distances[target]:
                distances[target] = proposed
                heapq.heappush(queue, (proposed, target))
    return distances


class DataContractTests(unittest.TestCase):
    def test_zero_based_original_arc_rows_are_not_node_ids(self):
        graph = metric_graph([(0, 0), (100, 0), (200, 0)],
                             [(2, 1), (1, 0), (1, 2), (0, 1), (0, 1)])
        np.testing.assert_array_equal(graph.arcs, [[2, 1], [1, 0], [1, 2], [0, 1], [0, 1]])
        np.testing.assert_array_equal(graph.canonical_edges, [0, 1, 2, 3, 3])
        self.assertTrue(validate_route(graph, [3, 2])["valid"])
        self.assertTrue(validate_route(graph, [4, 2])["valid"])
        self.assertFalse(validate_route(graph, [2, 3])["valid"])
        self.assertEqual(validate_route(graph, [-1])["invalid_positions"], [0])
        self.assertEqual(validate_route(graph, [5])["invalid_positions"], [0])
        self.assertEqual(validate_route(graph, [3, 2])["arc_count"], 2)

    def test_lon_lat_geometry_is_in_meters(self):
        nodes = np.array([[8.0, 50.0], [8.001, 50.0], [8.0, 50.001]])
        graph = Graph(nodes, np.array([[0, 1], [0, 2]]))
        self.assertAlmostEqual(graph.edge_lengths[0], 71.474, delta=0.05)
        self.assertAlmostEqual(graph.edge_lengths[1], 111.195, delta=0.05)
        self.assertLess(graph.edge_lengths[0], graph.edge_lengths[1])
        np.testing.assert_allclose(graph.unproject(graph.project(nodes)), nodes, atol=1e-12)

    def test_projection_preserves_equivalent_local_fixture_at_latitude(self):
        points = [(0, 0), (50, 0), (50, 100)]
        low = metric_graph(points, [(0, 1), (1, 2)], latitude=0)
        high = metric_graph(points, [(0, 1), (1, 2)], latitude=50)
        np.testing.assert_allclose(low.edge_lengths, high.edge_lengths, rtol=1e-5, atol=1e-5)

    def test_graph_rejects_invalid_coordinates_endpoints_and_fractional_ids(self):
        valid_nodes = np.array([[8.0, 50.0], [8.001, 50.0]])
        for arcs in ([[0, -1]], [[0, 2]], [[0, 0.5]], [[0, math.nan]], [[0, math.inf]]):
            with self.subTest(arcs=arcs), self.assertRaises(ValueError):
                Graph(valid_nodes, np.asarray(arcs))
        for nodes in ([[181, 0], [0, 0]], [[8, 91], [8, 50]],
                      [[math.nan, 0], [8, 50]], [[math.inf, 0], [8, 50]], []):
            with self.subTest(nodes=nodes), self.assertRaises(ValueError):
                Graph(np.asarray(nodes), np.array([[0, 1]]))
        with self.assertRaises(ValueError):
            Graph(valid_nodes, np.empty((0, 2), dtype=int))
        with self.assertRaises(ValueError):
            integer_array([True, False], "route")
        with self.assertRaises(ValueError):
            validate_route(Graph(valid_nodes, np.array([[0, 1]])), [0.2])

    def test_finite_strict_time_validation_without_sorting(self):
        irregular = metric_track([(0, 0), (0, 0), (1, 0)], [100.25, 108.25, 127.5])
        np.testing.assert_array_equal(validate_track(irregular), irregular)
        for times in ([1, 1, 2], [1, 0, 2], [1, math.nan, 2], [1, math.inf, 2]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                validate_track(metric_track([(0, 0), (0, 0), (1, 0)], times))
        with self.assertRaises(ValueError):
            validate_track(np.empty((0, 3)))
        with self.assertRaises(ValueError):
            validate_track(np.ones((3, 2)))

    def test_load_write_and_discovery_need_no_route_file(self):
        with tempfile.TemporaryDirectory(prefix="paths with spaces ") as root:
            second = write_fixture(root, "00000009")
            first = write_fixture(root, "00000002")
            prefixes = discover_trips(root)
            self.assertEqual(prefixes, [first, second])
            graph = load_graph(first)
            track = load_track(first.with_suffix(".track"))
            self.assertEqual(track.shape, (4, 3))
            self.assertEqual(len(graph.arcs), 4)
            route_path = Path(root) / "output folder" / "two.route"
            write_route(route_path, [1, 0])
            self.assertEqual(route_path.read_bytes(), b"1\n0\n")
            np.testing.assert_array_equal(load_route(route_path), [1, 0])
            write_route(route_path, [])
            self.assertEqual(load_route(route_path).shape, (0,))
            self.assertTrue(validate_route(graph, [1, 0])["valid"])

    def test_single_row_parser_and_bad_input_errors(self):
        with tempfile.TemporaryDirectory() as root:
            prefix = write_fixture(root, points=[(0, 0), (100, 0)],
                                   arcs=[(0, 1)], samples=[(20, 2)])
            self.assertEqual(load_graph(prefix).arcs.shape, (1, 2))
            self.assertEqual(load_track(prefix.with_suffix(".track")).shape, (1, 3))
            for text in ("8 50\n", "8 50 1 7\n", "8 50 nan\n", "# longitude latitude time\n8 50 1\n", ""):
                prefix.with_suffix(".track").write_text(text, encoding="utf-8")
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_track(prefix.with_suffix(".track"))
            prefix.with_suffix(".route").write_text("0.5\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_route(prefix.with_suffix(".route"))

    def test_graph_parser_never_skips_blank_physical_rows(self):
        # Implicit physical row IDs must not change when malformed input has a blank.
        with tempfile.TemporaryDirectory() as root:
            prefix = write_fixture(root)
            for suffix in (".nodes", ".arcs"):
                path = prefix.with_suffix(suffix)
                original = path.read_text(encoding="utf-8")
                rows = original.splitlines()
                for blank in ("", "   ", "\t"):
                    path.write_text("\n".join([rows[0], blank, *rows[1:]]) + "\n", encoding="utf-8")
                    with self.subTest(suffix=suffix, blank=blank), self.assertRaises(ValueError):
                        load_graph(prefix)
                path.write_text(original, encoding="utf-8")

    def test_discovery_rejects_incomplete_and_duplicate_trip_ids(self):
        with tempfile.TemporaryDirectory() as root:
            prefix = write_fixture(root)
            prefix.with_suffix(".arcs").rename(prefix.with_suffix(".arc_backup"))
            with self.assertRaises(FileNotFoundError):
                discover_trips(root)
        with tempfile.TemporaryDirectory() as root:
            write_fixture(root, "first")
            other = write_fixture(root, "second")
            for suffix in (".nodes", ".arcs", ".track"):
                other.with_suffix(suffix).rename(other.with_name("first").with_suffix(suffix))
            with self.assertRaises(ValueError):
                discover_trips(root)


class DirectedRoutingTests(unittest.TestCase):
    def test_direction_path_row_identity_and_source_self(self):
        graph = metric_graph([(0, 0), (100, 0), (200, 0)], [(1, 2), (0, 1)])
        router = Router(graph)
        distance, edges = router.shortest_path(0, 2)
        self.assertEqual(edges, [1, 0])
        self.assertAlmostEqual(distance, sum(graph.edge_lengths))
        self.assertEqual(router.shortest_path(1, 1), (0.0, []))
        reverse_distance, reverse_edges = router.shortest_path(2, 0)
        self.assertTrue(math.isinf(reverse_distance))
        self.assertEqual(reverse_edges, [])

    def test_parallel_weights_not_summed_and_duplicate_first_id(self):
        graph = metric_graph([(0, 0), (100, 0), (200, 0)],
                             [(0, 1), (0, 1), (1, 2)])
        distance, edges = Router(graph).shortest_path(0, 2)
        self.assertEqual(edges, [0, 2])
        self.assertAlmostEqual(distance, graph.edge_lengths[0] + graph.edge_lengths[2])

    def test_zero_length_cycles_and_disconnected_nodes(self):
        graph = metric_graph([(0, 0), (0, 0), (100, 0), (500, 0)],
                             [(0, 0), (0, 1), (1, 0), (1, 2)])
        router = Router(graph)
        self.assertEqual(router.shortest_path(0, 1, max_distance=0), (0.0, [1]))
        distance, edges = router.shortest_path(0, 2)
        self.assertEqual(edges, [1, 3])
        self.assertAlmostEqual(distance, graph.edge_lengths[3])
        self.assertEqual(router.shortest_path(3, 3), (0.0, []))
        self.assertTrue(math.isinf(router.shortest_path(3, 0)[0]))

    def test_cutoff_boundary_and_cache_do_not_leak_distances(self):
        graph = metric_graph([(0, 0), (100, 0), (200, 0), (400, 0)],
                             [(0, 1), (1, 2), (2, 3)])
        router = Router(graph, cache_size=1, distance_bucket_m=50)
        exact = float(graph.edge_lengths[:2].sum())
        self.assertTrue(math.isinf(router.shortest_path(0, 2, exact - 1e-6)[0]))
        self.assertEqual(router.shortest_path(0, 2, exact)[1], [0, 1])
        self.assertTrue(math.isinf(router.shortest_path(0, 3, exact)[0]))
        self.assertEqual(router.shortest_path(0, 3, math.inf)[1], [0, 1, 2])
        self.assertTrue(math.isinf(router.shortest_path(0, 3, exact)[0]))
        self.assertEqual(router.shortest_path(1, 3)[1], [1, 2])
        self.assertEqual(router.shortest_path(0, 2)[1], [0, 1])
        values = router.distances(0, [2, 2, 3], exact)
        np.testing.assert_allclose(values[:2], [exact, exact])
        self.assertTrue(math.isinf(values[2]))

    def test_invalid_nodes_and_cutoffs_are_rejected(self):
        router = Router(metric_graph([(0, 0), (100, 0)], [(0, 1)]))
        for node in (-1, 2, 0.5, True, math.nan):
            with self.subTest(node=node), self.assertRaises((ValueError, TypeError)):
                router.shortest_path(node, 0)
        for cutoff in (-1, math.nan):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                router.shortest_path(0, 1, cutoff)

    def test_matches_independent_random_graph_oracle(self):
        rng = np.random.default_rng(20261118)
        for case in range(8):
            points = rng.uniform(-300, 300, (10, 2))
            points[1] = points[0]  # topologically distinct, geometrically equal
            arcs = rng.integers(0, 10, (30, 2))
            arcs = np.vstack([arcs, arcs[:2], [[0, 1], [1, 0]]])
            graph = metric_graph(points, arcs)
            router = Router(graph, cache_size=3)
            for source in range(10):
                expected = independent_distances(graph, source)
                actual = router.distances(source, list(range(10)))
                np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-8)
                for target, value in enumerate(expected):
                    with self.subTest(case=case, source=source, target=target):
                        distance, edges = router.shortest_path(source, target)
                        if math.isinf(value):
                            self.assertTrue(math.isinf(distance))
                            self.assertEqual(edges, [])
                            continue
                        self.assertAlmostEqual(distance, value, delta=1e-8)
                        if edges:
                            self.assertEqual(int(graph.tails[edges[0]]), source)
                            self.assertEqual(int(graph.heads[edges[-1]]), target)
                            self.assertTrue(validate_route(graph, edges)["valid"])
                            self.assertAlmostEqual(float(graph.edge_lengths[edges].sum()), distance, delta=1e-8)
                        else:
                            self.assertEqual(source, target)


class CandidateGeometryTests(unittest.TestCase):
    def test_long_segment_interior_survives_many_near_midpoint_distractors(self):
        points = [(-5000, 0), (5000, 0)]
        arcs = [(0, 1)]
        for offset in range(24):
            start = len(points)
            points.extend([(3490, 5 + offset), (3510, 5 + offset)])
            arcs.append((start, start + 1))
        graph = metric_graph(points, arcs)
        query = graph.project(metric_track([(3500, 2)])[:, :2])[0]
        found = CandidateIndex(graph, spacing_m=40).query(query, radius_m=20, max_candidates=3)
        self.assertEqual(int(found.edges[0]), 0)
        self.assertAlmostEqual(found.fractions[0], 0.85, places=7)
        self.assertAlmostEqual(found.distances[0], 2, delta=0.001)
        self.assertEqual(len(found), 3)

    def test_projection_clamps_and_reverses_fraction(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1), (1, 0), (0, 1)])
        index = CandidateIndex(graph)
        for point, expected in [((-10, 0), (0, 1)), ((25, 3), (0.25, 0.75)), ((110, 0), (1, 0))]:
            query = graph.project(metric_track([point])[:, :2])[0]
            found = index.query(query, radius_m=30, max_candidates=10)
            self.assertEqual(set(found.edges.tolist()), {0, 1})
            mapping = dict(zip(found.edges.tolist(), found.fractions.tolist()))
            self.assertAlmostEqual(mapping[0], expected[0], places=7)
            self.assertAlmostEqual(mapping[1], expected[1], places=7)
        on_segment = graph.project(metric_track([(50, 0)])[:, :2])[0]
        np.testing.assert_array_equal(index.query(on_segment, 1, 10).edges, [0, 1])

    def test_zero_length_projection_is_finite(self):
        graph = metric_graph([(0, 0), (0, 0)], [(0, 0), (0, 1), (1, 0)])
        query = graph.project(metric_track([(3, 4)])[:, :2])[0]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            found = CandidateIndex(graph).query(query, 10, 10)
        self.assertEqual(len(found), 3)
        np.testing.assert_allclose(found.distances, 5, atol=1e-7)
        self.assertTrue(np.isfinite(found.fractions).all())
        self.assertTrue(((found.fractions >= 0) & (found.fractions <= 1)).all())

    def test_exact_radius_retrieval_agrees_with_exhaustive_oracle(self):
        rng = np.random.default_rng(409)
        graph = metric_graph(rng.uniform(-1000, 1000, (40, 2)),
                             rng.integers(0, 40, (75, 2)))
        index = CandidateIndex(graph, spacing_m=73)
        for i in range(40):
            point = rng.uniform(-1000, 1000, 2)
            radius = float(rng.uniform(10, 300))
            expected = exhaustive_segments(graph, point, radius)
            found = index.query(point, radius, len(graph.arcs))
            with self.subTest(query=i):
                self.assertEqual(set(found.edges.tolist()), set(expected))
                for edge, fraction, distance in zip(found.edges, found.fractions, found.distances):
                    self.assertAlmostEqual(float(fraction), expected[int(edge)][0], delta=1e-10)
                    self.assertAlmostEqual(float(distance), expected[int(edge)][1], delta=1e-8)
                actual_order = list(zip(found.distances.tolist(), found.edges.tolist()))
                self.assertEqual(actual_order, sorted(actual_order))

    def test_radius_boundary_and_empty_candidate_set(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1)])
        index = CandidateIndex(graph, spacing_m=40)
        query = graph.project(metric_track([(10, 20)])[:, :2])[0]
        self.assertEqual(index.query(query, 20, 5).edges.tolist(), [0])
        self.assertEqual(len(index.query(query, 19, 5)), 0)
        self.assertEqual(len(index.query(query + [0, 10000], 10, 5)), 0)
        for point in ([math.nan, 0], [0, math.inf], [0, 0, 0]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                index.query(np.array(point))
        for radius in (0, -1, math.nan, math.inf):
            with self.subTest(radius=radius), self.assertRaises(ValueError):
                index.query(query, radius, 5)


class MatcherAcceptanceTests(unittest.TestCase):
    def assert_result(self, graph, track, result):
        self.assertTrue(validate_route(graph, result.route)["valid"])
        self.assertEqual(result.breaks, [])
        length = len(result.observation_indices)
        self.assertEqual(len(result.matched_edges), length)
        self.assertEqual(len(result.matched_fractions), length)
        self.assertEqual(len(result.matched_distances), length)
        self.assertEqual(int(result.observation_indices[0]), 0)
        self.assertEqual(int(result.observation_indices[-1]), len(track) - 1)
        self.assertTrue((np.diff(result.observation_indices) > 0).all())
        self.assertTrue(((result.matched_edges >= 0) & (result.matched_edges < len(graph.arcs))).all())
        self.assertTrue(((result.matched_fractions >= 0) & (result.matched_fractions <= 1)).all())
        self.assertTrue(np.isfinite(result.matched_distances).all())
        self.assertFalse(result.diagnostics["uses_reference_route"])

    def test_sparse_track_recovers_unobserved_connecting_edges(self):
        graph = metric_graph([(0, 0), (100, 0), (200, 0), (300, 0)],
                             [(2, 3), (0, 1), (1, 2)])
        track = metric_track([(10, 1), (290, -1)], [0.25, 20.75])
        result = MapMatcher(graph, MatchConfig(candidate_radius_m=10)).match(track)
        self.assertEqual(result.route.tolist(), [1, 2, 0])
        self.assertEqual(result.matched_edges.tolist(), [1, 0])
        self.assert_result(graph, track, result)

    def test_stationary_and_backward_jitter_does_not_invent_traversals(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1)])
        points = [(50, 1), (50, 1), (45, -1), (52, 2), (47, -1), (53, 0)]
        track = metric_track(points, [1, 2, 17, 19, 27, 50])
        result = MapMatcher(graph).match(track)
        self.assertEqual(result.route.tolist(), [0])
        self.assertEqual(result.matched_edges.tolist(), [0] * len(track))
        self.assert_result(graph, track, result)

    def test_real_directed_loop_keeps_repeated_edge(self):
        graph = metric_graph([(0, 0), (100, 0), (50, 100)], [(0, 1), (1, 2), (2, 0)])
        track = metric_track([(20, 0), (80, 0), (90, 20), (60, 80),
                              (40, 80), (10, 20), (20, 0), (80, 0)])
        result = MapMatcher(graph, MatchConfig(candidate_radius_m=5, backward_tolerance_m=0)).match(track)
        self.assertEqual(result.route.tolist(), [0, 1, 2, 0])
        self.assertEqual(result.route.tolist().count(0), 2)
        self.assert_result(graph, track, result)

    def test_zero_length_connector_is_retained(self):
        graph = metric_graph([(0, 0), (100, 0), (100, 0), (200, 0)],
                             [(0, 1), (1, 2), (2, 3)])
        track = metric_track([(20, 0), (180, 0)])
        result = MapMatcher(graph, MatchConfig(candidate_radius_m=5)).match(track)
        self.assertEqual(result.route.tolist(), [0, 1, 2])
        self.assert_result(graph, track, result)

    def test_disconnected_track_fails_explicitly(self):
        graph = metric_graph([(0, 0), (100, 0), (1000, 0), (1100, 0)],
                             [(0, 1), (2, 3)])
        track = metric_track([(20, 0), (1080, 0)])
        with self.assertRaisesRegex(MatchingError, "connected.*transition|disconnected"):
            MapMatcher(graph, MatchConfig(candidate_radius_m=10, max_radius_m=10)).match(track)

    def test_bounded_candidate_retry_recovers_connected_road_pruned_by_nearer_islands(self):
        points = [(0, 0), (200, 0)]
        arcs = [(0, 1)]
        # These 13 closer roads are disconnected; none is within 80 m of sample 0.
        # Only the original main edge provides a legal cross-observation path.
        for i in range(13):
            first = len(points)
            points.extend([(185, 1 + i * 0.05), (195, 1 + i * 0.05)])
            arcs.append((first, first + 1))
        graph = metric_graph(points, arcs)
        track = metric_track([(10, 0), (190, 2)])
        config = MatchConfig(max_candidates=12, candidate_radius_m=80, max_radius_m=80)
        second = CandidateIndex(graph).query(graph.project(track[1:2, :2])[0], 80, 12)
        self.assertNotIn(0, second.edges.tolist(), "fixture must actually prune the connected edge")
        expanded = CandidateIndex(graph).query(graph.project(track[1:2, :2])[0], 80, 24)
        self.assertIn(0, expanded.edges.tolist())
        result = MapMatcher(graph, config).match(track)
        self.assertEqual(result.route.tolist(), [0])
        self.assertEqual(result.matched_edges.tolist(), [0, 0])
        self.assert_result(graph, track, result)

    def test_candidate_expansion_and_far_outlier_error(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1)])
        config = MatchConfig(candidate_radius_m=5, max_radius_m=100)
        track = metric_track([(50, 30)])
        result = MapMatcher(graph, config).match(track)
        self.assertEqual(result.route.tolist(), [0])
        self.assertEqual(result.diagnostics["expanded_candidate_observations"], 1)
        self.assert_result(graph, track, result)
        with self.assertRaisesRegex(MatchingError, "observation 0.*no road candidate"):
            MapMatcher(graph, config).match(metric_track([(50, 1000)]))

    def test_thinning_preserves_endpoints_and_reports_indices(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1)])
        track = metric_track([(50, 0)] * 20)
        config = MatchConfig(observation_distance_m=10, observation_interval=5)
        result = MapMatcher(graph, config).match(track)
        self.assertEqual(result.observation_indices.tolist(), [0, 5, 10, 15, 19])
        self.assertEqual(result.diagnostics["observation_count"], 20)
        self.assertEqual(result.diagnostics["matched_observation_count"], 5)
        self.assert_result(graph, track, result)

    def test_timestamp_shift_does_not_change_spatial_decoding(self):
        graph = metric_graph([(0, 0), (100, 0), (200, 0)], [(0, 1), (1, 2)])
        track = metric_track([(20, 0), (70, 0), (130, 0), (180, 0)], [0.5, 1.5, 20.5, 25.5])
        shifted = track.copy()
        shifted[:, 2] += 50000
        first = MapMatcher(graph).match(track)
        second = MapMatcher(graph).match(shifted)
        np.testing.assert_array_equal(first.route, second.route)
        np.testing.assert_array_equal(first.matched_edges, second.matched_edges)

    def test_deterministic_repeated_calls_and_finite_long_trace(self):
        graph = metric_graph([(0, 0), (100, 0)], [(0, 1), (1, 0), (0, 1)])
        track = metric_track([(50, 0)] * 1200)
        matcher = MapMatcher(graph)
        first = matcher.match(track)
        second = matcher.match(track)
        np.testing.assert_array_equal(first.route, second.route)
        np.testing.assert_array_equal(first.matched_edges, second.matched_edges)
        np.testing.assert_array_equal(first.matched_fractions, second.matched_fractions)
        self.assertTrue(math.isfinite(first.diagnostics["negative_log_likelihood_relative"]))
        self.assert_result(graph, track, first)

    def test_prediction_never_opens_reference_sentinel(self):
        with tempfile.TemporaryDirectory() as root:
            prefix = write_fixture(root)
            prefix.with_suffix(".route").write_text("DO NOT OPEN THIS REFERENCE\n", encoding="utf-8")
            original_builtin_open = builtins.open
            original_io_open = io.open
            opened = []

            def guarded(original):
                def open_without_reference(file, *args, **kwargs):
                    if isinstance(file, (str, bytes, Path)):
                        path = Path(file)
                        opened.append(str(path))
                        if path.suffix == ".route":
                            raise AssertionError("inference attempted to open a reference route")
                    return original(file, *args, **kwargs)
                return open_without_reference

            with mock.patch("builtins.open", side_effect=guarded(original_builtin_open)), \
                    mock.patch("io.open", side_effect=guarded(original_io_open)), \
                    mock.patch("mapmatch.data.load_route", side_effect=AssertionError("reference loader called")):
                discovered = discover_trips(root)
                self.assertEqual(discovered, [prefix])
                graph = load_graph(discovered[0])
                track = load_track(prefix.with_suffix(".track"))
                result = MapMatcher(graph).match(track)
            self.assertEqual(result.route.tolist(), [1, 0])
            self.assertFalse(any(path.endswith(".route") for path in opened))

    def test_config_rejects_nonfinite_and_invalid_integer_parameters(self):
        for key, value in (("sigma_m", 0), ("beta_m", math.nan), ("routing_slack_m", -1),
                           ("max_candidates", 1.5), ("cache_size", True),
                           ("backward_tolerance_m", -1), ("observation_distance_m", math.inf),
                           ("max_radius_m", 1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                MatchConfig(**{key: value})


if __name__ == "__main__":
    unittest.main()
