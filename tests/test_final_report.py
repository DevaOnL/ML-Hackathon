"""Synthetic hardening regressions; no real raw/model/inference/network access."""
from __future__ import annotations
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = ROOT / "scripts" / "build_final_report.py"
spec = importlib.util.spec_from_file_location("hardened_report_proposal", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def digest(data):
    return hashlib.sha256(data).hexdigest()


class HardenedProposalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_", dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(m, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else value.encode())
        return path

    def state(self, key, data):
        return {"inputs": {key: {"data": data}}, "load_errors": [], "missing": [], "assets": {}}

    def test_strict_json_rejects_nan_infinity_duplicate_and_overflow(self):
        for text in ('{"n":NaN}', '{"n":Infinity}', '{"n":1e999}', '{"n":1,"n":2}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                m.strict_json(text)
        self.assertEqual(m.strict_json('{"n":0}'), {"n": 0})

    def test_canonical_path_refuses_traversal_aliases_and_symlinks(self):
        self.write("plain.json", "{}")
        for path in ("../plain.json", "/plain.json", "./plain.json", "a//b", "a/../plain.json", "%70lain.json", " plain.json", "a\\b"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                m.scoped_file(path)
        (self.root / "link.json").symlink_to(self.root / "plain.json")
        with self.assertRaises(ValueError):
            m.scoped_file("link.json")
        self.assertEqual(m.scoped_file("plain.json"), self.root / "plain.json")

    def test_build_state_reports_invalid_mandatory_json(self):
        self.write("bad.json", '{"n":NaN}')
        with patch.object(m, "INPUTS", [{"key":"toy", "path":"bad.json", "kind":"json", "final":True}]):
            state = m.build_state("final")
        self.assertEqual(len(state["missing"]), 1)
        self.assertIn("nonfinite", state["load_errors"][0])

    def test_nested_delivery_false_and_all_malformed_check_trees_fail(self):
        good = {"passed":True, "checks":{"global":True}, "trips":{"x":{"offline_checks":{"a":True}}}}
        self.assertTrue(m.verify_delivery(self.state("toy", good), "toy", "toy")[0])
        for bad in (False, 1, "true", [], {}, {"nested":False}, {"nested":[]}):
            data = copy.deepcopy(good)
            data["trips"]["x"]["offline_checks"] = bad
            with self.subTest(value=bad):
                self.assertFalse(m.verify_delivery(self.state("toy", data), "toy", "toy")[0])
        data = copy.deepcopy(good)
        data["checks"] = {}
        self.assertFalse(m.verify_delivery(self.state("toy", data), "toy", "toy")[0])

    def test_legitimate_metric_false_and_unavailable_singular_check_not_reinterpreted(self):
        data = {"checks":{"ok":True}, "metric":{"exact_match":False},
                "probability_audit":{"independent_saved_probability_sum_check":"not available from historical CLI"}}
        self.assertEqual(m.validate_check_containers(data), 1)

    def test_raw_template_coverage_distinct_from_resolved_text(self):
        records = [{"name":"toy", "optional":False}]
        with patch.object(m, "FACTS_BY_NAME", {"toy":records[0]}), patch.object(m, "FACTS", records):
            self.assertEqual(m.scan_template("value {{fact:toy}}"), [])
            self.assertEqual(m.scan_template("value 1", check_tokens=False), [])
            self.assertTrue(m.scan_template("value 1"))
            self.assertTrue(m.scan_template("{{table:unknown}} {{fact:toy}}"))

    def test_all_facts_mandatory_even_if_legacy_optional_true(self):
        with patch.object(m, "FACTS", [{"name":"toy", "optional":True}]):
            self.assertFalse(m.evaluate_facts("", {"toy":None,"toy__err":"missing"})[0]["ok"])
            self.assertTrue(m.evaluate_facts("1", {"toy":1,"toy__str":"1","toy__err":None})[0]["ok"])

    def test_historical_paths_need_labelled_section(self):
        self.assertTrue(m._historical_path_problems("# Results\nexperiments/phase_three/results.json"))
        self.assertEqual(m._historical_path_problems("# Historical appendix\nexperiments/phase_three/results.json"), [])

    def suite_record(self):
        logs = {"python":"Ran 3 tests in 0.1s\n\nOK\n", "hosting":"Ran 2 tests in 0.1s\n\nOK\n",
                "node":"ℹ tests 4\nℹ pass 4\nℹ fail 0\n", "static":"PASS static smoke\n"}
        data = {"total":9,"passed":9,"failures":[],"complete_release_suite":True,
                "static_smoke_passed":True,"synthetic_tests_only":True,"production_fit_or_inference":False,
                "counts":{"python":3,"hosting":2,"node":4},"suites":[]}
        for name, log in logs.items():
            raw = log.encode()
            self.write(name + ".log", raw)
            data["suites"].append({"suite":name,"exit_code":0,"log":name+".log","log_sha256":digest(raw)})
        return data

    def test_complete_positive_suite_counts_and_bound_logs_pass(self):
        self.assertTrue(m.verify_test_record(self.state("testsjson", self.suite_record()))[0])

    def test_suite_zero_bool_counts_incomplete_failures_and_log_drift_fail(self):
        good = self.suite_record()
        mutations = [{"total":0,"passed":0},{"total":True,"passed":True},
                     {"complete_release_suite":False},{"failures":["failed"]},
                     {"counts":{"python":2,"hosting":3,"node":4}},{"suites":[]}]
        for change in mutations:
            data = copy.deepcopy(good)
            data.update(change)
            with self.subTest(change=change):
                self.assertFalse(m.verify_test_record(self.state("testsjson", data))[0])
        self.write("python.log", "Ran 99 tests in 0.1s\n\nOK\n")
        self.assertFalse(m.verify_test_record(self.state("testsjson", good))[0])

    def test_suite_logs_must_agree_even_after_rehash(self):
        data = self.suite_record()
        raw = b"Ran 99 tests in 0.1s\n\nOK\n"
        self.write("python.log", raw)
        data["suites"][0]["log_sha256"] = digest(raw)
        self.assertFalse(m.verify_test_record(self.state("testsjson", data))[0])

    def test_requirements_exact47_unique_terminal(self):
        rows = [{"id":"R-%d"%i,"status":"done_evidence_exists"} for i in range(47)]
        m.verify_requirements(self.state("reqs", {"requirements":rows}))
        for mutation in ("not_yet_done", "deferred_by_user", "partially_done"):
            rows[-1]["status"] = mutation
            with self.subTest(status=mutation), self.assertRaises(ValueError):
                m.verify_requirements(self.state("reqs", {"requirements":rows}))

    def test_png_requires_valid_bytes_dimensions_and_hash(self):
        from PIL import Image
        path = self.root / "tiny.png"
        Image.new("RGB", (2,3)).save(path)
        entry = {"path":"tiny.png","sha256":m.sha256_file(path),"pixels":{"width":2,"height":3}}
        state = {"assets":{}}
        m._png_binding(state, entry)
        self.assertIn("tiny.png", state["assets"])
        entry["pixels"]["width"] = 99
        with self.assertRaises(ValueError):
            m._png_binding(state, entry)
        path.write_bytes(b"not png")
        entry = {"path":"tiny.png","sha256":m.sha256_file(path)}
        with self.assertRaises(ValueError):
            m._png_binding(state, entry)

    def test_final_figure_manifest_must_be_complete_not_optional(self):
        state = self.state("figman5", {"partial":False,"figure_count":22,"model_figure_count":22,"decoder_figure_count":0})
        errors, notes = m.verify_figures(state, "")
        self.assertTrue(errors)
        self.assertEqual(notes, [])

    def test_screenshots_missing_hash_fail_not_notes(self):
        with self.assertRaises(ValueError):
            m._png_binding({"assets":{}}, {"path":"missing.png"})

    def test_final_map_missing_binding_fails_closed(self):
        with self.assertRaises(ValueError):
            m.verify_map(self.state("mapman", {"passed":True}))

    def test_private_source_dependencies_never_opened(self):
        for path in ("Student folder/x.track", "experiments/cache/x.npz", "private/x.json", "evaluator_test/x.json"):
            self.assertFalse(m._public_dependency(path), path)
        self.assertTrue(m._public_dependency("reports/final_five/evaluator/acceptance.json"))
        self.assertTrue(m._public_dependency("experiments/final_five/current/results.json"))

    def test_response_requires_present_model_config_and_all_observations(self):
        compact = "a" * 64
        row = {"http_status":200,"metrics_status":"ok","dataset_id":"development","trip_id":"same",
               "model":{"sha256":m.PINNED_MODEL_SHA256,"name":m.SELECTED_FAMILY},"match_config_hash":compact,
               "request_id":"b"*32,"diagnostics":{"observation_count":2,"matched_observation_count":2,
               "learned_emission_observations":2,"uses_reference_route":False},"observation_assignment_count":2,
               "metrics":{"macro_f1":0.8},"checks":{"ok":True},"response_checks":{"ok":True}}
        m._check_response(row,"development","same",compact)
        for field in ("model", "match_config_hash", "diagnostics", "checks"):
            bad = copy.deepcopy(row)
            del bad[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                m._check_response(bad,"development","same",compact)

    def test_exact_five_six_and_hpo_guard_rejects_incomplete(self):
        with self.assertRaises(ValueError):
            m.verify_selection_and_hpo(self.state("results5", {"family_count":5,"families":m.FIVE_FAMILIES[:-1]}))

    def test_preview_names_cannot_replace_final_names(self):
        with patch.multiple(m, TEMPLATE_PATH=self.root/"t.md",MD_PATH=self.root/"x",PDF_PATH=self.root/"y",MANIFEST_PATH=self.root/"z"):
            m._configure_paths(self.root, mode="preview")
            self.assertIn(".preview.", m.PDF_PATH.name)
            self.assertIn(".preview.", m.MD_PATH.name)
            self.assertIn(".preview.", m.MANIFEST_PATH.name)

    def test_real_cli_check_is_read_only_and_refuses_missing_evidence(self):
        self.write("reports/final_five/REPORT_TEMPLATE.md", "# Synthetic\n{{fact:phase_completed}}\n")
        before = {p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        result = subprocess.run([sys.executable,"-B",str(SOURCE),"--check","--root",str(self.root)],
                                capture_output=True,text=True,timeout=30,env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"})
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertNotIn("Traceback",result.stderr)
        self.assertIn("NOT READY",result.stdout)
        after = {p.relative_to(self.root).as_posix():p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before,after)

    def test_real_cli_final_refuses_without_writing_outputs(self):
        self.write("reports/final_five/REPORT_TEMPLATE.md", "# Synthetic\n{{fact:phase_completed}}\n")
        result = subprocess.run([sys.executable,"-B",str(SOURCE),"--final","--root",str(self.root)],
                                capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,1,result.stdout+result.stderr)
        self.assertFalse((self.root/"reports/final_five/TECHNICAL_REPORT.md").exists())
        self.assertFalse((self.root/"reports/final_five/PathSnap_Final_Technical_Report.pdf").exists())

    def test_pdf_renderer_real_draft_timestamp_and_watermark(self):
        if not m.HAS_PDF:
            self.skipTest("reportlab unavailable")
        path = self.root / "preview.pdf"
        m.render_pdf("# Synthetic report\nFacts are synthetic. sigma and beta.\n", path, draft=True)
        pages, text = m.pdf_page_count_and_text(path)
        self.assertGreater(pages,0)
        self.assertIn("DRAFT",text)
        self.assertIn("NOT FINAL",text)
        self.assertIn("Generated",text)

    def crop_state(self):
        from PIL import Image
        source = self.root / "original.png"
        Image.new("RGB", (4,6), (20,40,60)).save(source)
        source_hash = m.sha256_file(source)
        self.write("crop_script.py", "# synthetic")
        self.write("ocr.json", "{}")
        rows = []
        for name in m.REPORT_CROP_NAMES:
            relative = "reports/final_five/report_images/" + name + ".png"
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as image:
                detail = image.crop((0,0,2,3))
                detail.save(target)
                header = json.dumps({"mode":detail.mode,"size":detail.size},sort_keys=True).encode()
                pixel_hash = hashlib.sha256(header + b"\0" + detail.tobytes()).hexdigest()
            rows.append({"name":name,"path":relative,"output_path":relative,
                         "sha256":m.sha256_file(target),"output_sha256":m.sha256_file(target),
                         "source_path":"original.png","source_sha256":source_hash,
                         "source_width":4,"source_height":6,"crop_box":[0,0,2,3],
                         "output_width":2,"output_height":3,"pixel_equality_checks":True,
                         "pixel_sha256":pixel_hash,"ocr_path":"ocr.json","ocr_sha256":m.sha256_file(self.root/"ocr.json")})
        return {"assets":{},"inputs":{
            "report_crops":{"data":{"passed":True,"crop_count":12,"all_pixel_equality_checks":True,"crops":rows,
                                     "inputs":[{"path":"crop_script.py","sha256":m.sha256_file(self.root/"crop_script.py")}] }},
            "browser":{"data":{"screenshots":[{"path":"original.png","sha256":source_hash}]}},
            "termv":{"data":{"image":"original.png","image_sha256":source_hash}}}}

    def test_twelve_report_details_equal_actual_original_pixels(self):
        state = self.crop_state()
        self.assertEqual(len(m.verify_report_crops(state)),12)
        self.assertEqual(len(state["assets"]),12)
        state["inputs"]["report_crops"]["data"]["crops"][0]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError,"genuine"):
            m.verify_report_crops(state)

    def test_rehashed_modified_crop_is_not_accepted_as_original_pixels(self):
        from PIL import Image
        state = self.crop_state()
        row = state["inputs"]["report_crops"]["data"]["crops"][0]
        target = self.root / row["path"]
        with Image.open(target) as image:
            changed = image.copy()
        changed.putpixel((0,0),(255,0,0))
        changed.save(target)
        row["sha256"] = row["output_sha256"] = m.sha256_file(target)
        with self.assertRaisesRegex(ValueError,"exact source pixel"):
            m.verify_report_crops(state)

    def test_inline_uses_clickable_links_without_raw_markdown_or_fallback_spaces(self):
        rendered = m.inline("A [local source](<docs/FINAL_REQUIREMENTS.md>) and [site](https://ml.d3va.dev); gaussian_nb, tail.")
        self.assertNotIn("&#8203;", rendered)
        self.assertNotIn("\u200b", rendered)
        self.assertNotIn("](", rendered)
        self.assertIn('href="https://github.com/DevaOnL/ML-Hackathon/blob/main/docs/FINAL_REQUIREMENTS.md"', rendered)
        self.assertIn('href="https://ml.d3va.dev"', rendered)
        self.assertIn("gaussian_nb", rendered)
        with self.assertRaises(ValueError):
            m.inline("[bad](javascript:alert)")

    def test_section_starts_use_conditional_break_to_avoid_empty_page(self):
        if not m.HAS_PDF:
            self.skipTest("reportlab unavailable")
        flows = m.md_to_flowables("# First\nText.\n\n# Second\nMore.\n", m._styles())
        self.assertTrue(any(isinstance(f, m.CondPageBreak) for f in flows))
        self.assertFalse(any(isinstance(f, m.PageBreak) for f in flows))

    def test_pdf_render_preserves_underscores_and_clickable_source(self):
        if not m.HAS_PDF:
            self.skipTest("reportlab unavailable")
        from pypdf import PdfReader
        path = self.root / "links.pdf"
        m.render_pdf("# Synthetic\ngaussian_nb, hist_gradient_boosting_tuned and [source](<docs/FINAL_REQUIREMENTS.md>).\n", path, draft=False)
        doc = PdfReader(path)
        full = "\n".join(page.extract_text() or "" for page in doc.pages)
        self.assertIn("gaussian_nb", full)
        self.assertNotIn("\u25a0", full)
        annotations = [x.get_object() for p in doc.pages for x in p.get("/Annots", [])]
        self.assertTrue(any(a.get("/Subtype") == "/Link" for a in annotations))

    def test_table_font_and_no_artificial_page_cap(self):
        self.assertIsNone(m.REPORT_TARGET_PAGES)
        if m.HAS_PDF:
            styles = m._styles()
            self.assertGreaterEqual(styles["Cell"].fontSize,7.5)
            table = m._flow_table([["Model"]*9,["synthetic_config_name, key=value"]*9],styles)
            self.assertAlmostEqual(sum(table._colWidths),m.FRAME_W)

    def test_registry_digest_is_stable_across_python_hash_seeds(self):
        digests = []
        code = ("import runpy; m=runpy.run_path(" + repr(str(SOURCE)) + "); print(m['_registry_sha256']())")
        for seed in ("1", "2", "42"):
            result = subprocess.run([sys.executable,"-B","-c",code], cwd=ROOT,
                                    env={**os.environ,"PYTHONHASHSEED":seed,"PYTHONDONTWRITEBYTECODE":"1"},
                                    capture_output=True,text=True,check=True)
            digests.append(result.stdout.strip())
        self.assertEqual(len(set(digests)),1)
        resolved = next(f for f in m.FACTS if f["name"] == "req_resolved")
        self.assertEqual(resolved["count"]["in"], sorted(m.REQ_TERMINAL_STATUSES))

    def test_persisted_manifest_rejects_legacy_or_registry_drift(self):
        path = self.write("report_manifest.json", '{"mode":"final","schema_version":1}')
        with patch.object(m,"MANIFEST_PATH",path):
            self.assertTrue(m.verify_persisted_manifest({"inputs":{}},{}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
