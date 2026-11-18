"""Exact-index publication privacy checks; no access to real credentials or raw data."""
from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile

from scripts import review_publication as publication


def submission(names=None, content=b"0\n1\n"):
    output=io.BytesIO()
    with ZipFile(output,"w") as archive:
        for name in names or [f"{i:08d}.route" for i in range(10)]:
            archive.writestr(name,content)
    return output.getvalue()


class PublicationPrivacyTests(unittest.TestCase):
    def scan(self,name,data):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            path=root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            # Working bytes deliberately differ; the scanner must audit git's index.
            path.write_bytes(b"working tree is not the reviewed staged blob")
            with mock.patch.object(publication,"ROOT",root), mock.patch.object(publication.subprocess,"check_output",return_value=data) as git:
                result=publication.audit([name])
            git.assert_called_once_with(["git","show",":"+name],cwd=root)
            return result

    def test_exact_staged_blob_is_scanned_not_working_content(self):
        result=self.scan("docs/review.txt",(b"-----BEGIN " + b"PRIVATE " + b"KEY-----"))
        self.assertFalse(result["passed"])
        self.assertEqual(result["findings"][0]["rule"],"private_key")
        self.assertFalse(result["secret_values_echoed"])
        self.assertEqual(result["files"][0]["bytes"],len((b"-----BEGIN " + b"PRIVATE " + b"KEY-----")))

    def test_private_evaluator_and_row_cache_paths_are_rejected(self):
        for name in ["evaluator_test/trip.json","nested/private_labels/x.csv","experiments/predictions/x.csv","experiments/cache/x.json","experiments/fixture.npz","Student folder/file.json"]:
            with self.subTest(path=name):
                result=self.scan(name,b"non-secret synthetic contents")
                self.assertFalse(result["passed"])
                self.assertTrue(any(x["rule"]=="raw_private_or_row_level_input" for x in result["findings"]))

    def test_only_flat_ten_derived_route_submission_archive_is_allowed(self):
        self.assertTrue(self.scan("outputs/phase_three_student_test/submission.zip",submission())["passed"])
        cases=[("archive.zip",submission()),("evaluator_test.zip",submission()),
               ("outputs/phase_three_student_test/submission.zip",submission(["../escape.route"])),
               ("outputs/phase_three_student_test/submission.zip",b"not a ZIP archive")]
        for name,data in cases:
            with self.subTest(path=name,bytes=len(data)):
                self.assertFalse(self.scan(name,data)["passed"])

    def test_secret_inside_an_otherwise_valid_submission_is_rejected(self):
        result=self.scan("outputs/phase_three_student_test/submission.zip",submission(content=(b"-----BEGIN " + b"PRIVATE " + b"KEY-----")))
        self.assertFalse(result["passed"])
        self.assertTrue(any(x["rule"]=="secret_in_submission_member" for x in result["findings"]))


if __name__=="__main__":
    unittest.main()
