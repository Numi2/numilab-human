from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

from numilab_human.cvsim_parameters import parse_cvsim_parameters
from numilab_human.model import ImportError as HumanImportError

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "third_party/physionet/cvsim21"
SOURCE = SOURCE_ROOT / "21-comp-backend/main/initial.c"
# Exact pin is repeated here so changing source and its manifest together cannot
# silently rewrite the baseline expected by these independent parser tests.
SOURCE_SHA256 = "c6a791e837e1542ebbb910dae1a2ac10bce5f7071ed1a9836b861b8d62267c03"


class CVSimParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.raw = SOURCE.read_bytes()

    def parse(self, raw: bytes | None = None) -> dict:
        if raw is None:
            return parse_cvsim_parameters(SOURCE, expected_sha256=SOURCE_SHA256)
        # Repinning a mutation deliberately exercises the restricted grammar,
        # independently from the first hash-admission test.
        return parse_cvsim_parameters(raw, expected_sha256=hashlib.sha256(raw).hexdigest())

    def test_curated_source_lock_and_complete_tables(self) -> None:
        lock = json.loads((SOURCE_ROOT / "source-lock.json").read_text())
        self.assertEqual(lock["files"]["main/initial.c"]["sha256"], SOURCE_SHA256)
        parsed = self.parse()
        self.assertEqual(parsed, self.parse(self.raw))
        self.assertEqual(parsed["parameter_count"], 153)
        self.assertEqual(len(parsed["source_tables"]), 142)
        self.assertEqual(len(parsed["source_members"]), 568)
        self.assertEqual([row["index"] for row in parsed["parameters"]], list(range(153)))
        self.assertEqual(parsed["parameter_vector"][70], 5150.0)
        self.assertEqual(parsed["parameter_vector"][90], 70.0)

    def test_independent_original_c_initializer_matches_every_scalar(self) -> None:
        compiler = shutil.which("cc")
        if compiler is None:
            self.skipTest("C compiler unavailable for independent initializer crosscheck")
        parsed = self.parse()
        statements = [f'printf("%a\\n", {key});' for key in parsed["source_members"]]
        driver = '\n'.join([
            '#include "initial.h"',
            'int main(void) { Hemo hemo; Cardiac cardiac; Micro_r micro_r;',
            'System_parameters system; Reflex reflex; Timing timing; Parameter_vector theta={{0.0}};',
            'initial_ptr(&hemo,&cardiac,&micro_r,&system,&reflex,&timing);',
            'mapping_ptr(&hemo,&cardiac,&micro_r,&system,&reflex,&timing,&theta);',
            'for(int i=0;i<N_PARAMETER;i++) printf("%a\\n",theta.vec[i]);',
            *statements, 'return 0; }',
        ])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            c_file = directory / "initializer_export.c"
            binary = directory / "initializer_export"
            c_file.write_text(driver)
            subprocess.run([compiler, "-std=c99", "-I", str(SOURCE.parent), str(c_file), str(SOURCE), "-o", str(binary)],
                           check=True, capture_output=True, text=True, timeout=30)
            output = subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=10)
        values = [float.fromhex(line) for line in output.stdout.splitlines()]
        self.assertEqual(values[:153], parsed["parameter_vector"])
        self.assertEqual(values[153:], [row["value"] for row in parsed["source_members"].values()])

    def test_crossed_arterial_owners_are_preserved(self) -> None:
        parsed = self.parse()
        self.assertEqual(parsed["parameter_vector"][138:140], [200.0, 16.0])
        self.assertEqual(parsed["parameters"][138]["source_member"], "hemo[2].v[0][0]")
        self.assertEqual(parsed["parameters"][139]["source_member"], "hemo[3].v[0][0]")

    def test_provenance_unused_tables_and_contradictory_ranges(self) -> None:
        parsed = self.parse()
        table = parsed["source_tables"]["reflex[0].vt[0]"]
        self.assertEqual([table[key] for key in ("nominal", "sd", "lower", "upper")], [5.3, 0.85, 3.6, 6.15])
        self.assertEqual(table["provenance"]["nominal"]["source_expression"], "5.3*alpha")
        self.assertEqual(table["provenance"]["nominal"]["constant_refs"], ["alpha"])
        self.assertEqual(parsed["parameters"][70]["mapping_line"], 570)
        self.assertEqual(parsed["source_members"]["system.bv[0][0]"]["line"], 315)
        self.assertIn("reflex[1].rr[0]", parsed["unmapped_source_tables"])
        self.assertIn("hemo[0].c[0][1]", parsed["unused_source_members"])
        self.assertEqual({row["member"] for row in parsed["source_inconsistencies"]},
                         {"hemo[10].h[0]", "hemo[12].h[0]", "hemo[6].h[0]", "system.T[2]", "system.w[0]", "timing.alpha_v[1]"})

    def test_line_preserving_comments_and_crlf(self) -> None:
        changed = self.raw.replace(b"float alpha = 1.;", b"float /* ignored assignment x=9; */ alpha = 1.;")
        changed = changed.replace(b"\n", b"\r\n")
        parsed = self.parse(changed)
        self.assertEqual(parsed["parameter_vector"], self.parse()["parameter_vector"])
        self.assertEqual(parsed["parameters"][70]["mapping_line"], 570)

    def test_float_constant_promotion_and_mapping_multiplication(self) -> None:
        changed = self.raw.replace(b"float alpha = 1.;", b"float alpha = 0.1;")
        changed = changed.replace(b"tmp -> vec[70] =  system->bv[0][0];", b"tmp -> vec[70] =  2.0*system->bv[0][0];", 1)
        parsed = self.parse(changed)
        alpha = struct.unpack("f", struct.pack("f", 0.1))[0]
        self.assertEqual(parsed["source_members"]["reflex[0].vt[0][0]"]["value"], 5.3 * alpha)
        self.assertEqual(parsed["parameter_vector"][70], 10300.0)
        self.assertEqual(parsed["parameters"][70]["source_members"], ["system.bv[0][0]"])

    def test_wrong_pin_and_missing_pin(self) -> None:
        for pin in ("0" * 64, "", None, SOURCE_SHA256.upper()):
            with self.subTest(pin=pin), self.assertRaises(HumanImportError):
                parse_cvsim_parameters(self.raw, expected_sha256=pin)

    def test_duplicate_scalar_and_parameter_and_constant(self) -> None:
        for before, after in [
            (b"float alpha = 1.;", b"float alpha = 1.; float alpha = 1.;"),
            (b"(*hemo)[0].c[0][0] = 0.28;", b"(*hemo)[0].c[0][0] = 0.28; (*hemo)[0].c[0][0] = 0.3;"),
            (b"tmp -> vec[70] =  system->bv[0][0];", b"tmp -> vec[70] =  system->bv[0][0]; tmp->vec[70] = 1.0;"),
        ]:
            self.assertIn(before, self.raw)
            with self.subTest(before=before), self.assertRaisesRegex(HumanImportError, "duplicate"):
                self.parse(self.raw.replace(before, after, 1))

    def test_unresolved_member_and_out_of_range_indices(self) -> None:
        for before, after in [
            (b"tmp -> vec[70] =  system->bv[0][0];", b"tmp -> vec[70] =  system->missing[0][0];"),
            (b"(*hemo)[0].c[0][0] = 0.28;", b"(*hemo)[17].c[0][0] = 0.28;"),
            (b"tmp -> vec[70]", b"tmp -> vec[153]"),
        ]:
            with self.subTest(after=after), self.assertRaises(HumanImportError):
                self.parse(self.raw.replace(before, after, 1))

    def test_missing_unused_scalar_and_missing_parameter_are_rejected(self) -> None:
        for statement in (b"(*reflex)[1].rr[0][1] = 0.0;", b"tmp -> vec[70] =  system->bv[0][0];"):
            self.assertIn(statement, self.raw)
            with self.subTest(statement=statement), self.assertRaisesRegex(HumanImportError, "missing"):
                self.parse(self.raw.replace(statement, b"", 1))

    def test_nonfinite_and_unsupported_arithmetic_are_rejected(self) -> None:
        for replacement in (b"1e999", b"NAN", b"INFINITY", b"1e300*1e300", b"1.0/0.0", b"sqrt(2.0)", b"1.0+2.0", b"010", b"10", b"alpha*beta", b"alpha*beta*5.3", b"system(\"touch should_not_execute\")"):
            with self.subTest(replacement=replacement), self.assertRaises(HumanImportError):
                self.parse(self.raw.replace(b"(*hemo)[0].c[0][0] = 0.28;", b"(*hemo)[0].c[0][0] = " + replacement + b";", 1))
        with self.assertRaisesRegex(HumanImportError, "nonfinite"):
            self.parse(self.raw.replace(b"float alpha = 1.;", b"float alpha = 1e100;", 1))

    def test_unconsumed_statements_and_nested_flow_are_rejected(self) -> None:
        for inserted in (b"mystery = 42.0;", b"(*hemo)[0].c[0][0] += 1.0;", b"unexpected_call();", b"if(1) { unexpected_call(); }", b"#define alpha 2.0\n"):
            with self.subTest(inserted=inserted), self.assertRaises(HumanImportError):
                self.parse(self.raw.replace(b"float alpha = 1.;", b"float alpha = 1.; " + inserted, 1))

    def test_malformed_source_is_rejected(self) -> None:
        for changed in (self.raw + b"/*", self.raw + b"\x00", self.raw + b"\xff", self.raw.replace(b"void mapping_ptr(", b"void omitted_mapping_ptr(", 1),
                        self.raw.replace(b"float alpha = 1.;", b"float alpha = 1.", 1), self.raw * 2):
            with self.subTest(size=len(changed)), self.assertRaises(HumanImportError):
                self.parse(changed)


if __name__ == "__main__":
    unittest.main()
