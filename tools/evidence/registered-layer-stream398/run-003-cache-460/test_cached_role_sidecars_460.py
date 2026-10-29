#!/usr/bin/env python3
"""Focused tests for cached 398 O/I sidecars and full-input SHA rejection."""
from __future__ import annotations
import argparse
import filecmp
import hashlib
import os
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = Path(os.environ.get("NUMI_HUMAN_TRANSFER_ROOT", HERE.parents[1]))
SCRIPT = HERE / "export_authoring_registered_surface_sidecars_231_cached.py"
CACHE_ROOT = P / "full-current-surface-registration-399"
CACHE_DIR = CACHE_ROOT / "sidecars"
CACHE_LAUNCH = CACHE_ROOT / "launch.json"
RUN_458 = P / "full-current-surface-registration-458/run-001"
ENGINE_SHA = "18fbc383044c18f29d7fd7dccf822cc5b65957bc75aca0f4b6ac964ea27cb74a"
CACHE_RECEIPT_SHA = "06ca892d7d9e11e7aad807a679bc6c6c753c34cf811355664dc269d8c70d3f88"
CACHE_LAUNCH_SHA = "07d64e47447be8970f13ff89979fe2901bf35ba7c37923d8d7ebd4b607d52cd0"

spec = importlib.util.spec_from_file_location("cached_sidecars_460", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def option(argv, name):
    return Path(argv[argv.index(name) + 1])

def make_args():
    launch = json.loads((RUN_458 / "launch.json").read_text())
    argv = launch["argv"]
    return argparse.Namespace(
        outer=option(argv, "--outer"),
        inner=option(argv, "--inner"),
        deep=option(argv, "--deep"),
        parent=option(argv, "--parent"),
        poses=option(argv, "--poses"),
        parent_full86_weights=option(argv, "--parent-full86-weights"),
        out=None,
        maximum_parent_distance_m=float(argv[argv.index("--maximum-parent-distance-m") + 1]),
        tie_tolerance_m=float(argv[argv.index("--tie-tolerance-m") + 1]),
        reuse_oi_cache_dir=CACHE_DIR,
        reuse_oi_cache_launch=CACHE_LAUNCH,
        reuse_oi_cache_receipt_sha256=CACHE_RECEIPT_SHA,
        reuse_oi_cache_launch_sha256=CACHE_LAUNCH_SHA,
        reuse_oi_engine_sha256=ENGINE_SHA,
        cache_preflight_only=True,
        preflight_only=False,
        sample_check_count=0,
    )

def append_newline(source, destination):
    shutil.copyfile(source, destination)
    with destination.open("ab") as stream:
        stream.write(b"\n")

class RoleCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args = make_args()
        cls.cache = module.validate_cache_inputs(cls.args)
        cls.preflight = module.cache_preflight(cls.args)

    def test_real_full_input_pins_and_no_registration(self):
        self.assertEqual(self.preflight["status"], "hash_and_schema_preflight_passed_no_registration")
        self.assertEqual(self.preflight["registrationCalls"], 0)
        self.assertEqual(self.preflight["runtime_created"], 0)
        self.assertEqual(self.preflight["accepted_steps"], 0)
        self.assertFalse(self.preflight["physical_admission"])
        self.assertEqual(self.preflight["candidateCounts"]["O"]["nodes"], 56054)
        self.assertEqual(self.preflight["candidateCounts"]["I"]["nodes"], 102622)
        self.assertEqual(self.preflight["candidateCounts"]["D"]["nodes"], 67239)
        self.assertEqual(self.preflight["parentNodeCount"], 7233)
        self.assertEqual(self.preflight["parentTetCount"], 28632)
        self.assertEqual(len(self.preflight["parentSHA256"]), 5)
        self.assertEqual(self.preflight["bindingCount"], 86)

    def test_each_cached_oi_sidecar_is_byte_identical_to_458(self):
        output = RUN_458 / "sidecars"
        checks = []
        for role in ("O", "I"):
            for name in module.ROLE_FILES[role]:
                left = CACHE_DIR / name
                right = output / name
                self.assertTrue(left.is_file(), str(left))
                self.assertTrue(right.is_file(), str(right))
                self.assertEqual(module.sha256_file(left), module.sha256_file(right), name)
                self.assertTrue(filecmp.cmp(left, right, shallow=False), name)
                checks.append(name)
        self.assertEqual(len(checks), 8)

    def test_changed_oi_parent_pose_or_full86_input_fails_closed(self):
        with tempfile.TemporaryDirectory(prefix="role-cache-460-negative-") as tmp:
            root = Path(tmp)
            cases = []
            o = root / "changed-O.npz"
            append_newline(self.args.outer, o)
            changed = argparse.Namespace(**vars(self.args)); changed.outer = o
            cases.append((changed, "O candidate input SHA-256"))

            i = root / "changed-I.npz"
            append_newline(self.args.inner, i)
            changed = argparse.Namespace(**vars(self.args)); changed.inner = i
            cases.append((changed, "I candidate input SHA-256"))

            parent_base = root / "changed-parent.csv"
            append_newline(self.args.parent, parent_base)
            changed = argparse.Namespace(**vars(self.args)); changed.parent = parent_base
            cases.append((changed, "parent mass SHA-256"))

            pose = root / "changed-poses.json"
            append_newline(self.args.poses, pose)
            changed = argparse.Namespace(**vars(self.args)); changed.poses = pose
            cases.append((changed, "pose snapshot SHA-256"))

            weights = root / "changed-full86.csv"
            append_newline(self.args.parent_full86_weights, weights)
            changed = argparse.Namespace(**vars(self.args)); changed.parent_full86_weights = weights
            cases.append((changed, "parent full86 weights SHA-256"))

            for args, expected in cases:
                with self.subTest(expected=expected):
                    with self.assertRaises(module.CacheAdmissionError) as error:
                        module.validate_cache_inputs(args)
                    self.assertIn(expected, str(error.exception))

    def test_cache_receipt_pin_and_cached_output_hashes_are_enforced(self):
        changed_pin = argparse.Namespace(**vars(self.args))
        changed_pin.reuse_oi_cache_receipt_sha256 = "0" * 64
        with self.assertRaises(module.CacheAdmissionError):
            module.validate_cache_inputs(changed_pin)

        with tempfile.TemporaryDirectory(prefix="role-cache-460-copy-") as tmp:
            root = Path(tmp)
            cache = root / "cache"
            output = root / "copied"
            cache.mkdir()
            receipt = {"outputSHA256": {}}
            names = [name for role in ("O", "I") for name in module.ROLE_FILES[role]]
            names.extend(("node-parent-registration.csv", "donor-frames.csv"))
            for name in names:
                payload = ("fixture:" + name + "\n").encode()
                (cache / name).write_bytes(payload)
                receipt["outputSHA256"][name] = hashlib.sha256(payload).hexdigest()
            copied = module.copy_cached_role_artifacts(cache, output, receipt)
            self.assertEqual(len(copied), 8)
            for role in ("O", "I"):
                for name in module.ROLE_FILES[role]:
                    self.assertEqual((cache / name).read_bytes(), (output / name).read_bytes())
            (cache / "registered-O.csv").write_bytes(b"tampered\n")
            with self.assertRaises(module.CacheAdmissionError):
                module.copy_cached_role_artifacts(cache, root / "bad-copy", receipt)

def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RoleCacheTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    checks = {
        "schema": "registered-authoring-role-cache-checks.v1",
        "test_count": result.testsRun,
        "all_passed": True,
        "cache_receipt_sha256": CACHE_RECEIPT_SHA,
        "cache_launch_sha256": CACHE_LAUNCH_SHA,
        "engine_sha256": ENGINE_SHA,
        "cache_variant_sha256": module.sha256_file(SCRIPT),
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cache_output_dir": str(CACHE_DIR),
        "comparison_output_dir": str(RUN_458 / "sidecars"),
        "preflight": RoleCacheTests.preflight,
        "physical_admission": False,
        "runtime_created": 0,
        "accepted_steps": 0,
    }
    report = Path(os.environ.get("NUMI_ROLE_CACHE_TEST_OUTPUT_DIR", str(HERE / "role-cache-check-001")))
    report.mkdir(parents=True, exist_ok=False)
    (report / "tests.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n")
    print(json.dumps(checks, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
