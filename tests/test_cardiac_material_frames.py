"""Focused synthetic controls; these fixtures never claim Rodero source identity."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from numilab_human import cardiac_material_frames as frames
from numilab_human.model import ImportError as HumanImportError


def fixture(directory: Path, fibres=None, sheets=None):
    directory.mkdir()
    fibres = fibres or [(1.00001, 0, 0), (0, 2, 0)]
    sheets = sheets or [(0.00001, 1, 0), (0, 0, 3)]
    configuration = {"schema": "synthetic_frame_fixture", "mesh": {"points": 5, "cells": 2}}
    config_sha = frames.sha256(frames.canonical(configuration))
    points = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (0,0,-1)]
    data = {
        "nodes.f64le": (b"".join(struct.pack("<3d", *v) for v in points), [5,3]),
        "tetrahedra.u32le": (struct.pack("<8I", 0,1,2,3,0,2,1,4), [2,4]),
        "labels.u32le": (struct.pack("<2I", 1,2), [2]),
        "fibres.f64le": (b"".join(struct.pack("<3d", *v) for v in fibres), [2,3]),
        "sheets.f64le": (b"".join(struct.pack("<3d", *v) for v in sheets), [2,3]),
        "source_reversed_cells.u32le": (b"", [0]),
        "boundary.u32le": (struct.pack("<18I", 1,2,3,0,3,2,0,1,3,2,1,4,0,4,1,0,2,4), [6,3]),
        "boundary_owners.u32le": (struct.pack("<6I", 0,0,0,1,1,1), [6]),
        "boundary_components.u32le": (struct.pack("<6I", *([0]*6)), [6]),
    }
    for key in ("rho", "phi", "z", "v"):
        data[f"uvc_{key}.f64le"] = (struct.pack("<5d", *([-10]*5)), [5])
    pins = {}
    for name, (raw, shape) in data.items():
        (directory/name).write_bytes(raw)
        pins[name] = (hashlib.sha256(raw).hexdigest(), shape, len(raw))
    manifest = {"schema": frames.wall.SCHEMA, "source_config": configuration,
        "source_config_sha256": config_sha, "buffers": {name: {"sha256": p[0], "shape": p[1], "bytes": p[2]}
            for name, p in pins.items()}, "topology": {"positive_oriented_tetrahedra": 2},
        "qualification": {"status": "synthetic_fixture_only", "physical_steps": 0}}
    raw = frames.canonical(manifest)
    (directory/"manifest.json").write_bytes(raw)
    return configuration, config_sha, pins, frames.sha256(raw)


def prepare(asset, output, contract, policy=frames.POLICY_ID):
    configuration, config_sha, pins, manifest_sha = contract
    return frames._prepare(asset, output, manifest_sha, policy, configuration, config_sha, pins,
                           "synthetic_fixture_only")


def contents(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir()}


class CardiacMaterialFrameTests(unittest.TestCase):
    def test_declared_correction_is_right_handed_and_retains_raw_vectors(self):
        fibre, sheet = [2.0, 0.0, 0.0], [0.25, 3.0, 0.0]
        before = copy.deepcopy((fibre, sheet))
        q, diagnostics = frames.convert_axes(fibre, sheet)
        self.assertEqual((fibre, sheet), before)
        self.assertEqual(q, (0,0,0,1))
        self.assertAlmostEqual(diagnostics["sheet_correction_radians"], math.atan2(0.25,3.0))
        self.assertEqual(diagnostics["fibre_direction_correction_radians"], 0)
        self.assertEqual(diagnostics["raw_fibre_norm_error"], 1)

    def test_quaternion_pi_rotations_and_sign_ties_are_canonical(self):
        cases = [((1,0,0),(0,-1,0),(1,0,0,0)),
                 ((-1,0,0),(0,1,0),(0,1,0,0)),
                 ((-1,0,0),(0,-1,0),(0,0,1,0)),
                 ((0,1,0),(1,0,0),(math.sqrt(.5),math.sqrt(.5),0,0))]
        for fibre, sheet, expected in cases:
            with self.subTest(expected=expected):
                q, _ = frames.convert_axes(fibre,sheet)
                for a,b in zip(q,expected): self.assertAlmostEqual(a,b)
                for v in q:
                    if v == 0: self.assertEqual(math.copysign(1,v),1)

    def test_varied_frames_reconstruct_source_directions(self):
        for i in range(1, 41):
            angle = i*0.113
            f = (math.cos(angle),math.sin(angle),0)
            s = (-math.sin(angle),math.cos(angle),0)
            q, diagnostic = frames.convert_axes(tuple(1.03*v for v in f),
                tuple(0.017*f[k]+0.94*s[k] for k in range(3)))
            matrix = frames.quaternion_matrix(q)
            for k in range(3):
                self.assertAlmostEqual(matrix[3*k], f[k], places=13)
                self.assertAlmostEqual(matrix[3*k+1], s[k], places=13)
            self.assertGreaterEqual(q[3],0)
            self.assertLess(diagnostic["quaternion_reconstruction_error"],1e-12)

    def test_finite_nonzero_and_angular_degeneracy_admission(self):
        for f,s in [((0,0,0),(0,1,0)),((1,0,0),(0,0,0)),((1,0,0),(2,0,0)),
                    ((1,0,0),(1,1e-10,0)),((float('nan'),0,0),(0,1,0)),
                    ((1,0,0),(float('inf'),1,0)),((1.7e308,1.7e308,1.7e308),(0,1,0)),
                    ((1,0),(0,1,0))]:
            with self.subTest(f=f,s=s), self.assertRaises(HumanImportError):
                frames.convert_axes(f,s)
        q,_ = frames.convert_axes((1e300,0,0),(0,1e-300,0))
        self.assertEqual(q,(0,0,0,1))

    def test_fixture_streaming_output_is_deterministic_and_input_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); asset=root/'asset'; contract=fixture(asset); original=contents(asset)
            a=prepare(asset,root/'one',contract); b=prepare(asset,root/'two',contract)
            self.assertEqual(a,b); self.assertEqual(contents(root/'one'),contents(root/'two'))
            self.assertEqual(original,contents(asset))
            self.assertEqual(set(contents(root/'one')),{"manifest.json",frames.QUATERNION_BUFFER})
            self.assertEqual(a['buffer']['bytes'],64)
            self.assertEqual(a['source_identity_record']['source_kind'],'synthetic_fixture_only')
            self.assertFalse(a['qualification']['anatomical_wall_simulation'])
            self.assertEqual(a,prepare(asset,root/'one',contract))
            self.assertEqual(a['source_identity_sha256'],frames.sha256(frames.canonical(a['source_identity_record'])))
            words=frames.native_identity_words(a['source_identity_sha256'])
            self.assertEqual(struct.pack('<4Q',*words).hex(),a['source_identity_sha256'])

    def test_rotations_preserve_source_order_and_are_native_unit_quaternions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); asset=root/'asset'; contract=fixture(asset)
            result=prepare(asset,root/'out',contract)
            rows=list(struct.iter_unpack('<4d',(root/'out'/frames.QUATERNION_BUFFER).read_bytes()))
            self.assertEqual(rows[0],(0,0,0,1))
            m=frames.quaternion_matrix(rows[1])
            for a,b in zip(m,(0,0,1,1,0,0,0,1,0)):self.assertAlmostEqual(a,b)
            self.assertTrue(all(abs(sum(v*v for v in q)-1)<=1e-12 for q in rows))
            self.assertEqual(result['diagnostics']['regions']['1']['count'],1)
            self.assertEqual(result['diagnostics']['regions']['2']['count'],1)

    def test_public_source_route_rejects_synthetic_manifest_even_with_matching_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); asset=root/'asset'; contract=fixture(asset)
            with self.assertRaises(HumanImportError):
                frames.prepare_material_frames(asset,root/'out',expected_manifest_sha256=contract[3],
                                                conversion_policy=frames.POLICY_ID)
            self.assertFalse((root/'out').exists())

    def test_missing_opt_in_and_wrong_manifest_pin_reject_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); asset=root/'asset'; contract=fixture(asset)
            for policy in ('', 'automatic', frames.POLICY_ID+'-modified'):
                with self.assertRaises(HumanImportError):prepare(asset,root/'out',contract,policy)
            for sha in ('0'*64, 'x'*64, contract[3].upper()):
                bad=(*contract[:3],sha)
                with self.assertRaises(HumanImportError):prepare(asset,root/'out',bad)
            self.assertFalse((root/'out').exists())

    def test_manifest_shape_inventory_and_source_contract_forgeries_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); asset=root/'asset'; contract=fixture(asset)
            original=json.loads((asset/'manifest.json').read_text())
            mutations=[lambda m:m['buffers']['fibres.f64le'].update(shape=[1,6]),
                lambda m:m['buffers']['fibres.f64le'].update(bytes=True),
                lambda m:m['buffers'].pop('sheets.f64le'),
                lambda m:m['buffers'].update({'../invented':{}}),
                lambda m:m['source_config']['mesh'].update(cells=3),
                lambda m:m['topology'].update(positive_oriented_tetrahedra=3)]
            for change in mutations:
                value=copy.deepcopy(original);change(value);raw=frames.canonical(value)
                (asset/'manifest.json').write_bytes(raw)
                with self.assertRaises(HumanImportError):prepare(asset,root/'out',(*contract[:3],frames.sha256(raw)))
            self.assertFalse((root/'out').exists())

    def test_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);asset=root/'asset';contract=fixture(asset)
            original=(asset/'manifest.json').read_bytes().strip()
            for raw in (b'{"schema":"forged",'+original[1:],
                        original[:-1]+b',"bad":NaN}', original[:-1]+b',"bad":1e999}'):
                (asset/'manifest.json').write_bytes(raw)
                with self.assertRaises((HumanImportError,ValueError)):
                    prepare(asset,root/'out',(*contract[:3],frames.sha256(raw)))

    def test_source_drift_and_coherent_rehash_do_not_change_source_pin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);asset=root/'asset';contract=fixture(asset)
            raw=struct.pack('<6d',2,0,0,0,2,0);(asset/'fibres.f64le').write_bytes(raw)
            with self.assertRaises(HumanImportError):prepare(asset,root/'out',contract)
            manifest=json.loads((asset/'manifest.json').read_text())
            manifest['buffers']['fibres.f64le']['sha256']=frames.sha256(raw)
            encoded=frames.canonical(manifest);(asset/'manifest.json').write_bytes(encoded)
            with self.assertRaises(HumanImportError):prepare(asset,root/'out',(*contract[:3],frames.sha256(encoded)))

    def test_bad_axes_leave_no_output_or_staging_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);asset=root/'asset';contract=fixture(asset,sheets=[(0,1,0),(0,3,0)])
            with self.assertRaisesRegex(HumanImportError,'source cell 1'):
                prepare(asset,root/'out',contract)
            self.assertEqual({p.name for p in root.iterdir()},{'asset'})

    def test_changed_output_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);asset=root/'asset';contract=fixture(asset);output=root/'out'
            prepare(asset,output,contract)
            data=(output/frames.QUATERNION_BUFFER).read_bytes()
            (output/frames.QUATERNION_BUFFER).write_bytes(b'x'+data[1:]); before=contents(output)
            with self.assertRaises(HumanImportError):prepare(asset,output,contract)
            self.assertEqual(before,contents(output))

    def test_source_edits_after_consumption_are_detected_before_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);asset=root/'asset';contract=fixture(asset);original=frames.convert_axes
            mutated=False
            def edit(f,s):
                nonlocal mutated
                q=original(f,s)
                if not mutated:
                    mutated=True;(asset/'fibres.f64le').write_bytes(struct.pack('<6d',2,0,0,0,2,0))
                return q
            with mock.patch.object(frames,'convert_axes',side_effect=edit), self.assertRaises(HumanImportError):
                prepare(asset,root/'out',contract)
            self.assertFalse((root/'out').exists())

    def test_manifest_edits_during_conversion_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);asset=root/'asset';contract=fixture(asset);original=frames.convert_axes
            def edit(f,s):
                (asset/'manifest.json').write_bytes((asset/'manifest.json').read_bytes()+b' ')
                return original(f,s)
            with mock.patch.object(frames,'convert_axes',side_effect=edit), self.assertRaises(HumanImportError):
                prepare(asset,root/'out',contract)
            self.assertFalse((root/'out').exists())

    def test_conversion_policy_edits_do_not_publish_misidentified_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);asset=root/'asset';contract=fixture(asset);original=frames.convert_axes
            policy=copy.deepcopy(frames.POLICY)
            def edit(f,s):
                result=original(f,s)
                frames.POLICY['minimum_sheet_sine']=2e-8
                return result
            try:
                with mock.patch.object(frames,'convert_axes',side_effect=edit), self.assertRaises(HumanImportError):
                    prepare(asset,root/'out',contract)
            finally:
                frames.POLICY.clear();frames.POLICY.update(policy)
            self.assertFalse((root/'out').exists())

    def test_symlink_inputs_outputs_and_output_inside_asset_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);asset=root/'asset';contract=fixture(asset)
            with self.assertRaises(HumanImportError):prepare(asset,asset/'derived',contract)
            (root/'out').symlink_to(asset,target_is_directory=True)
            with self.assertRaises(HumanImportError):prepare(asset,root/'out',contract)
            (asset/'fibres.f64le').rename(root/'raw');(asset/'fibres.f64le').symlink_to(root/'raw')
            with self.assertRaises(HumanImportError):prepare(asset,root/'elsewhere',contract)

    def test_source_pins_match_retained_case18_manifest(self):
        source=frames.wall.ROOT/'Docs/media/cardiac-wall-anatomy-20260912/manifest.json'
        manifest=json.loads(source.read_text())
        actual={name:(p['sha256'],p['shape'],p['bytes']) for name,p in manifest['buffers'].items()}
        self.assertEqual(frames.SOURCE_BUFFERS,actual)

    def test_cli_requires_explicit_conversion_policy(self):
        with self.assertRaises(SystemExit) as error:frames.main(['--asset','x','--output','y','--asset-manifest-sha256','0'*64])
        self.assertEqual(error.exception.code,2)
        command=frames.wall.ROOT/'.numi/commands/human-cardiac-material-frames'
        result=subprocess.run([str(command),'--numi-describe'],text=True,capture_output=True,check=True)
        self.assertIn('explicit',result.stdout)


if __name__ == '__main__':
    unittest.main()
