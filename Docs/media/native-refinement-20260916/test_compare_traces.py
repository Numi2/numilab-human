"""Synthetic negative tests for the offline comparison, never physical evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from compare_traces import compare, orientation_delta


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.folders = []
        for name, ns, steps in [('100us', 100000, 64), ('50us', 50000, 128),
                                ('25us', 25000, 256), ('12p5us', 12500, 512)]:
            directory = self.root / name
            logs = directory / name
            logs.mkdir(parents=True)
            metrics = {'myosim_articulated_mechanics': 'ok', 'rendering_performed': 'false',
                'persistent_root_assistance': 'none', 'core_bodies': '157',
                'visual_coverage_qualified': 'false', 'persistent_metal_horizon': 'true',
                'compiled_stand_recruited_muscles': '416',
                'compiled_stand_balanced': 'true', 'persistent_completed_steps': str(steps),
                'muscle_step_count': str(steps), 'muscle_step_seconds': str(ns*1e-9),
                'stand_deterministic_replay': 'bitwise',
                'persistent_passive_joint_law': 'current_state_linear_backward_euler',
                'source_support_metal_device': 'SYNTHETIC_TEST_NOT_GPU',
                'compiled_stand_normalized_residual_rms': '0',
                'compiled_stand_total_support_force_n': '900',
                'persistent_max_acceleration': '0', 'persistent_max_penetration_m': '0'}
            q = [0.0]*129
            q[6] = 1.0
            trace = {'schema': 'numi.human.persistent-stand-trace.v4',
                'endpoint_equivalent': 'bitwise', 'endpoint_max_q_delta': 0, 'endpoint_max_v_delta': 0,
                'samples': [{'step': i, 'time_seconds': i*ns*1e-9, 'q': q, 'v': [0.0]*128,
                             'normal_impulse': 0 if i == 0 else 900*ns*1e-9} for i in range(steps+1)]}
            out = logs / 'stdout.txt'
            out.write_text(' '.join(k+'='+v for k, v in metrics.items()) + '\n'
                           + 'persistent_stand_trace='+json.dumps(trace)+'\n')
            err = logs / 'stderr.txt'
            err.write_text('')
            case = {'name': name, 'timestep_nanoseconds': ns, 'steps': steps,
                'exit_code': 0, 'status': 'process_completed', 'source_artifacts_unchanged': True,
                'stdout_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
                'stderr_sha256': hashlib.sha256(err.read_bytes()).hexdigest(), 'metrics': metrics}
            record = {'schema': 'numi.human.hosted-bounded-release.v1', 'native_commit': 'a'*40,
                'input_commit': 'b'*40, 'synthetic_fixture': True,
                'initialization': 'canonical_numi_human_stand_authored_support_stance',
                'rendering_requested': False, 'cases': [case],
                'artifacts': {key: {'sha256': 'c'*64} for key in
                    ['rigid','muscle','tendon','support_contact','joint_equalities','launcher','binary']}}
            (directory/'execution.json').write_text(json.dumps(record))
            self.folders.append(directory)

    def tearDown(self):
        self.temporary.cleanup()

    def test_constant_trajectories_compare_without_qualification(self):
        result = compare(self.folders)
        self.assertFalse(result['force_convergence'])
        self.assertFalse(result['sustained_standing'])
        for row in result['comparisons']:
            self.assertEqual(row['generalized_velocity_max_mixed_units'], 0)
            self.assertLess(row['normal_reaction_common_interval_max_delta_n'], 1e-10)

    def test_quaternion_angle_and_sign(self):
        import math
        self.assertAlmostEqual(orientation_delta([0,0,0,1],[0,0,math.sin(.2),math.cos(.2)]), .4)
        self.assertEqual(orientation_delta([0,0,0,1],[0,0,0,-1]), 0)

    def test_reject_incomplete_or_duplicate_grids(self):
        with self.assertRaises(ValueError): compare(self.folders[:3])
        with self.assertRaises(ValueError): compare([self.folders[0]]*4)

    def test_reject_tampered_output(self):
        path = self.folders[0]/'100us/stdout.txt'
        path.write_text(path.read_text()+'tampered')
        with self.assertRaises(ValueError): compare(self.folders)

    def test_reject_mixed_source_and_assistance(self):
        path = self.folders[0]/'execution.json'
        original = json.loads(path.read_text())
        changed = copy.deepcopy(original)
        changed['native_commit'] = 'd'*40
        path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError): compare(self.folders)
        changed = copy.deepcopy(original)
        changed['cases'][0]['metrics']['persistent_root_assistance'] = 'enabled'
        path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError): compare(self.folders)

    def test_reject_duplicate_json_key(self):
        path = self.folders[0]/'execution.json'
        text = path.read_text()
        path.write_text(text[:-1]+',"native_commit":"'+'a'*40+'"}')
        with self.assertRaises(ValueError): compare(self.folders)


if __name__ == '__main__':
    unittest.main()
