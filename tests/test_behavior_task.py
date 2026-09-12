"""Synthetic admission fixtures, never runtime or behavioral evidence."""
import copy
import hashlib
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from numilab_human import behavior_task as b


class BehaviorTaskTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.task={"schema":b.SCHEMA,"task":"standing","step_ns":25000,"target_speed_mps":None,"criteria":{
            "minimum_root_height_m":.7,"maximum_trunk_tilt_rad":.3,"maximum_planar_speed_mps":.1,
            "root_body_semantic_id":"pelvis","trunk_body_semantic_id":"pelvis","velocity_body_semantic_id":"pelvis",
            "velocity_observable":"body_com_linear_velocity","world_reference_origin_m":[0,0,0],
            "world_up_axis":[0,0,1],"world_forward_axis":[1,0,0],"trunk_up_axis_body":[0,0,1],
            "forbidden_contact_semantic_ids":["pelvis"]}}
        source={"archive_sha256":"a"*64,"revision":"fixture"}
        body={"id":3,"name":"pelvis","default_com_position_world_m":[0,0,1.1],
            "default_inertial_quaternion_world_xyzw":[0,0,0,1],"default_body_position_world_m":[0,0,1.0],
            "default_body_quaternion_world_xyzw":[0,0,0,1],"inertial_position_body_m":[0,0,.1],
            "inertial_quaternion_body_xyzw":[0,0,0,1]}
        self.export={"schema":"numi.human.myosim-mujoco-export.v1","source":source,"bodies":[body]}
        header=struct.pack('<8s10I32s',b'NHRIGID2',1,1,1,1,0,7,6,0,0,0,bytes.fromhex('a'*64))
        rigid=header+bytes(144+160+64*6+4*(7+6))+struct.pack('<I7f',0,0,0,1.1,0,0,0,1)
        (self.root/'rigid').write_bytes(rigid)
        record={"source_body_id":3,"name":"pelvis","core_body_index":0,
            "default_com_position_world_m":[0,0,1.1],"default_inertial_quaternion_world_xyzw":[0,0,0,1]}
        self.human={"schema":"numi.human.myosim-fullbody-reference.v1","source":source,
            "core_tree":{"source_body_records":[record]},"payloads":{"rigid":{"file":"rigid","bytes":len(rigid),"sha256":hashlib.sha256(rigid).hexdigest()}}}
    def run_author(self):
        for name,obj in [('task',self.task),('human',self.human),('export',self.export)]:
            (self.root/name).write_text(json.dumps(obj))
        return b.author_task(task_path=self.root/'task',human_manifest=self.root/'human',source_export=self.root/'export')
    def test_source_origin_and_determinism(self):
        raw,m=self.run_author();self.assertEqual(raw,self.run_author()[0]);self.assertEqual(len(raw),576)
        self.assertEqual(m['catalog']['pelvis']['source_origin_in_original_com_frame_m'],[0,0,-.1])
        self.assertFalse(m['qualification']['behavior_qualified'])
        self.assertEqual(b.HEADER.unpack_from(raw)[9],25000)
    def test_exact_fractional_microsecond_clock(self):
        self.task['step_ns']=12500;raw,_=self.run_author();self.assertEqual(b.HEADER.unpack_from(raw)[9],12500)
    def test_whole_human_com_denied(self):
        self.task['criteria']['velocity_observable']='whole_human_com_linear_velocity'
        with self.assertRaisesRegex(b.EvidenceError,'whole-Human'):self.run_author()
    def test_unresolved_semantic_denied(self):
        self.task['criteria']['root_body_semantic_id']='body0'
        with self.assertRaisesRegex(b.EvidenceError,'unresolved'):self.run_author()
    def test_forged_mapping_denied(self):
        self.human['core_tree']['source_body_records'][0]['core_body_index']=1
        with self.assertRaisesRegex(b.EvidenceError,'mapping'):self.run_author()
    def test_forged_origin_denied(self):
        self.export['bodies'][0]['inertial_position_body_m'][2]=.2
        with self.assertRaisesRegex(b.EvidenceError,'origin transform'):self.run_author()
    def test_forged_orientation_denied(self):
        self.export['bodies'][0]['inertial_quaternion_body_xyzw']=[0,0,1,0]
        with self.assertRaisesRegex(b.EvidenceError,'orientation transform'):self.run_author()
    def test_export_pose_drift_denied(self):
        self.export['bodies'][0]['default_com_position_world_m'][0]=.01
        with self.assertRaisesRegex(b.EvidenceError,'pose mismatch'):self.run_author()
    def test_rigid_drift_denied(self):
        p=self.root/'rigid';r=bytearray(p.read_bytes());r[250]=1;p.write_bytes(r)
        with self.assertRaisesRegex(b.EvidenceError,'hash'):self.run_author()
    def test_duplicate_source_id_denied(self):
        self.export['bodies'].append(copy.deepcopy(self.export['bodies'][0]))
        with self.assertRaisesRegex(b.EvidenceError,'duplicate'):self.run_author()
    def test_unknown_key_denied(self):
        self.task['criteria']['assistance']=False
        with self.assertRaisesRegex(b.EvidenceError,'criterion'):self.run_author()
    def test_nonfinite_denied(self):
        self.task['criteria']['minimum_root_height_m']=math.nan
        with self.assertRaisesRegex(b.EvidenceError,'nonfinite'):self.run_author()
    def test_duplicate_json_denied(self):
        self.run_author();p=self.root/'task';p.write_text(p.read_text()[:-1]+',"step_ns":25000}')
        with self.assertRaisesRegex(b.EvidenceError,'duplicate'):b.author_task(task_path=p,human_manifest=self.root/'human',source_export=self.root/'export')
    def test_unsafe_rigid_path_denied(self):
        self.human['payloads']['rigid']['file']='../rigid'
        with self.assertRaisesRegex(b.EvidenceError,'sibling'):self.run_author()
    def test_boolean_clock_denied(self):
        self.task['step_ns']=True
        with self.assertRaisesRegex(b.EvidenceError,'step_ns'):self.run_author()
    def test_nonorthogonal_axes_denied(self):
        self.task['criteria']['world_forward_axis']=[0,0,1]
        with self.assertRaisesRegex(b.EvidenceError,'orthogonal'):self.run_author()
    def test_empty_forbidden_denied(self):
        self.task['criteria']['forbidden_contact_semantic_ids']=[]
        with self.assertRaisesRegex(b.EvidenceError,'unique'):self.run_author()
    def test_walking_target(self):
        self.task['task']='walking';self.task['target_speed_mps']=1.5
        raw,_=self.run_author();self.assertEqual(b.NUMERICAL.unpack_from(raw,248+3*64)[-1],1.5)
    def test_large_scalar_denied(self):
        self.task['criteria']['minimum_root_height_m']=10**1000
        with self.assertRaisesRegex(b.EvidenceError,'finite'):self.run_author()
    def test_unhashable_task_denied(self):
        self.task['task']=[]
        with self.assertRaisesRegex(b.EvidenceError,'kind'):self.run_author()
    def test_uint32_body_extent_denied(self):
        p=self.root/'rigid';raw=bytearray(p.read_bytes());struct.pack_into('<I',raw,20,2**32-1);p.write_bytes(raw)
        self.human['payloads']['rigid']['sha256']=hashlib.sha256(raw).hexdigest()
        with self.assertRaisesRegex(b.EvidenceError,'shape'):self.run_author()
    def test_mapping_boolean_denied(self):
        self.human['core_tree']['source_body_records'][0]['core_body_index']=False
        with self.assertRaisesRegex(b.EvidenceError,'mapping'):self.run_author()
    def test_immutable_repeat_and_tamper(self):
        self.run_author();kw=dict(task_path=self.root/'task',human_manifest=self.root/'human',source_export=self.root/'export',output=self.root/'out')
        first=b.compile_task(**kw);self.assertEqual(first,b.compile_task(**kw))
        (self.root/'out'/'task.nhbhv').write_bytes(b'tamper')
        with self.assertRaisesRegex(b.EvidenceError,'immutable'):b.compile_task(**kw)

if __name__=='__main__':unittest.main()
