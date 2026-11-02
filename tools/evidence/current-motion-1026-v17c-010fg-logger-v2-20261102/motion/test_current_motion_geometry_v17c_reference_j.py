from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
VERIFIER=H/"current-motion-1026-verifier/verify_current_motion_geometry_1026_v17c.py"
spec=importlib.util.spec_from_file_location("motion_v17c",VERIFIER)
motion=importlib.util.module_from_spec(spec)
spec.loader.exec_module(motion)

def det4(points):
    b,c,d=points[1:].astype(np.float64)-points[0].astype(np.float64)
    return float(b[0]*(c[1]*d[2]-c[2]*d[1])-b[1]*(c[0]*d[2]-c[2]*d[0])+b[2]*(c[0]*d[1]-c[1]*d[0]))

def meta(rest,ih,il,ah,al,im=1.0,m=1.0):
    out={"initial_reference_mode":float(im),"reference_mode":float(m)}
    for axis,a in enumerate("xyz"):
        out["initial_rest_"+a]=float(rest[axis]);out["rest_"+a]=float(rest[axis])
        out["initial_reference_u_hi_"+a]=float(ih[axis]);out["initial_reference_u_lo_"+a]=float(il[axis])
        out["reference_u_hi_"+a]=float(ah[axis]);out["reference_u_lo_"+a]=float(al[axis])
    return out

def capture_tuple(rest,hi,lo,logged):
    rest32=rest.astype(np.float32)
    p1=(rest32.astype(np.float64)+hi.astype(np.float64)+lo.astype(np.float64)).astype(np.float32)
    z=np.zeros_like(rest32,dtype=np.float32)
    nodes={i:(rest32[i],p1[i],meta(rest32[i],z[i],z[i],hi[i],lo[i])) for i in range(4)}
    tets={0:(np.asarray([0,1,2,3],np.int64),float(logged))}
    begin={"version":"2","nodes":"4","tetrahedra":"1","steps":"2","endpoint_time_s":"0.002"}
    mesh={"joint":rest32,"expected":np.asarray([[0,1,2,3]],np.int64),"byrole":np.asarray(["Skin"],dtype="U4")}
    return (begin,nodes,tets,"boundary"),mesh,p1

class ReferenceDisplacementTerminalReplay(unittest.TestCase):
    def test_sub_ulp_rigid_motion_uses_split_words_not_projected_xyz(self):
        rest=np.array([[1048576.,-1048576.,1048576.],[1048577.875,-1048576.,1048576.],
                       [1048577.125,-1048575.875,1048576.],[1048576.125,-1048575.5,1048576.125]],dtype=np.float64)
        rotation=np.array([[.9992409934673947,-.026322977249176597,.028714766985532002],
                           [.026338852939785922,.9996530569654942,-.00017471507859639468],
                           [-.028700205576099935,.00093089649354249,.9995876307916219]])
        translation=np.array([.0033818074249490847,.004659219563480099,.00006010697130598597])
        displacement=(rest@rotation+translation)-rest
        hi=displacement.astype(np.float32);lo=(displacement-hi.astype(np.float64)).astype(np.float32)
        rest32=rest.astype(np.float32);physical=rest32.astype(np.float64)+hi.astype(np.float64)+lo.astype(np.float64)
        initial_det=det4(rest32);accepted_det=det4(physical);projected=physical.astype(np.float32).astype(np.float64)
        xyz_j=det4(projected)/initial_det
        self.assertLess(xyz_j,0.0,"fixture must expose projected-F32 inversion")
        self.assertGreater(accepted_det/initial_det,0.99)
        data,mesh,p1=capture_tuple(rest32,hi,lo,accepted_det/initial_det)
        replay=motion.terminal_j(data,mesh)
        self.assertAlmostEqual(replay["minimum_J_by_material"]["Skin"]["minimum_J"],accepted_det/initial_det,delta=1e-12)
        self.assertEqual(replay["reference_displacement_mode_tetrahedra"],1)
        self.assertEqual(replay["projected_xyz_mode_tetrahedra"],0)
        self.assertEqual(replay["terminal_J_replay_basis"],"reference_rest_plus_initial_accepted_hi_lo_for_mode1_projected_xyz_for_mode0")
        self.assertTrue(np.array_equal(mesh["joint"],np.stack([data[1][i][0] for i in range(4)])))
        self.assertTrue(np.array_equal(p1,np.stack([data[1][i][1] for i in range(4)])))


    def test_logger_ten_significant_digit_fields_roundtrip_to_exact_f32_words(self):
        rest=np.array([[1048576.,-1048576.,1048576.],[1048577.875,-1048576.,1048576.],
                       [1048577.125,-1048575.875,1048576.],[1048576.125,-1048575.5,1048576.125]],dtype=np.float64)
        rotation=np.array([[.9992409934673947,-.026322977249176597,.028714766985532002],
                           [.026338852939785922,.9996530569654942,-.00017471507859639468],
                           [-.028700205576099935,.00093089649354249,.9995876307916219]])
        translation=np.array([.0033818074249490847,.004659219563480099,.00006010697130598597])
        displacement=(rest@rotation+translation)-rest
        hi=displacement.astype(np.float32);lo=(displacement-hi.astype(np.float64)).astype(np.float32)
        rest32=rest.astype(np.float32);physical=rest32.astype(np.float64)+hi.astype(np.float64)+lo.astype(np.float64)
        native_j=det4(physical)/det4(rest32)
        data,mesh,p1=capture_tuple(rest32,hi,lo,native_j)
        begin,nodes,tets,boundary=data
        def f32(v): return format(float(np.float32(v)), '.10g')
        lines=['resting_FEM_endpoint_capture=begin version=2 nodes=4 tetrahedra=1 steps=2 endpoint_time_s=0.002']
        saw_decimal_rounding=False
        for i in range(4):
            initial,accepted,md=nodes[i]
            fields={'global':i}
            for axis,a in enumerate('xyz'):
                fields['initial_'+a]=initial[axis];fields['accepted_'+a]=accepted[axis]
            fields.update(md)
            items=[]
            for k,v in fields.items():
                rendered=f32(v) if k != 'global' else str(v)
                if k in md and float(rendered)!=float(np.float32(v)):
                    saw_decimal_rounding=True
                items.append(f'{k}={rendered}')
            lines.append('resting_FEM_node,'+','.join(items))
        corners,j=tets[0]
        lines.append('resting_FEM_tet,global=0,'+','.join(f'n{k}={int(v)}' for k,v in enumerate(corners))+',J='+format(float(j),'.10g'))
        lines.append('resting_FEM_endpoint_capture=end')
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'native-capture-10g.stdout';path.write_text('\n'.join(lines)+'\n')
            parsed=motion.capture(path)
            for i in range(4):
                got=parsed[1][i][2]
                for key,value in nodes[i][2].items():
                    self.assertEqual(np.float32(got[key]).view(np.uint32),np.float32(value).view(np.uint32),key)
            self.assertTrue(saw_decimal_rounding,'fixture should exercise decimal-to-F32 recovery')
            replay=motion.terminal_j(parsed,mesh)
            self.assertAlmostEqual(replay['minimum_J_by_material']['Skin']['minimum_J'],native_j,delta=2e-8)
            self.assertEqual(replay['reference_mode_logger_decimal_roundtrip'],'parse_decimal_then_round_to_binary32_before_widening')


    def test_full_report_assembly_includes_reference_word_replay_basis(self):
        rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        z=np.zeros_like(rest);data,mesh,_=capture_tuple(rest,z,z,1.0)
        replay,steps,record=motion.build_terminal_j_report_record(data,mesh,{
            "expected_steps":2,"minimum_J_global":1.0,"maximum_J_global":1.0
        })
        self.assertIsNotNone(replay);self.assertEqual(steps,2)
        self.assertEqual(record["terminal_J_replay_basis"],"reference_rest_plus_initial_accepted_hi_lo_for_mode1_projected_xyz_for_mode0")
        self.assertEqual(record["reference_mode_logger_decimal_roundtrip"],"parse_decimal_then_round_to_binary32_before_widening")
        self.assertEqual(record["reference_displacement_mode_tetrahedra"],1)

    def test_compact_report_assembly_labels_native_only_scope(self):
        compact=(None,{}, {}, "boundary")
        replay,steps,record=motion.build_terminal_j_report_record(compact,{"expected":np.empty((0,4),np.int64)},
            {"expected_steps":2,"minimum_J_global":0.95,"maximum_J_global":1.02})
        self.assertIsNone(replay);self.assertEqual(steps,2)
        self.assertEqual(record["terminal_J_replay_basis"],"native_compact_all_tet_audit_only_no_independent_endpoint_word_replay")
        self.assertEqual(record["reference_mode_logger_decimal_roundtrip"],"NOT_CAPTURED_COMPACT_MODE")
        self.assertFalse(record["independent_node_level_replay"])

    def test_missing_mixed_and_invalid_metadata_reject(self):
        rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32);z=np.zeros_like(rest)
        data,mesh,_=capture_tuple(rest,z,z,1.0)
        nodes=dict(data[1]);nodes[2]=(nodes[2][0],nodes[2][1],None)
        with self.assertRaises(ValueError):motion.terminal_j((data[0],nodes,data[2],data[3]),mesh)
        nodes=dict(data[1]);m=dict(nodes[0][2]);m["reference_mode"]=0.0;nodes[0]=(nodes[0][0],nodes[0][1],m)
        with self.assertRaises(ValueError):motion.terminal_j((data[0],nodes,data[2],data[3]),mesh)
        nodes=dict(data[1]);m=dict(nodes[0][2]);m["reference_mode"]=2.0;nodes[0]=(nodes[0][0],nodes[0][1],m)
        with self.assertRaises(ValueError):motion.terminal_j((data[0],nodes,data[2],data[3]),mesh)
        nodes=dict(data[1]);m=dict(nodes[0][2]);del m["initial_reference_u_hi_x"];nodes[0]=(nodes[0][0],nodes[0][1],m)
        with self.assertRaises((ValueError,KeyError)):motion.terminal_j((data[0],nodes,data[2],data[3]),mesh)

    def test_true_reference_mode_inversion_rejects(self):
        rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        hi=np.zeros_like(rest);lo=np.zeros_like(rest);hi[3,2]=-2.0
        determinant=det4(rest.astype(np.float64)+hi.astype(np.float64)+lo.astype(np.float64))
        data,mesh,_=capture_tuple(rest,hi,lo,determinant/det4(rest))
        with self.assertRaises(ValueError):motion.terminal_j(data,mesh)

    def test_legacy_xyz_replay_remains_supported(self):
        p0=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32)
        p1=p0.copy();p1[1]*=1.5;p1[2]*=1.5;p1[3]*=1.5
        d0=det4(p0);d1=det4(p1);j=d1/d0
        nodes={i:(p0[i],p1[i]) for i in range(4)}
        tets={0:(np.array([0,1,2,3],np.int64),j)}
        begin={"nodes":"4","tetrahedra":"1","steps":"2","endpoint_time_s":"0.002"}
        mesh={"joint":p0,"expected":np.array([[0,1,2,3]],np.int64),"byrole":np.array(["Skin"],dtype="U4")}
        replay=motion.terminal_j((begin,nodes,tets,"boundary"),mesh)
        self.assertEqual(replay["reference_displacement_mode_tetrahedra"],0)
        self.assertEqual(replay["terminal_J_replay_basis"],"projected_xyz_legacy_log")
        self.assertAlmostEqual(replay["minimum_J_by_material"]["Skin"]["minimum_J"],j,delta=1e-14)

    def test_capture_parser_requires_complete_v2_endpoint_metadata(self):
        rest=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=np.float32);z=np.zeros_like(rest)
        lines=["resting_FEM_endpoint_capture=begin version=2 nodes=4 tetrahedra=1 steps=2 endpoint_time_s=0.002"]
        for i in range(4):
            md=meta(rest[i],z[i],z[i],z[i],z[i])
            fields={"global":i,"initial_x":rest[i,0],"initial_y":rest[i,1],"initial_z":rest[i,2],
                    "accepted_x":rest[i,0],"accepted_y":rest[i,1],"accepted_z":rest[i,2]}
            fields.update(md);lines.append("resting_FEM_node,"+",".join(f"{k}={v}" for k,v in fields.items()))
        lines.append("resting_FEM_tet,global=0,n0=0,n1=1,n2=2,n3=3,J=1")
        lines.append("resting_FEM_endpoint_capture=end")
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"capture.stdout";path.write_text("\n".join(lines)+"\n")
            begin,nodes,tets,_=motion.capture(path)
            self.assertEqual(begin["version"],"2");self.assertEqual(nodes[0][2]["initial_reference_mode"],1.0)
            self.assertEqual(len(tets),1)
            lines[1]=lines[1].replace("initial_reference_u_hi_x=","removed_initial_reference_u_hi_x=")
            path.write_text("\n".join(lines)+"\n")
            with self.assertRaises(ValueError):motion.capture(path)

if __name__=="__main__":unittest.main()
