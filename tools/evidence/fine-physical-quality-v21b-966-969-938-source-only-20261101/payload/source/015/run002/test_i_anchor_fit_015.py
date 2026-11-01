#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FIT_PATH=ROOT/"coupled-anchor-collapse-780-root"/"coupled_anchor_collapse_795_outer_role.py"
FIT_SHA="68124ffcffcc2bf5147de081e50f87739089c3be5c2370be8ae3b27bef4e922a"
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
fit=load(FIT_PATH,"i015_fit_owner")
import i_anchor_fit_015 as ipair

def fixture():
    keep=np.asarray([0,0,0],np.float32)
    drop=np.asarray([0.0002,0,0],np.float32)
    a=np.asarray([0.001,0,0],np.float32)
    b=np.asarray([0,0.001,0],np.float32)
    c=np.asarray([0,0,0.001],np.float32)
    a2=np.asarray([-0.001,0,0],np.float32)
    b2=np.asarray([0,0,0.001],np.float32)
    c2=np.asarray([0,0.001,0],np.float32)
    triangles=np.asarray([[a,b,c],[a2,b2,c2]],np.float32)
    volp=np.vstack((keep,triangles.reshape(-1,3),drop)).astype(np.float32)
    volt=np.asarray([[0,1,2,3],[0,4,5,6]],np.int64)
    det,q=fit.E.measurements(volp,volt)
    src=np.asarray([keep,[0,0.001,0],[0,0,0.001],drop],np.float32)
    oldf=np.asarray([[3,1,2]],np.int64)
    newf=np.asarray([[0,1,2]],np.int64)
    return keep,drop,triangles,volp,volt,det,src,oldf,newf

def test_paired_fit_enforces_exact_skin_fat_cancellation_and_caps():
    keep,drop,tri,volp,volt,det,src,oldf,newf=fixture()
    old_exact=fit.exact_sum(volp,volt)
    out=ipair.fit_single_I_3d(fit,keep,drop,tri,float(det.sum()),old_exact,
        float(det.sum())*100.0,src,oldf,newf,0,cap_m=250e-6)
    assert out["status"]=="PASS_I_3D_PAIRED_EDGE_FIT",out
    assert out["role"]=="I" and out["paired_exact_signed_volume_delta_det6_integer"]=="0"
    assert out["exact_det_delta_integer"]=="0"
    assert out["solver_success"] is True
    assert np.array_equal(np.asarray(out["anchor_point_f32"],np.float32),keep)
    # SLSQP's continuous residual is nonzero but far below F32 geometry scale;
    # the original anchor must still be tested explicitly.
    assert np.any(np.asarray(out["continuous_anchor_world_m"],np.float64)!=0.0)
    assert out["candidate_count"]>=1
    assert out["minimum_new_native_q"]>=1.01e-5
    assert out["anchor_l1_m"]<=250e-6 and out["drop_to_anchor_m"]<=250e-6
    assert out["minimum_linear_winding_slack"]>=-1e-12

def test_signed_paired_delta_control_cancels_x_but_not_y_motion():
    keep,drop,tri,volp,volt,det,src,oldf,newf=fixture()
    before=fit.exact_sum(volp,volt)
    moved_x=volp.copy();moved_x[0,0]=np.float32(0.00005)
    moved_y=volp.copy();moved_y[0,1]=np.float32(0.00005)
    assert fit.exact_sum(moved_x,volt)-before==0
    assert fit.exact_sum(moved_y,volt)-before!=0

def test_exact_delta_gate_rejects_any_nonzero_integer_residual():
    assert ipair.paired_exact_delta_ok(0)
    assert not ipair.paired_exact_delta_ok(1)
    assert not ipair.paired_exact_delta_ok(-1)

def test_pair_fit_does_not_run_as_single_role():
    class Fake:
        targets=("Fat",)
    out=ipair.fit_edge(Fake(),0,1,fit,250e-6)
    assert out["status"]=="REJECT_I_PAIR_TARGETS"



def test_3d_fit_moves_anchor_to_repair_sliver_with_exact_paired_cancellation():
    # The paired face areas are exact opposites, so the total signed det6 sum
    # is invariant under x motion. One long, thin tet starts below q_min while
    # the second remains well conditioned. The 3D fit must move off the endpoint
    # enough to clear the unchanged 1.01*q margin without losing exact volume.
    keep=np.asarray([0,0,0],np.float32)
    drop=np.asarray([2**-13,0,0],np.float32)
    a=np.asarray([2**-16,0,0],np.float32)
    b=np.asarray([2**-16,2**-7,0],np.float32)
    c=np.asarray([2**-16,0,2**-16],np.float32)
    a2=np.asarray([-2**-11,0,0],np.float32)
    b2=np.asarray([-2**-11,0,2**-12],np.float32)
    c2=np.asarray([-2**-11,2**-11,0],np.float32)
    triangles=np.asarray([[a,b,c],[a2,b2,c2]],np.float32)
    before_points=np.vstack((keep,triangles.reshape(-1,3),drop)).astype(np.float32)
    tets=np.asarray([[0,1,2,3],[0,4,5,6]],np.int64)
    old_det,q0=fit.E.measurements(before_points,tets)
    assert q0[0] < ipair.NATIVE_Q and q0[1] > ipair.NATIVE_Q
    source=np.asarray([keep,[0,2**-10,0],[0,0,2**-10],drop],np.float32)
    old_faces=np.asarray([[3,1,2]],np.int64);new_faces=np.asarray([[0,1,2]],np.int64)
    old_exact=fit.exact_sum(before_points,tets)
    out=ipair.fit_single_I_3d(fit,keep,drop,triangles,float(old_det.sum()),old_exact,
        float(old_det.sum())*100.0,source,old_faces,new_faces,0,cap_m=250e-6)
    assert out["status"]=="PASS_I_3D_PAIRED_EDGE_FIT",out
    anchor=np.asarray(out["anchor_point_f32"],np.float32)
    assert np.linalg.norm(anchor.astype(np.float64)-keep.astype(np.float64))>1e-6
    assert out["anchor_l1_m"]<=250e-6 and out["drop_to_anchor_m"]<=250e-6
    after_points=before_points.copy();after_points[0]=anchor
    det,q1=fit.E.measurements(after_points,tets)
    assert np.all(det>0) and np.all(det/6>ipair.MIN_CELL_VOLUME_M3)
    assert np.all(q1>=ipair.NATIVE_Q*1.01) and np.all(fit.E.M.passes(det,q1))
    assert fit.exact_sum(after_points,tets)-old_exact==0
    assert np.all(fit.J.score(source,new_faces)>=fit.J.GATE)
