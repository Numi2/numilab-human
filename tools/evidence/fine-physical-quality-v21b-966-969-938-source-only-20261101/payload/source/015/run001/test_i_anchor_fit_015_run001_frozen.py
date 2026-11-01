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
    assert np.array_equal(np.asarray(out["anchor_point_f32"],np.float32),keep)
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

