#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
FIT_PATH=ROOT/"coupled-anchor-collapse-780-root"/"coupled_anchor_collapse_795_outer_role.py"
FIT_SHA="68124ffcffcc2bf5147de081e50f87739089c3be5c2370be8ae3b27bef4e922a"
def load(path,name):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
fit=load(FIT_PATH,"d014_fit_owner")
import d_anchor_fit_014 as dfit

def test_min_displacement_feasible_3d_fit_controls_all_caps():
    # A single well-shaped tetrahedron with a 0.30 mm endpoint edge. The
    # closest point satisfying both 250 um endpoint/L1 limits is off the
    # edge-free domain and must be returned as an actual 3D point.
    xkeep=np.asarray([0.0,0.0,0.0],np.float32)
    xdrop=np.asarray([0.00030,0.0,0.0],np.float32)
    a=np.asarray([0.0,0.001,0.0],np.float32)
    b=np.asarray([0.0,0.0,0.001],np.float32)
    c=np.asarray([0.001,0.0,0.0],np.float32)
    triangles=np.asarray([[c,a,b]],np.float32)
    points=np.asarray([xkeep,a,b,xdrop],np.float32)
    old_faces=np.asarray([[0,1,2]],np.int64)
    new_faces=old_faces.copy()
    volp=np.vstack((xkeep,triangles.reshape(-1,3))).astype(np.float32)
    volt=np.asarray([[0,1,2,3]],np.int64)
    det,q=fit.E.measurements(volp,volt)
    old_exact=fit.exact_sum(volp,volt)
    got=dfit.fit_single_D_3d(fit,xkeep,xdrop,triangles,float(det.sum()),old_exact,
        float(det.sum())*100.0,points,old_faces,new_faces,0,cap_m=250e-6)
    assert got["status"]=="PASS_D_3D_FEASIBLE_EDGE_FIT",got
    assert got["anchor_l1_m"]<=250e-6
    assert got["drop_to_anchor_m"]<=250e-6
    assert got["minimum_new_native_q"]>=1.01e-5
    assert got["candidate_count"]>0
    # The optimizer is not restricted to an edge parameterization: perturbing
    # a non-edge coordinate remains in the allowed 3D trust domain.
    z=np.asarray(got["anchor_point_f32"],np.float64)-xkeep.astype(np.float64)
    assert 1e-6 < np.linalg.norm(z[1:]) < 1e-4
    assert got["l1_cap_slack_m"]>=0 and got["removed_endpoint_cap_slack_m"]>=0
    assert got["minimum_linear_winding_slack"]>=-1e-12
    rhs,rows=dfit.l1_constraint_rows(250e-6,max(250e-6,np.max(np.linalg.norm(triangles-xkeep,axis=2))))
    assert rows.shape==(8,3) and rhs.shape==(8,)

def test_winding_constraints_are_linear_and_reject_missing_keep():
    points=np.asarray([[0,0,0],[0,1,0],[0,0,1],[1,0,0]],np.float32)
    off,mat=dfit.source_winding_system(points,[[0,1,2]],[[0,1,2]],0,points[0],1.0)
    assert off.shape==(1,) and mat.shape==(1,3)
    assert off[0]>0
    try:
        dfit.source_winding_system(points,[[0,1,2]],[[1,2,3]],0,points[0],1.0)
    except ValueError as exc:
        assert "exactly one moved keep" in str(exc)
    else:
        raise AssertionError("surviving face without unique keep vertex was accepted")

def test_l1_sign_planes_cover_each_octant():
    cap=250e-6;scale=1e-3
    rhs,rows=dfit.l1_constraint_rows(cap,scale)
    for vector in (np.asarray([.2,.1,.05]),np.asarray([-.2,.1,-.05])):
        assert np.max(rows@(vector*cap/np.sum(np.abs(vector)))-rhs)<=0
    assert len({tuple(x) for x in rows})==8

if __name__=="__main__":
    import pytest
    raise SystemExit(pytest.main([__file__,"-q"]))

