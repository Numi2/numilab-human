"""Fit the frozen trial space to an independent full-order native endpoint."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.sparse import diags,load_npz
from scipy.sparse.linalg import splu
from compile_connected_joint import digest
from supported_material_admission import geom
from test_source_embedded_motion import audit_field
from audit_material_walls import run as audit_walls


def run(work):
    stage=work/'witness-material';vol=stage/'volume';trial=work/'half-step-material-08'
    cfg=json.loads((trial/'launch.json').read_bytes())
    execution=json.loads((trial/'execution.json').read_bytes())
    raw=stage/'source-embedded-physical-rows.npz'
    configured=json.loads((stage/'source-embedding-motion.json').read_bytes())['refined']
    if cfg['held_out_basis_sha256']!=digest(raw) or digest(raw)!=configured['sparse_matrix']:
        raise ValueError('Holdout basis changed')
    result=dict(frozen_basis=digest(raw),launch=digest(trial/'launch.json'),exit=execution['exit'],wall_s=execution['wall_s'])
    path=stage/'holdout-motion.json'
    if execution['exit'] or not (trial/'physical-accepted.node').is_file():
        result.update(native_completed=False,scope='Full-order run did not export an accepted physical endpoint.')
        failure=trial/'failure.json'
        if failure.exists():
            f=json.loads(failure.read_bytes());result['diagnostics']=f['matter_diagnostics']
        path.write_text(json.dumps(result,indent=2)+'\n')
        return
    receipt=trial/'native-traces/accepted-geometry/step-1.receipt.json'
    with (trial/'native-traces/resting-coupled.csv').open()as stream:
        trace=list(csv.DictReader(stream))
    frame=json.loads(receipt.read_bytes())
    dt=float(cfg['argv'][cfg['argv'].index('--muscle-step-seconds')+1])
    if len(trace)!=1 or int(trace[0]['step'])!=1 or abs(float(trace[0]['time_s'])-dt)>dt*1e-6 or frame['accepted_step']!=1 or cfg['environment']['NM_VASCULAR_STEP_EXTERNAL_PRESSURE']!='1':
        raise ValueError('Inconsistent native endpoint or clock')
    load=lambda p:np.loadtxt(p,skiprows=1,usecols=(1,2,3))
    x=load(vol/'current.node');actual=load(trial/'physical-accepted.node');u=actual-x
    B=load_npz(raw);mass=np.fromfile(vol/'node-mass.f64','<f8')
    if not (B.shape[0]==len(u)==len(mass)):
        raise ValueError('Basis/material row mismatch')
    # The Galerkin mass is unregularized. Each column is equilibrated for this
    # diagnostic solve; the velocity projection uses the exact B^T M right hand side.
    M=B.T @ B.multiply(mass[:,None]);scale=1/np.sqrt(M.diagonal());D=diags(scale)
    LU=splu((D @ M @ D).tocsc());rhs=B.T @ (mass[:,None]*u)
    sol=scale[:,None]*LU.solve(scale[:,None]*rhs)
    v=B @ sol
    # Unreduced rows are exact coordinate identities. Copy their source values
    # directly to avoid arithmetical noise on nearly stationary elements.
    fine_cols=B.shape[0]-configured['embedded_nodes']
    retained=np.arange(B.shape[1]-fine_cols,B.shape[1])
    identity=B.tocsc()[:,retained]
    if np.any(np.diff(identity.indptr)!=1) or np.any(identity.data!=1):
        raise ValueError('Missing independent coordinates')
    v[identity.indices]=u[identity.indices]
    z=np.loadtxt(vol/'mesh.ele',skiprows=1,usecols=(1,2,3,4,5),dtype='i4')
    cells=z[:,:4]-1
    reg=np.loadtxt(vol/'current-child.labels.csv',skiprows=1,delimiter=',',usecols=(3,),dtype='i4')
    field=audit_field(x,u,v,cells,z[:,4],reg)
    exact_volume,exact_shape=geom(actual,cells)
    alt_volume,alt_shape=geom(x+v,cells)
    material={}
    for role,tag in [('skin',15),('fat',16)]:
        mask=z[:,4]==tag
        material[role]=dict(native_inverted=int(np.sum(exact_volume[mask]<=0)),native_low_shape=int(np.sum(exact_shape[mask]<1e-5)),added_inversions=int(np.sum((alt_volume[mask]<=0)&(exact_volume[mask]>0))),added_shape_failures=int(np.sum((alt_shape[mask]<1e-5)&(exact_shape[mask]>=1e-5))))
    base=next(p for p in work.parents if p.name=='human-integrated-gaps-20261104')
    lib=base/'resting-clearance-and-atrium-017/extended-equilibrium-027/connected-arm-volume-043/boundary_pairs.dylib'
    w=np.load(vol/'material-walls.npz');walls=[w[r]for r in 'OIDC']
    captures={}
    for name,data in [('physical',actual),('projected',x+v)]:
        report=stage/('holdout-'+name+'-walls.json')
        audit_walls(data,walls,report,lib)
        count=json.loads(report.read_bytes())['pair_counts']
        captures[name]=dict(sum_pairs=sum(map(sum,count)),sha256=digest(report))
    rms={k:max((a['rms_error']/a['rms_reference'] if a['rms_reference']>0 else float(a['rms_error']>0))for a in s['regions'])for k,s in field.items()}
    result.update(native_completed=True,accepted_s=float(trace[0]['time_s']),native_nodes_sha256=digest(trial/'physical-accepted.node'),receipt=digest(receipt),mass_projection_residual_relative=np.linalg.norm(M @ sol-rhs)/np.linalg.norm(rhs),incremental_strain_and_validity=field,material=material,wall_audits=captures,worst_regional_relative_incremental_strain_error=rms,passed_motion_comparison=all(v<=.05 for v in rms.values()) and all(not (s['added_inversions'] or s['added_shape_failures'])for s in material.values()) and captures['projected']['sum_pairs']==captures['physical']['sum_pairs']==0,scope='Unseen half-interval full-order mechanical release. Fixed-basis projection is a kinematic comparison. Forces, accepted dynamic advancement of the reduced space, and sustained speed require their own tests.')
    path.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    run(Path(__file__).resolve().parent)
