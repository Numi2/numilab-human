"""Screen the entire installed skin/fat operator for material-driven release, using both laws and nodal mass."""
import json
import sys
from pathlib import Path
import numpy as np
from fit_dorsal_material import inputs
from fit_posterior_equilibrium import materials
from material_context import sha

def element_blocks(X,Y,props):
    edge=X[:,1:]-X[:,:1];vol=np.linalg.det(edge)/6
    D=np.swapaxes(np.linalg.inv(edge),1,2)
    G=np.concatenate((-D.sum(1,keepdims=True),D),axis=1)
    F=G.transpose(0,2,1)@(Y-Y[:,:1]);J=np.linalg.det(F)
    if np.any(vol<=0) or np.any(J<=0):raise ValueError('Nonpositive material state')
    invt=np.linalg.inv(F).transpose(0,2,1);T=G@invt
    mu,bulk=props.T;vterm=bulk*np.log(J)-mu
    P=mu[:,None,None]*F+vterm[:,None,None]*invt
    grad=(G@P)*vol[:,None,None]
    H=(mu[:,None]*(G*G).sum(2))[:,:,None,None]*np.eye(3)
    H+=(bulk-vterm)[:,None,None,None]*T[:,:,:,None]*T[:,:,None,:]
    H*=vol[:,None,None,None]
    return grad,H,J,vol

def summarize(score,rows):
    if not np.isfinite(score).all():raise ValueError('Nonfinite screening score')
    order=np.argsort(-score,kind='stable')
    rank=np.empty(len(score),dtype=int);rank[order]=np.arange(1,len(score)+1)
    return dict(witness_rank=rank[rows].tolist(),witness_score=score[rows].tolist(),max=float(score.max()),
                counts_at_or_above_one=int((score>=1.).sum()),top_installed_cells=order[:12].tolist())

def screen(run,failed,flow_path,owner,offset_source):
    x, y, t, tags,pins=inputs(run);a=json.loads((run/'launch.json').read_text());b=json.loads((failed/'launch.json').read_text())
    if a['pins']!=b['pins']:raise ValueError('Different native inputs')
    e=json.loads((run/'verified-endpoint.json').read_text());endpath=run/'physical-accepted.node'
    if sha(endpath)!=e['physical_nodes_sha256']:raise ValueError('Unmatched native coordinates')
    y=y.astype(float);x=x.astype(float);end=np.loadtxt(endpath,skiprows=1,usecols=(1,2,3))
    old=json.loads(offset_source.read_text());off=old['source_element_offset']
    if pins['elements_sha256']!=old['source']['elements_sha256']:raise ValueError('Different registered connectivity')
    meta=json.loads((failed/'failure.json').read_text());root=meta['segment_start']
    if root!=e['accepted_steps'] or root<=0:raise ValueError('Wrong physical clock')
    log=[json.loads(l.partition('=')[2]) for l in (failed/'stderr.log').read_text().splitlines() if l.startswith('fem_backtracking_trials=')]
    row=next(l for l in reversed(log) if l['root']==root and l['object']==15)
    full=next(v for v in row['trials'] if v['alpha']==1)
    if not full['tile_witnesses_complete']:raise ValueError('Incomplete per-tile witness list')
    obs=np.array(full['tile_witness_tets'],dtype=int)-off
    if sha(flow_path)!=json.loads((flow_path.parent/'mass-transfer.json').read_text())['flow_sha256']:raise ValueError('Unmatched cell mass')
    prop,mat=materials(owner)
    dt=float(a['argv'][a['argv'].index('--muscle-step-seconds')+1])
    cavity=Path(a['environment']['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_NPZ'])
    if sha(cavity)!=a['environment']['NUMI_HUMAN_SKIN_FAT_FIXED_CAVITIES_SHA256']:raise ValueError('Unmatched remesh')
    with np.load(flow_path) as f:
        em=f['element_mass_kg'];old_nodal=f['nodal_mass_kg']
    with np.load(cavity) as f:
        keep=f['retained_input_cell_rows'];new=f['replacement_tetrahedra'];newmass=f['replacement_mass_kg']
        newtags=tags[f['replacement_parent_cell_rows'][f['replacement_parent_offsets'][:-1]]]
    # Preserve the actual FE lumping, including both bonded materials and the replacement cavities.
    removed=np.setdiff1d(np.arange(len(t)),keep);nodal=old_nodal.copy()
    for k in range(4):
        np.add.at(nodal,t[removed,k],-em[removed]/4)
        np.add.at(nodal,new[:,k],newmass/4)
    t=np.concatenate((t[keep],new));tags=np.r_[tags[keep],newtags]
    if nodal.min()<=0 or np.any((obs<0)|(obs>=len(t))) or set(tags)!={15,16}:raise ValueError('Unexpected support or material IDs')
    delta=end-y
    report={}
    group=16384
    for phase,q in [('loaded',y),('accepted',end)]:
        residual=np.zeros_like(q)
        H=np.zeros((len(q),3,3));refvol=np.zeros(len(t));J=np.zeros(len(t))
        for i in range(0,len(t),group):
            sl=slice(i,i+group);c=t[sl]
            g,block,J[sl],refvol[sl]=element_blocks(x[c],q[c],prop[tags[sl]-15])
            for k in range(4):
                np.add.at(residual,c[:,k],-g[:,k])
                np.add.at(H,c[:,k],block[:,k])
        elastic_net=residual.sum(0);residual[:,2]-=9.81*nodal
        A=dt**2*H+nodal[:,None,None]*np.eye(3)
        if (np.linalg.eigvalsh(A)[:,0]<=0).any():raise ValueError('Indefinite inertia plus diagonal material tangent')
        step=np.linalg.solve(A,(dt**2*residual)[:,:,None])[:,:,0]
        predictors={'mass_plus_nodal_tangent':q+step}
        if phase=='accepted':predictors['accepted_displacement_secant']=q+delta/root
        sources={}
        for key,nxt in predictors.items():
            P=np.zeros(len(t))
            for i in range(0,len(t),group):
                sl=slice(i,i+group);c=t[sl];P[sl]=np.linalg.det(nxt[c[:,1:]]-nxt[c[:,:1]])/(6*refvol[sl])
            # A score of one uses the entire remaining admissible volume margin in this increment.
            if np.any(J<=.35) or np.any(J>=2.5):raise ValueError('Frame outside native volume limits')
            score=np.maximum((J-P)/(J-.35),(P-J)/(2.5-J))
            sources[key]=summarize(score,obs)|dict(predicted_witness_J=P[obs].tolist())
        report[phase]=dict(witness_J=J[obs].tolist(),elastic_internal_force_sum_N=elastic_net.tolist(),
            calculated_net_force_N=residual.sum(0).tolist(),predictions=sources)
    return dict(pins=pins,accepted_sha256=sha(endpath),proposal_sha256=sha(cavity),mass_sha256=sha(flow_path),
        material_sha256=mat,failed_trace_sha256=sha(failed/'stderr.log'),dt_s=dt,accepted_steps=root,
        material_nodes=len(x),material_cells=len(t),skin_fat_mass_kg=float(nodal.sum()),installed_witness_rows=obs.tolist(),
        native_full_trial=full,results=report,
        scope='Offline ranking at the installed and accepted poses. The local implicit predictor includes true elastic forces, gravity, physical lumped mass and the nodal diagonal of the two-material Hessian. Off-diagonal coupling, bed, attachment, damping and skeletal reactions act in the full native solver. A rank measures association with the retained first failing face per reduction tile.')

if __name__=='__main__':print(json.dumps(screen(*(Path(p).resolve() for p in sys.argv[1:])),indent=2,allow_nan=False))
