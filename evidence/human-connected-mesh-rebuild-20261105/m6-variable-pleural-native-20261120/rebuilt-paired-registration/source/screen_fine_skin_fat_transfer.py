"""Screen the retained fine topology using its exact registered P1 parent map."""
import csv
import json
from pathlib import Path
import subprocess

import numpy as np

from check_current_parent_volume import c, source_points, volumes
from rebind_current_skin import sha256

root=Path.cwd().parent/'captured-fat-sliver-495/proposal/compiler'
proof_path=root/'mass-admission/run-manifest.json'
argv=json.loads(proof_path.read_text())['argv']
assignments=dict(arg.split('=',1) for arg in argv if arg.startswith('NUMI_') and '=' in arg)
get=lambda f:Path(assignments['NUMI_H706_FRACTIONAL_SOURCE330_'+f])
assert all(sha256(get(f))==assignments['NUMI_H706_FRACTIONAL_SOURCE330_'+f+'_SHA256'] for f in ('NODES','TETS'))
parent=np.loadtxt(get('NODES'),delimiter=',',skiprows=1)
ctet=np.loadtxt(get('TETS'),delimiter=',',skiprows=1,usecols=(3,4,5,6),dtype=int)
assert np.array_equal(ctet,c)
assert np.array_equal(parent[:,0],np.arange(len(parent)))
parent_world=parent[:,2:5]
parent_owner=parent[:,9].astype(int)

with np.load('paired-registration-with-face-ancestry/registration.npz',allow_pickle=False) as registration:
    source_frame=np.tile(registration['world_to_donor'].reshape(-1,3,3),(3,1,1))
    source_donor=np.tile(registration['donor_body'],3)
assert np.array_equal(parent_owner,source_donor)
x=np.loadtxt(root/'volume-export/registered/donor-frames.csv',delimiter=',',skiprows=1)
frame_rows={int(row[1]):row for row in x}
frames=np.asarray([frame_rows[body] for body in parent_owner])
world_frame=frames[:,5:14].reshape(-1,3,3)
predicted=frames[:,2:5]+np.einsum('nij,nj->ni',world_frame,parent[:,10:13])
predicted_transpose=frames[:,2:5]+np.einsum('nji,nj->ni',world_frame,parent[:,10:13])
forward=np.linalg.norm(predicted-parent_world,axis=1).max()
transpose=np.linalg.norm(predicted_transpose-parent_world,axis=1).max()
print('donor_frame_parity_m',forward,transpose,flush=True)
if min(forward,transpose)>3e-6:
    raise ValueError('parent and explicit donor poses belong to different frames')
if transpose<forward:
    world_frame=np.swapaxes(world_frame,1,2)

with np.load('paired-skin-fat-refit-001/paired-geometry-and-mass.npz',allow_pickle=False) as pair:
    geometry=pair['points_world_m']
delta=(geometry-source_points).reshape(-1,3)
parent_delta=np.einsum('nij,njk,nk->ni',world_frame,source_frame,delta)

xyz=np.loadtxt(root/'volume-export/volume/mesh.node',skiprows=1)
assert np.array_equal(xyz[:,0],np.arange(1,len(xyz)+1))
xyz=xyz[:,1:]
operand=root/'bindings/combined-node-map.csv.zst'
proc=subprocess.Popen(['/opt/homebrew/bin/zstd','-qcd',str(operand)],stdout=subprocess.PIPE,text=True)
rows=csv.reader(proc.stdout)
header=next(rows)
fields=['joint_node_id','parent_tet_row',*[f'parent_bary{i}' for i in range(4)],
        *[f'parent_current_residual_{x}_m' for x in 'xyz']]
cols=[header.index(name) for name in fields]
roles=np.empty(len(xyz),dtype='U8')
role_col=header.index('role')
ref=np.full((len(xyz),len(cols)),np.nan)
other=[]
for row in rows:
    item=np.asarray([float(row[k]) for k in cols])
    j=int(item[0])-1
    if np.isfinite(ref[j,0]):
        if not np.array_equal(item,ref[j]):
            other.append((j,item))
    else:
        ref[j]=item
        roles[j]=row[role_col]
assert proc.wait()==0
assert np.array_equal(ref[:,0],np.arange(1,len(ref)+1))
def transfer(rows, points):
    cell=rows[:,1].astype(int)
    bary=rows[:,2:6]
    print("op range",cell.min(),cell.max(), int(np.count_nonzero((cell<0)|(cell>=len(c)))),"min bary",bary.min(),flush=True)

    assert np.all((cell>=0)&(cell<len(c))) and np.all(bary>=-1e-12)
    assert np.max(np.abs(bary.sum(axis=1)-1))<1e-10
    parent_xyz=np.einsum('ni,nij->nj',bary,parent_world[c[cell]])
    error=np.linalg.norm(parent_xyz+rows[:,6:9]-points,axis=1).max()
    if error>3e-6:
        raise ValueError(f'parent rest residual does not reproduce registered coordinates: {error} m')
    return np.einsum('ni,nij->nj',bary,parent_delta[c[cell]]),error
fine_delta,reference_error=transfer(ref,xyz)
merge_error=0.
if other:
    ids=np.asarray([j for j,_ in other])
    alternate=np.asarray([row for _,row in other])
    extra,extra_error=transfer(alternate,xyz[ids])
    merge_error=float(np.linalg.norm(extra-fine_delta[ids],axis=1).max())
print('registered_parent_error_m',reference_error,'shared interface alternate delta error m',merge_error,flush=True)
t=np.loadtxt(root/'volume-export/volume/mesh.ele',skiprows=1,dtype=np.int32)
assert np.array_equal(t[:,0],np.arange(1,len(t)+1))
tets,material=t[:,1:5]-1,t[:,5]
assert set(material)=={15,16}
new=xyz+fine_delta
baseline=np.empty(len(tets))
candidate=np.empty(len(tets))
quality=np.empty(len(tets))
for start in range(0,len(tets),100_000):
    s=slice(start,start+100_000)
    baseline[s]=volumes(xyz.astype(np.float32).astype(float),tets[s])[0]
    candidate[s],quality[s]=volumes(new.astype(np.float32).astype(float),tets[s])
inverted=np.flatnonzero(candidate<=0)
introduced=np.flatnonzero((baseline>0)&(candidate<=0))
role_by_cell=roles[tets[introduced]]
witness_nodes=np.unique(tets[introduced])
summary=dict(invalid_fine_cell_role_histogram={
    str(key):int((np.sum(role_by_cell=='O',axis=1)==key).sum()) for key in range(5)},
    invalid_fine_witness_nodes_by_role={key:int((roles[witness_nodes]==key).sum()) for key in sorted(set(roles))},
    fine_nodes=len(xyz),fine_cells=len(tets),baseline_inverted=int(np.count_nonzero(baseline<=0)),
             candidate_inverted=len(inverted),introduced_inverted=len(introduced),
             minimum_normalized_determinant=float(quality.min()),fine_parent_replay_error_m=float(reference_error),shared_interface_alternate_delta_error_m=merge_error,
             max_node_displacement_m=float(np.linalg.norm(fine_delta,axis=1).max()),
             role_signed_volume_before_m3={str(i):float(baseline[material==i].sum()) for i in (15,16)},
             role_signed_volume_after_m3={str(i):float(candidate[material==i].sum()) for i in (15,16)},
             first_inverted_child_rows=introduced[:300].tolist(),physical_admission=False)
Path('fine-transfer-screen.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='first_inverted_child_rows'}),flush=True)

np.savez_compressed('fine-transfer-screen-map.npz', old_world_m=xyz,
                    fine_node_delta_m=fine_delta, roles=roles)
