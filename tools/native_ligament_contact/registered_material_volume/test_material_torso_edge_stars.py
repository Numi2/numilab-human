"""Test source-curved edges on complete physical edge stars, with conserved parent masses."""
import csv,json,sys
from pathlib import Path
from collections import Counter
from itertools import combinations
import numpy as np
from emit_skin_source_registration import sha
launch,targets,flow,mass_report,out=map(Path,sys.argv[1:6]);out.mkdir(exist_ok=True)
use_groups=len(sys.argv)==7
if len(sys.argv) not in (6,7):raise ValueError('Expected launch, targets, flow, mass report, output and optional native volumes')
pfx='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_';env=json.loads(launch.read_text())['environment']
def path(k):
    p=Path(env[pfx+k]);assert sha(p)==env[pfx+k.replace('_CSV','')+'_SHA256'];return p
def surface(k):
    p=Path(env[pfx+'SURFACE_O_'+k+'_CSV']);assert sha(p)==env[pfx+'SURFACE_O_'+k+'_SHA256'];return p
with path('TETGEN_NODE').open() as f:
    n=int(next(f).split()[0]);x=np.loadtxt(f,max_rows=n,usecols=(1,2,3)).astype('f4').astype('f8')
with path('TETGEN_ELEMENTS').open() as f:
    n=int(next(f).split()[0]);a=np.loadtxt(f,dtype='i4',max_rows=n);t=a[:,1:5]-1;tags=a[:,5]
surf=np.loadtxt(surface('FACES'),delimiter=',',skiprows=1,usecols=(1,2,3),dtype='i4')
outer=np.full(76153,-1,dtype=int);donor=np.full(len(x),-1,dtype=int)
with path('COMBINED_NODE_MAP_CSV').open() as f:
    for row in csv.DictReader(f):
        i=int(row['joint_node_id'])-1;d=int(row['donor_body'])
        if donor[i]>=0 and donor[i]!=d:raise ValueError('Inconsistent joined donor')
        donor[i]=d
        if row['registered_surface_layer']=='O':outer[int(row['registered_surface_vertex_row'])]=i
assert (outer>=0).all() and (donor>=0).all()
m=json.loads(mass_report.read_text());assert sha(flow)==m['new_flow_sha256']
ledger=np.load(flow);mass=ledger['element_mass_kg']
if use_groups:
    parent_now=np.fromfile(sys.argv[6],dtype='f8');group=ledger['group_for_cell']
    assert len(parent_now)==len(tags) and np.all(parent_now>1e-18)
    group_mass=np.bincount(group,weights=mass);group_vol=np.bincount(group,weights=parent_now)
    group_role=np.zeros(len(group_mass),dtype=int);group_role[group]=tags
    if (group_role[group]!=tags).any():raise ValueError('Mixed material recipient group')
    # Retain every original carrier-to-recipient allocation; only subdivide within that recipient.
    if np.max(abs(group_mass-np.bincount(ledger['group'],weights=ledger['mass_flow_kg'])))>1e-12:raise ValueError('Original recipient quotas differ')
s=np.load(targets);rows=s['skin_face_rows'];nn=len(s['surface_node']);nr=len(rows)
face=outer[surf];work={};votes=Counter()
for j in range(3):
    for row,p in zip(rows,s['end_m'][nn+(j+1)*nr:nn+(j+2)*nr],strict=True):
        edge=tuple(sorted((int(face[row,j]),int(face[row,(j+1)%3]))))
        if edge in work and max(abs(work[edge]-p))>1e-12:raise ValueError('Common edge projection mismatch')
        work[edge]=p;votes[edge]+=1
seed=np.array(sorted(set(v for e in work for v in e)),dtype='i4')
local=np.flatnonzero(np.isin(t,seed).sum(1)>=2)
owners=local.copy();cells=t[local].copy();mt=tags[local]
changed_faces=np.flatnonzero(np.isin(face,seed).sum(1)>=2);sf=face[changed_faces].copy();sfparent=changed_faces.copy()
def geom(v):
    ed=v[:,1:]-v[:,:1];D=np.einsum('ij,ij->i',ed[:,0],np.cross(ed[:,1],ed[:,2]))
    e=v[:,[0,0,0,1,1,2]]-v[:,[1,2,3,2,3,3]];l=np.max(np.einsum('nij,nij->ni',e,e),axis=1)
    return D/6,D/l**1.5
def split(record,mask,pair,node):
    src=record[mask];left=src.copy();right=src.copy()
    left[left==pair[0]]=node;right[right==pair[1]]=node
    return np.concatenate((record[~mask],left,right))
attempts=[];original_count=len(x)
for edge,target in sorted(work.items(),key=lambda kv:-np.linalg.norm(kv[1]-x[list(kv[0])].mean(0))):
    a,b=edge;mask=(cells==a).any(1)&(cells==b).any(1);sm=(sf==a).any(1)&(sf==b).any(1)
    if not mask.any() or not sm.any():raise ValueError('Incomplete physical star')
    mid=(x[a]+x[b])/2;report=dict(edge=edge,native_star_cells=int(mask.sum()),boundary_faces=int(sm.sum()),source_midpoint_distance_m=float(np.linalg.norm(target-mid)))
    if donor[a]!=donor[b]:report['status']='mixed donor: no qualified new frame';attempts.append(report);continue
    ix=len(x);next_t=split(cells,mask,edge,ix);parents=np.r_[owners[~mask],owners[mask],owners[mask]]
    active=np.unique(owners[mask]);ci=np.isin(parents,active);pt=parents[ci]
    # Recompute every descendant of each touched parent at emitted Float32 precision.
    accepted=False;best=None
    for fraction in (1.,.5,.25,.125,.0625,.03125,0.):
        p=(mid+fraction*(target-mid)).astype('f4').astype('f8')
        verts=np.concatenate([x,p[None,:]])[next_t[ci]]
        v,q=geom(verts)
        dens=np.array([mass[k]/v[pt==k].sum() if v[pt==k].sum()>0 else -1 for k in active])
        rind=tags[active];lo=np.where(rind==15,900.,850.);hi=np.where(rind==15,1100.,950.)
        if use_groups:
            new_vol=np.array([v[pt==k].sum() for k in active])
            groups,ids=np.unique(group[active],return_inverse=True)
            delta=np.bincount(ids,weights=new_vol-parent_now[active])
            dens=group_mass[groups]/(group_vol[groups]+delta)
            rind=group_role[groups];lo=np.where(rind==15,900.,850.);hi=np.where(rind==15,1100.,950.)
        reason=dict(invalid_volume=int((v<=1e-18).sum()),invalid_quality=int((q<1e-5).sum()),invalid_parent_density=int(np.sum((dens<lo-1e-8)|(dens>hi+1e-8))))
        if fraction==1.:report['full_fraction_failures']=reason
        if fraction==0.:report['straight_split_failures']=reason
        if max(reason.values()) or fraction==0.:continue
        if use_groups:
            group_vol[groups]+=delta;parent_now[active]=new_vol
        x=np.vstack((x,p));cells=next_t;owners=parents;sf=split(sf,sm,edge,ix);sfparent=np.r_[sfparent[~sm],sfparent[sm],sfparent[sm]]
        report.update(status='admitted for offline physical screen',new_node=ix,fraction=fraction,moved_m=float(np.linalg.norm(p-mid)));accepted=True;break
    if not accepted:report['status']='no positive source-directed split within physical gates'
    attempts.append(report)
np.savez_compressed(out/'local-volume-proposal.npz',new_points_world_m=x[original_count:],affected_tets=local,tets=cells,parent=owners,affected_skin_faces=changed_faces,skin_faces=sf,skin_parent=sfparent)
valid=[a for a in attempts if 'new_node' in a]
result=dict(scope='offline full-star candidate; parent quotas screened, native material rows not emitted',mass_quota='fixed original carrier-to-recipient group' if use_groups else 'fixed original tetrahedron',tested_edges=len(attempts),admitted_edges=len(valid),fully_source_matched=sum(a['fraction']==1. for a in valid),counts=dict(Counter(a['status'] for a in attempts)),original_nodes=original_count,affected_parent_count=len(np.unique(owners)),source=sha(targets),launch=sha(launch),proposal_sha256=sha(out/'local-volume-proposal.npz'),trials=attempts)
(out/'edge-stars.json').write_text(json.dumps(result,indent=2)+'\n')
