#!/usr/bin/env python3
"""Independent complete Skin F32 volume preparation through existing 217 owner."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,sys,time
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
S=H/'skin010-restored-cavity-batch-072-expand1/skin-restored-boundary-proposal.npz';D=H/'skin010-source004-admission-073-retry001';D.mkdir(exist_ok=False)
with np.load(S)as z:a={k:z[k]for k in z.files}
p=a['points_world_m'];t=a['tetrahedra'];m=a['material_indices'];assert p.dtype==np.float32 and np.all(m==15)
with np.load(H/'verified-volume-preparation-010-merged038005/plcs/skin-shell.npz')as z:sp=z['points_world_m'];sf=z['faces'];sv=float(z['shell_signed_volume_m3'])
assert np.array_equal(p[:len(sp)].view(np.uint32),sp.view(np.uint32))
assert len(np.unique(p.view(np.uint32),axis=0))==len(p)
q=p[t].astype(np.float64);u=q[:,1]-q[:,0];v=q[:,2]-q[:,0];w=q[:,3]-q[:,0]
dv=np.einsum('ij,ij->i',np.cross(u,v),w)
permanent=np.sum(np.abs(u)*np.column_stack([np.abs(v[:,1]*w[:,2])+np.abs(v[:,2]*w[:,1]),np.abs(v[:,0]*w[:,2])+np.abs(v[:,2]*w[:,0]),np.abs(v[:,0]*w[:,1])+np.abs(v[:,1]*w[:,0])]),axis=1)
uncertain=np.flatnonzero(np.abs(dv)<=32*np.finfo(np.float64).eps*permanent)
def ip(row):
 return tuple(int(float(v).as_integer_ratio()[0])<<(149-(float(v).as_integer_ratio()[1].bit_length()-1))for v in p[row])
def exactdet(tet):
 a,b,c,d=map(ip,tet);u=[b[k]-a[k]for k in range(3)];v=[c[k]-a[k]for k in range(3)];w=[d[k]-a[k]for k in range(3)]
 return u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0])
es=[exactdet(t[i])for i in uncertain];assert np.all(dv>0)and all(v>0 for v in es)
pat=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
f=t[:,pat].reshape(-1,3);s=np.sort(f.astype(np.uint64),axis=1);keys=(s[:,0]<<40)|(s[:,1]<<20)|s[:,2];order=np.argsort(keys,kind='stable');ks=keys[order];starts=np.r_[0,np.flatnonzero(ks[1:]!=ks[:-1])+1];ends=np.r_[starts[1:],len(ks)];counts=ends-starts;assert np.all(counts<=2)
parity=((f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2])+(f[:,1]>f[:,2]))%2;orient=(1-2*parity).astype(np.int64);sums=np.add.reduceat(orient[order],starts);assert np.all(sums[counts==2]==0)
b=f[order[starts[counts==1]]];sbf=sums[counts==1]
ss=np.sort(sf.astype(np.uint64),axis=1);sk=(ss[:,0]<<40)|(ss[:,1]<<20)|ss[:,2];so=np.argsort(sk);bs=keys[order[starts[counts==1]]]
assert np.array_equal(bs,sk[so])
sp=((sf[:,0]>sf[:,1]).astype(np.int8)+(sf[:,0]>sf[:,2])+(sf[:,1]>sf[:,2]))%2;assert np.array_equal(sbf,(1-2*sp).astype(np.int64)[so])
edges=np.sort(np.concatenate([b[:,[0,1]],b[:,[1,2]],b[:,[2,0]]]),axis=1);_,ec=np.unique(edges,axis=0,return_counts=True);assert np.all(ec==2)
owner=H/'native-domain-217-source-004/build_native_domain_217.py';assert hashlib.sha256(owner.read_bytes()).hexdigest()=='46799a2d63fd20bc8a191f319f1f56244610a5e719983a7ce44cc5222659b6bd'
spec=importlib.util.spec_from_file_location('source004_skin073',owner);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);mod.configure_capacity_profile('full-reference-500k-nodes-2m-cells')
node=D/'skin.node';ele=D/'skin.ele';mod._write_joint_node_element_files(node,ele,p,t,m)
tic=time.monotonic();read=mod.read_tetgen_mesh(node,ele,15)
assert np.array_equal(read['points'],p)and np.array_equal(read['tetrahedra'],t)and np.array_equal(read['material_indices'],m)
assert not np.any(read['element_reoriented'])
r={'status':'PASS_COMPLETE_SKIN_POSITIVE_F32_VOLUME_SOURCE004_READBACK','physical_admission':False,'runtime':0,'accepted_steps':0,'source_path':str(S),'source_sha256':hashlib.sha256(S.read_bytes()).hexdigest(),'owner_path':str(owner),'owner_sha256':hashlib.sha256(owner.read_bytes()).hexdigest(),'capacity':mod.capacity_profile_receipt(),'nodes':len(p),'cells':len(t),'all_original_PLC_points_F32_bit_exact':True,'all_global_point_F32_bits_unique':True,'all_cells_positive':True,'filter_uncertain_cells_exactly_certified':len(uncertain),'filter_error_bound':'32*binary64_epsilon*sum_abs_6_products; all F32 coordinate differences exactly representable in binary64.','internal_face_orientation_cancellation':True,'source_PLC_boundary_oriented_exact':True,'boundary_faces':len(b),'boundary_edge_incidence_two':True,'nonmanifold_tet_faces':0,'source_closed_volume_m3':sv,'candidate_volume_m3':float(dv.sum()/6),'volume_abs_difference_m3':abs(float(dv.sum()/6)-sv),'node_sha256':hashlib.sha256(node.read_bytes()).hexdigest(),'element_sha256':hashlib.sha256(ele.read_bytes()).hexdigest(),'owner_readback_wall_s':time.monotonic()-tic,'mass_mapping':'NOT_RUN','native_install_and_motion':'NOT_RUN'}
(D/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
