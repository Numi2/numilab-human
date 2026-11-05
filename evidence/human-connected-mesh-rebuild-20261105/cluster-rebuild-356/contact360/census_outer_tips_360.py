from pathlib import Path
import numpy as np,json,hashlib
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=W/'contact-stall-360'
V=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/current-native-1024-root/current-433-1037/candidate-current-typed-volume.npz')
C=W/'witness-cluster-rebuild-356/proposal/compiler/rich433/candidate-current-typed-volume.npz'
O=W/'witness-cluster-rebuild-356/proposal/candidate-O-typed-current.npz'
with np.load(C,allow_pickle=False) as z:
 t=z['volume_tetrahedra'];p=z['points_reference_world_m_f64'];m=z['current_typed_role_source_row_to_joint_node_O']
with np.load(O,allow_pickle=False) as z:f=z['faces'];po=z['points_world_m_f32']
with np.load(V,allow_pickle=False) as z:originalmap=z['current_typed_role_source_row_to_joint_node_O']
active=m>=0
degree=np.bincount(f.ravel(),minlength=len(m))
count=np.bincount(t.ravel(),minlength=len(p))
tips=np.flatnonzero(active&(degree==3))
tips=tips[count[m[tips]]==1]
report=[]
for v in tips:
 faces=f[np.any(f==v,axis=1)];nei=np.setdiff1d(faces.ravel(),[v])
 length=np.linalg.norm(po[nei].astype(float)-po[v].astype(float),axis=1)
 keep=int(nei[np.argmin(length)]);dist=float(length.min())
 if dist>1e-6 and v!=59895:continue
 xyz=po[[v,*nei]].astype(float);vol=abs(np.linalg.det(xyz[1:]-xyz[:1]))/6
 report.append(dict(source_O_tip=int(v),source_O_neighbors=nei.tolist(),neighbor_distances_m=length.tolist(),minimum_edge_m=dist,tip_tet_volume_m3=float(vol),original_drop=int(originalmap[v]),original_keep=int(originalmap[keep]),rich_drop=int(m[v]),rich_keep=int(m[keep]),captured_stall_witness=int(v)==59895))
r=dict(source_asset=str(O),source_sha256=hashlib.file_digest(O.open('rb'),'sha256').hexdigest(),degree3_one_tet_tip_count=len(tips),submicron_tip_count=sum(x['minimum_edge_m']<=1e-6 for x in report),selection='O surface valence3, one incident volume cell, minimum existing edge<=1um; prospective candidates only, all original topology/material/mass/attachment/back/bone gates remain mandatory',tips=report,physical_admission=False)
(D/'outer-tip-census.json').write_text(json.dumps(r,indent=2)+'\n')
print('COUNTS',len(tips),r['submicron_tip_count'])
print('WITNESS',[x for x in report if x['captured_stall_witness']])
print('SMALLEST',sorted(report,key=lambda x:x['minimum_edge_m'])[:8])
