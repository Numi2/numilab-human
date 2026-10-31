"""Independent bounded reader of the existing FSSUP090/FSSADJ01 support."""
from pathlib import Path
import hashlib,struct
import numpy as np
def load_support(path,material,expected_children,parent_donors,parent_regions,parent_faces):
 data=Path(path).read_bytes();cursor=0
 def raw(n):
  nonlocal cursor
  if cursor+n>len(data):raise ValueError('truncated FSSUP090')
  b=data[cursor:cursor+n];cursor+=n;return b
 def array(dtype,n):
  return np.frombuffer(raw(np.dtype(dtype).itemsize*n),dtype=dtype).copy()
 if raw(8)!=b'FSSUP090':raise ValueError('FSSUP090 magic')
 nc,ns,ng,np_,npr,nfr=struct.unpack('<QQQQQQ',raw(48))
 if nc!=expected_children or np_!=len(parent_donors) or ns>4000000 or ng>10000 or ng==0 or npr>200000 or nfr>200000:raise ValueError('FSSUP090 count domain')
 d={}
 for name,dtype,n in [('child_offsets','<u4',nc+1),('slot_groups','<u4',ns),('fractions','<f8',ns),('group_materials','<u4',ng),('group_donors','<u4',ng),('group_regions','<u4',ng),('group_ids','<u8',ng),('parent_faces','<u4',np_),('group_parent_offsets','<u4',ng+1),('group_parents','<u4',npr),('group_face_offsets','<u4',ng+1),('group_faces','<u4',nfr),('child_materials','<u4',nc),('child_donors','<u4',nc),('child_regions','<u4',nc)]:
  d[name]=array(dtype,n)
 d['typed_support_sha256']=raw(64).decode()
 if raw(8)!=b'FSSADJ01':raise ValueError('FSSADJ01 magic')
 na,hops=struct.unpack('<QI',raw(12))
 if na>50000000 or hops!=1:raise ValueError('FSSADJ01 bound/hops')
 d['admissibility_source_sha256']=raw(64).decode()
 d['region_offsets']=array('<u4',nc+1);d['regions']=array('<u4',na)
 if cursor!=len(data):raise ValueError('FSSUP090 trailing bytes')
 for key,n,end in [('child_offsets',nc+1,ns),('group_parent_offsets',ng+1,npr),('group_face_offsets',ng+1,nfr),('region_offsets',nc+1,na)]:
  a=d[key].astype(np.int64)
  if len(a)!=n or a[0]!=0 or a[-1]!=end or np.any(np.diff(a)<0):raise ValueError('FSSUP090 CSR '+key)
 if np.any(d['slot_groups']>=ng) or np.any(np.diff(d['child_offsets'])==0):raise ValueError('FSSUP090 child support domain')
 if np.any(d['group_materials']!=material) or np.any(d['child_materials']!=material):raise ValueError('FSSUP090 material domain')
 if not np.array_equal(d['parent_faces'],parent_faces):raise ValueError('FSSUP090 parent stable source faces')
 if not np.array_equal(np.sort(d['group_parents']),np.arange(np_,dtype=np.uint32)):raise ValueError('FSSUP090 complete parent partition')
 pg=np.repeat(np.arange(ng,dtype=np.uint32),np.diff(d['group_parent_offsets'].astype(np.int64)))
 d['parent_groups']=np.empty(np_,dtype=np.uint32);d['parent_groups'][d['group_parents']]=pg
 if not np.array_equal(d['group_donors'][d['parent_groups']],parent_donors) or not np.array_equal(d['group_regions'][d['parent_groups']],parent_regions):raise ValueError('FSSUP090 source-parent material/donor/region identity')
 d['child_group_keys']=(np.repeat(np.arange(nc,dtype=np.uint64),np.diff(d['child_offsets'].astype(np.int64)))<<np.uint64(32))|d['slot_groups'].astype(np.uint64)
 d['child_region_keys']=(np.repeat(np.arange(nc,dtype=np.uint64),np.diff(d['region_offsets'].astype(np.int64)))<<np.uint64(32))|d['regions'].astype(np.uint64)
 if len(np.unique(d['child_group_keys']))!=ns:raise ValueError('FSSUP090 duplicate child semantic group')
 return d,hashlib.sha256(data).hexdigest()
def check_edges(d,children,parents,parent_regions):
 keys=(np.asarray(children,dtype=np.uint64)<<np.uint64(32))|d['parent_groups'][parents].astype(np.uint64)
 nonpreferred=int(np.count_nonzero(~np.isin(keys,d['child_group_keys'])))
 rk=(np.asarray(children,dtype=np.uint64)<<np.uint64(32))|parent_regions[parents].astype(np.uint64)
 if not np.isin(rk,d['child_region_keys']).all():raise ValueError('flow outside one-hop source region')
 return nonpreferred
