"""Compose a fixed-boundary volume patch onto a disjoint current volume.
Uses existing NPZ fields and immediate cell-parent unions; assigns no mass.
"""
from collections import Counter
import numpy as np
PAT=np.asarray(((1,2,3),(0,3,2),(0,1,3),(0,2,1)),np.int64)
def cell_keys(t,mat):
 t=np.asarray(t);mat=np.asarray(mat)
 if t.ndim!=2 or t.shape[1]!=4 or not len(t) or mat.shape!=(len(t),):raise ValueError('cell shape')
 if np.any(np.diff(np.sort(t,axis=1),axis=1)==0):raise ValueError('degenerate cell indices')
 s=np.sort(t,axis=1);keys=np.empty(len(t),dtype=[('mat','<i4'),('a','<i8'),('b','<i8'),('c','<i8'),('d','<i8')]);keys['mat']=mat
 for j,k in enumerate(('a','b','c','d')):keys[k]=s[:,j]
 if len(np.unique(keys))!=len(keys):raise ValueError('duplicate material cell keys')
 return keys
def oriented_boundary(t):
 f=t[:,PAT].reshape(-1,3);s=np.sort(f,axis=1);order=np.lexsort((s[:,2],s[:,1],s[:,0]));ss=s[order]
 start=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))];count=np.diff(np.r_[start,len(s)])
 signs=1-2*(((f[:,0]>f[:,1]).astype(np.int8)+(f[:,0]>f[:,2]).astype(np.int8)+(f[:,1]>f[:,2]).astype(np.int8))%2)
 total=np.add.reduceat(signs[order],start)
 if np.any(count>2) or np.any(total[count==2]!=0):raise ValueError('nonmanifold or inconsistent interior')
 return np.c_[ss[start[count==1]],total[count==1]]
def compose(base,patch,current,patch_parent_offsets,patch_parent_base_rows):
 p=base['points_world_m']
 if p.dtype!=np.float32 or not np.all(np.isfinite(p)):raise ValueError('finite emitted Float32 coordinates required')
 for v in (patch,current):
  if v['points_world_m'].dtype!=p.dtype or not np.array_equal(p.view('u4'),v['points_world_m'].view('u4')):raise ValueError('point domain changed')
 for v in (base,patch,current):
  if 'points_world_m_f32' in v and not np.array_equal(p.view('u4'),v['points_world_m_f32'].view('u4')):raise ValueError('coordinate alias mismatch')
  if np.any(v['tetrahedra']<0) or np.any(v['tetrahedra']>=len(p)):raise ValueError('point index')
 bt=base['tetrahedra'];pt=patch['tetrahedra'];ct=current['tetrahedra']
 bm=base['material_indices'];pm=patch['material_indices'];cm=current['material_indices']
 bk=cell_keys(bt,bm);pk=cell_keys(pt,pm);ck=cell_keys(ct,cm)
 bo=np.argsort(bk);co=np.argsort(ck)
 def match(keys,reference,order):
  pos=np.searchsorted(reference[order],keys);ok=pos<len(reference);rows=np.full(len(keys),-1,np.int64)
  ii=np.flatnonzero(ok);ii=ii[reference[order[pos[ii]]]==keys[ii]];rows[ii]=order[pos[ii]]
  return rows
 p_to_b=match(pk,bk,bo);b_to_p=match(bk,pk,np.argsort(pk));b_to_c=match(bk,ck,co)
 removed=np.flatnonzero(b_to_p<0);added=np.flatnonzero(p_to_b<0)
 if np.any(b_to_c[removed]<0):raise ValueError('current overlaps retired patch source cells')
 common=np.flatnonzero(p_to_b>=0)
 # Equal material/corner keys represent the same cell; positive-determinant
 # validation below certifies legitimate even corner permutations.

 if not np.array_equal(ct[b_to_c[removed]],bt[removed]):raise ValueError('current changed patch source cells')
 for vv in (base,patch,current):
  x=vv['points_world_m'][vv['tetrahedra']].astype(np.float64);u=x[:,1:]-x[:,:1]
  det=np.einsum('ij,ij->i',u[:,0],np.cross(u[:,1],u[:,2]))
  if np.any(~np.isfinite(det)) or np.any(det<=0):raise ValueError('nonpositive cell')
 if not np.array_equal(oriented_boundary(bt),oriented_boundary(pt)):raise ValueError('patch boundary changed')
 if not np.array_equal(np.unique(bm),np.unique(pm)):raise ValueError('material domain changed')
 for material in np.unique(bm):
  if not np.array_equal(oriented_boundary(bt[bm==material]),oriented_boundary(pt[pm==material])):raise ValueError('material interface changed')

 offsets=np.asarray(patch_parent_offsets,np.int64);parents=np.asarray(patch_parent_base_rows,np.int64)
 if offsets.shape!=(len(pt)+1,) or offsets[0]!=0 or offsets[-1]!=len(parents) or np.any(np.diff(offsets)<0):raise ValueError('parent CSR')
 keep=np.ones(len(ct),bool);keep[b_to_c[removed]]=False
 ot=np.vstack((ct[keep],pt[added]));om=np.r_[cm[keep],pm[added]]
 cell_keys(ot,om)
 if not np.array_equal(oriented_boundary(ct),oriented_boundary(ot)):raise ValueError('composed boundary changed')
 newparents=[];newoff=[0]
 for row in added:
  src=np.unique(parents[offsets[row]:offsets[row+1]])
  if not len(src) or np.any(src<0) or np.any(src>=len(bt)) or np.any(b_to_c[src]<0):raise ValueError('unmapped parent union')
  if np.any(bm[src]!=pm[row]):raise ValueError('material parent mismatch')
  if not np.array_equal(ct[b_to_c[src]],bt[src]):raise ValueError('current changed patch lineage source')
  newparents.extend(b_to_c[src].tolist());newoff.append(len(newparents))
 mapping=np.r_[np.flatnonzero(keep),np.full(len(added),-1,np.int64)];reverse=np.full(len(ct),-1,np.int64);reverse[np.flatnonzero(keep)]=np.arange(keep.sum())
 out=current.copy();out.update(tetrahedra=ot,material_indices=om,retained_input_cell_rows=np.flatnonzero(keep),output_cell_to_input_cell=mapping,input_cell_to_output_cell=reverse,replacement_tetrahedra=pt[added],replacement_parent_offsets=np.asarray(newoff,np.int64),replacement_parent_cell_rows=np.asarray(newparents,np.int64),replacement_tet_offsets=np.asarray([0,len(added)],np.int64),output_point_to_input_point_row=np.arange(len(p),dtype=np.int64),output_point_parent_offsets=np.zeros(len(p)+1,np.int64),output_point_parent_input_cell_rows=np.empty(0,np.int64))
 return out,{'removed_base_cell_rows':removed,'added_patch_cell_rows':added,'retired_current_cell_rows':b_to_c[removed]}
