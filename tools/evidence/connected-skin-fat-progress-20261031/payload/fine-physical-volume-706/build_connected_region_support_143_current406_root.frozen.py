"""Authenticate local same-tissue region adjacency without changing P1 labels.
Original region IDs are binding provenance; this does not merge region ledgers."""
from pathlib import Path
from collections import defaultdict
import numpy as np,importlib.util,sys,json,hashlib,time
H=Path(__file__).resolve().parent;ROOT=H.parent
OUT=H/'connected-region-mass-support-143-current406-root'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 start=time.monotonic();assert not OUT.exists();OUT.mkdir()
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'))
 op=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py';spec=importlib.util.spec_from_file_location('reg143',op);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 base=ROOT/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv';parent,_,_,pm=m.load_parent_current(base)
 t=parent['tets'];materials=parent['materials'];regions=parent['regions']
 # Complete source330 tetrahedral face incidence, no spatial-nearness adjacency.
 local_faces=np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
 faces=np.sort(t[:,local_faces].reshape(-1,3),axis=1);owners=np.repeat(np.arange(len(t)),4)
 order=np.lexsort((faces[:,2],faces[:,1],faces[:,0]));fs=faces[order];os=owners[order]
 adjacency={mat:defaultdict(set)for mat in (15,16)};edge_proofs={mat:{}for mat in (15,16)}
 for mat in (15,16):
  for r in np.unique(regions[materials==mat]):adjacency[mat][int(r)].add(int(r))
 for j in np.flatnonzero(np.all(fs[:-1]==fs[1:],axis=1)):
  a,b=int(os[j]),int(os[j+1]);mat=int(materials[a])
  if mat not in (15,16)or materials[b]!=mat:continue
  r,s=int(regions[a]),int(regions[b])
  if r==s:continue
  adjacency[mat][r].add(s);adjacency[mat][s].add(r)
  edge_proofs[mat].setdefault(tuple(sorted((r,s))),{'source_parent_tets':[a,b],'shared_source_node_rows':fs[j].tolist()})
 datasets=[('Skin',15,H/'current-semantic-atlas-088/skin-semantic-atlas-002/output-001/skin-current-semantic-support.npz'),('Fat',16,H/'fat-current-semantic-support-406-root-run002/fat-current-semantic-support.npz')]
 results=[]
 for role,mat,path in datasets:
  with np.load(path)as z:a={k:z[k]for k in z.files}
  g=a['group_tuples'];ids=a['cell_group_ids'];off=a['cell_group_offsets'];count=len(off)-1
  if role=='Fat':
   v406=H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'
   if str(a['volume_sha256'].item())!=sha(v406):raise ValueError('current Fat P1 support is not pinned to target 406')
  child=np.repeat(np.arange(count,dtype=np.int64),np.diff(off));r=g[ids,2]
  factor=int(max(parent['regions'].max(),r.max()))+1
  direct=np.unique(child*factor+r);dc=direct//factor;dr=direct%factor
  # Vectorize by original region; each output region needs one source-face path.
  pairs=[]
  for region in sorted(set(dr.tolist())):
   rows=dc[dr==region];adj=sorted(adjacency[mat][int(region)])
   pairs.append((rows[:,None]*factor+np.array(adj)[None,:]).reshape(-1))
  expanded=np.unique(np.concatenate(pairs));ec=expanded//factor;er=expanded%factor
  lengths=np.bincount(ec,minlength=count);assert np.all(lengths>0)
  offsets=np.concatenate(([0],np.cumsum(lengths)))
  assert np.all(np.isin(direct,expanded))
  # Original source-parent/P1 arrays stay bytewise identical.
  a['cell_admissible_region_offsets']=offsets.astype(np.int64);a['cell_admissible_region_ids']=er.astype(np.int32)
  a['admissibility_source_support_sha256']=np.array(sha(path))
  a['admissibility_max_source_region_hops']=np.array(1,np.int32)
  dest=OUT/(role.lower()+'-connected-region-support.npz');np.savez_compressed(dest,**a)
  with np.load(dest)as z:
   for key,value in a.items():assert np.array_equal(z[key],value)
  results.append({'role':role,'material':mat,'source_support_path':str(path),'source_support_sha256':sha(path),'output':str(dest),'output_sha256':sha(dest),'cells':count,'original_cell_group_slots':len(ids),'original_cell_region_slots':len(direct),'admissible_cell_region_slots':len(er),'maximum_regions_per_cell':int(lengths.max()),'authenticated_region_edges':[{'regions':list(k),**v}for k,v in sorted(edge_proofs[mat].items())]})
 report={'status':'SOURCE_ONLY_AUTHENTICATED_ONE_HOP_SAME_TISSUE_ADMISSIBILITY','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','policy':'Current child P1/geometric group labels remain exact. Additional mass contributor regions require direct same-material shared-tetrahedron-face adjacency in immutable source330. Each original parent donor/region/stable-face contribution remains distinct in the mass ledger. Every actual transfer must still satisfy19.353mm and unchanged density bounds. One-hop support is not a nearest-label override, nor permission to transfer across materials, disconnected source tissues, arbitrary regions or distances.','source330':pm,'semantics_source':{'path':'/Users/numi/Work/skin-fat-native-integration-209/apps/numilab_human_myosim_visual_probe.mm','line':27015,'quote':'region_owner=dominant_original_registered_skin_binding_body_id'},'results':results,'builder':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__).resolve())},'current_fat_target':{'path':str(H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'),'sha256':sha(H/'current-D309-I399-target-406-root/target397/fat-current-typed-target-rebound.npz'),'semantic_p1_path':str(H/'fat-current-semantic-support-406-root-run002/fat-current-semantic-support.npz'),'semantic_p1_sha256':sha(H/'fat-current-semantic-support-406-root-run002/fat-current-semantic-support.npz')},'elapsed_s':time.monotonic()-start}
 (OUT/'receipt.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'results':[{k:v for k,v in d.items()if k!='authenticated_region_edges'}for d in results],'elapsed_s':report['elapsed_s']}),flush=True)
if __name__=='__main__':main()
