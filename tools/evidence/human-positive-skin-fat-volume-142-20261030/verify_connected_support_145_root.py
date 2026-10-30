"""Independent admission-metadata controls; no flow/physical admission."""
from pathlib import Path
from collections import defaultdict
import numpy as np,importlib.util,sys,json,hashlib,time
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def edge_ok(parent,mat,pair,rows,nodes):
 if len(set(rows))!=2 or len(set(nodes))!=3:return False
 a,b=map(int,rows)
 if any(parent['materials'][v]!=mat for v in rows):return False
 if set(map(int,parent['regions'][rows]))!=set(pair):return False
 shared=set(map(int,parent['tets'][a]))&set(map(int,parent['tets'][b]))
 return shared==set(nodes)
def main():
 start=time.monotonic();folder=H/'connected-region-mass-support-143-root';rp=folder/'receipt.json';r=json.loads(rp.read_text())
 assert sha(rp)=='055a1fb7f909205248b6d8ebe160ef8ea1c743f335615fc93dd14192728de246'and r['physical_admission']is False and r['native_steps']==0
 sys.path.insert(0,str(H/'downstream-003-current-typed-228/native-domain-217-source-005'))
 op=H/'current-semantic-atlas-088/export_authoring_registered_surface_sidecars_230_semantic_088.py';s=importlib.util.spec_from_file_location('reg145',op);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
 base=H.parent/'native-physical-parent-mass-330/native-source-identity-control-003/compiled-parent-cell-mass.csv'
 parent,_,_,pm=m.load_parent_current(base);assert pm==r['source330'];result=[]
 for role in r['results']:
  old=Path(role['source_support_path']);new=Path(role['output'])
  assert sha(old)==role['source_support_sha256']and sha(new)==role['output_sha256']
  with np.load(old)as z:original={k:z[k]for k in z.files}
  with np.load(new)as z:current={k:z[k]for k in z.files}
  for k in original:assert np.array_equal(original[k],current[k])
  assert str(current['admissibility_source_support_sha256'].item())==sha(old)and int(current['admissibility_max_source_region_hops'])==1
  adjacency=defaultdict(set);mat=role['material']
  for region in np.unique(parent['regions'][parent['materials']==mat]):adjacency[int(region)].add(int(region))
  for e in role['authenticated_region_edges']:
   assert edge_ok(parent,mat,e['regions'],e['source_parent_tets'],e['shared_source_node_rows'])
   a,b=e['regions'];adjacency[a].add(b);adjacency[b].add(a)
  if role['authenticated_region_edges']:
   e=role['authenticated_region_edges'][0]
   assert not edge_ok(parent,31-mat,e['regions'],e['source_parent_tets'],e['shared_source_node_rows'])
   assert not edge_ok(parent,mat,e['regions'],e['source_parent_tets'],e['shared_source_node_rows'][:2]+[int(parent['tets'].max())+1])
   assert not edge_ok(parent,mat,e['regions'],[e['source_parent_tets'][0]]*2,e['shared_source_node_rows'])
  g=original['group_tuples'];ids=original['cell_group_ids'];off=original['cell_group_offsets'];count=len(off)-1
  direct_children=np.repeat(np.arange(count,dtype=np.int64),np.diff(off));direct_regions=g[ids,2];factor=int(parent['regions'].max())+1
  direct=np.unique(direct_children*factor+direct_regions)
  dc=direct//factor;dr=direct%factor
  expected=[]
  for region in np.unique(dr):
   children=dc[dr==region]
   expected.append((children[:,None]*factor+np.array(sorted(adjacency[int(region)]))[None,:]).reshape(-1))
  expected=np.unique(np.concatenate(expected))
  off=current['cell_admissible_region_offsets'];regions=current['cell_admissible_region_ids']
  assert off.shape==(count+1,)and off[0]==0 and off[-1]==len(regions)and np.all(np.diff(off)>0)
  actual=np.repeat(np.arange(count,dtype=np.int64),np.diff(off))*factor+regions
  assert np.array_equal(actual,expected)
  result.append({'role':role['role'],'cells':count,'unchanged_original_arrays':len(original),'verified_source_face_adjacency_edges':len(role['authenticated_region_edges']),'verified_admissible_cell_region_slots':len(actual)})
 report={'status':'PASS_INDEPENDENT_AUTHENTICATED_CONNECTED_SUPPORT','physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN','producer_receipt_sha256':sha(rp),'controls':['Cross-material source edge rejected','Fabricated shared node face rejected','Same-parent non-edge rejected','Every original P1/node/source-support array unchanged','Every allowed child region exactly original-or-one-hop authenticated; no transitive closure'],'results':result,'elapsed_s':time.monotonic()-start}
 out=folder/'independent-verification.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
