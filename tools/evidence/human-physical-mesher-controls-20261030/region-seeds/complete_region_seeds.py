from pathlib import Path
import hashlib,importlib.util,json,sys,numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209');H=P/'fine-physical-volume-706'
ROOT=H/'tetgen-region-seed-completion-056'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def components(tets,rows):
 rows=list(map(int,rows));parent={r:r for r in rows};seen={}
 def root(r):
  while parent[r]!=r:parent[r]=parent[parent[r]];r=parent[r]
  return r
 for r in rows:
  tet=tets[r]
  for omit in range(4):
   face=tuple(sorted(int(tet[j])for j in range(4)if j!=omit))
   if face in seen:
    a,b=root(r),root(seen[face]);parent[max(a,b)]=min(a,b)
   else:seen[face]=r
 groups={}
 for r in rows:groups.setdefault(root(r),[]).append(r)
 return list(groups.values())
def seeds_from_unowned(points,tets,attrs,material,contains):
 if material not in (15,16)or not np.isin(attrs,[0,material]).all():raise ValueError('unexpected material attribute')
 if len(points)==0 or not np.isfinite(points).all():raise ValueError('invalid coordinates')
 rows=np.flatnonzero(attrs==0);p=np.asarray(points,dtype=np.float32).astype(np.float64);records=[]
 for group in components(tets,rows):
  best=None
  for r in group:
   q=p[tets[r]];v=float(np.dot(q[1]-q[0],np.cross(q[2]-q[0],q[3]-q[0]))/6)
   if not np.isfinite(v)or v==0:raise ValueError('unowned component has collapsed F32 cell')
   center=q.mean(0)
   if not contains(center):raise ValueError('unowned component cell outside existing material shell')
   if best is None or abs(v)>best[0]:best=(abs(v),r,center)
  records.append({'unowned_cell_rows':group,'seed_cell_row':best[1],'point_world_m':best[2].tolist(),'material':material})
 return records
def complete_poly(source,dest,seeds,owner):
 text=Path(source).read_text();lines=[x for x in text.splitlines()if x.strip()and not x.lstrip().startswith('#')]
 nh=list(map(int,lines[0].split()));assert nh[1:]==[3,0,1];pos=nh[0]+1;fh=list(map(int,lines[pos].split()));assert fh[1]==1;pos+=1+2*fh[0]
 holes=int(lines[pos]);pos+=1+holes;regions=int(lines[pos]);assert pos+1+regions==len(lines)
 expected={int(seed['material'])for seed in seeds}
 if len(expected)!=1 or not expected.issubset({15,16}):raise ValueError('region seed material ownership mismatch')
 for line in lines[pos+1:]:
  a=line.split()
  if len(a)!=6 or int(a[4])not in expected:raise ValueError('existing region material ownership mismatch')
 lines[pos]=str(regions+len(seeds))
 for i,seed in enumerate(seeds,regions+1):
  p=seed['point_world_m'];lines.append(f'{i} {p[0]:.17g} {p[1]:.17g} {p[2]:.17g} {seed["material"]} 0')
 Path(dest).write_text('# Region completion from verified interior cells; physical admission false.\n'+'\n'.join(lines)+'\n')
 owner.validate_poly_sections(Path(dest),nh[0],fh[0],holes,regions+len(seeds))
 # Every source point, facet and hole byte-token is unchanged; only region count/records appended.
 return {'points':nh[0],'faces':fh[0],'holes':holes,'existing_regions':regions,'added_regions':len(seeds)}
def main():
 path=H/'native-domain-217-source-004/build_native_domain_217.py';sys.path[:0]=[str(path.parent),str(P.parent/'authoring-deps')]
 spec=importlib.util.spec_from_file_location('seed_completion_owner056',path);owner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=owner;spec.loader.exec_module(owner)
 src=H/'tetgen-explicit-no-improvement-053/skin037-run-001';out=ROOT/'skin037-preparation-001';out.mkdir(exist_ok=False)
 result=json.loads((src/'result.json').read_text());assert sha(src/'skin.poly')==result['source_poly_sha256']
 with (src/'skin.1.node').open()as f:head=f.readline();nodes=np.loadtxt(f,comments='#')
 with (src/'skin.1.ele').open()as f:eh=f.readline();ele=np.loadtxt(f,comments='#')
 assert np.array_equal(nodes[:,0],np.arange(1,len(nodes)+1))
 tets=ele[:,1:5].astype(np.int64)-1;attrs=ele[:,5];points=nodes[:,1:4]
 z=np.load(H/'geometry-feasibility-008-warning037/skin-shell.npz');shell={k:z[k]for k in z.files};whole=owner._manifold(shell['points_world_m'],shell['faces'],shell['face_marker'])
 q=points[tets];vol=np.einsum('ij,ij->i',q[:,1]-q[:,0],np.cross(q[:,2]-q[:,0],q[:,3]-q[:,0]))/6
 if abs(np.abs(vol).sum()-whole.volume())>max(1e-15,1e-9*abs(whole.volume())):raise ValueError('mesh and source shell volume disagree')
 seeds=seeds_from_unowned(points,tets,attrs,15,lambda p:owner._point_inside(whole,p))
 counts=complete_poly(src/'skin.poly',out/'skin.poly',seeds,owner)
 receipt={'status':'VERIFIED_INTERIOR_REGION_SEEDS_PREPARED_ONLY','source_poly_sha256':sha(src/'skin.poly'),'source_node_sha256':sha(src/'skin.1.node'),'source_ele_sha256':sha(src/'skin.1.ele'),'completed_poly_sha256':sha(out/'skin.poly'),'counts':counts,'seeds':seeds,'physical_admission':False,'runtime':0,'accepted_steps':0}
 (out/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
 print(json.dumps({k:v for k,v in receipt.items()if k!='seeds'}))
if __name__=='__main__':main()

