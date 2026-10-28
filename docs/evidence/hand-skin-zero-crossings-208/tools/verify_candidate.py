#!/usr/bin/env python3
"""Ordered existing Human geometry checks. A passing result never admits physics."""
from pathlib import Path
import argparse,csv,gzip,hashlib,importlib.util,json,subprocess,sys
import numpy as np
from fractions import Fraction
R=Path(__file__).resolve().parent
# Source snapshots in a published tools directory keep their working owner path.
if not (R/'repair_connected.py').exists():raise RuntimeError('Missing prepared Human authoring owner')
B=R.parent/'publication-206-stage-001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--baseline',type=Path,default=B/'inputs/baseline-806.npz');a=ap.parse_args();p=a.candidate.resolve();o=a.out.resolve();o.mkdir(parents=True,exist_ok=False);h=sha(p)
 with np.load(p,allow_pickle=False)as z:
  X=z['points_world_m'];F=z['source_O_faces'];assert X.dtype==np.float32 and np.isfinite(X).all() and X.shape[1:]==(3,);assert F.ndim==2 and F.shape[1]==3 and F.min()>=0 and F.max()<len(X)
 topo=R/'boolean-prototype-001/topology_check.py';subprocess.run([sys.executable,str(topo),'--candidate',str(p),'--baseline',str(a.baseline),'--out',str(o/'topology.json')],check=True,stdout=subprocess.DEVNULL);t=json.loads((o/'topology.json').read_text())
 if not t['strict_topology_acceptance']:
  print(json.dumps({'status':'REJECTED_TOPOLOGY','candidate_sha256':h,'reasons':t['strict_topology_rejection_reasons'],'native_admission':False}));return 2
 subprocess.run([sys.executable,str(R.parent.parent/'audit_skin_bone_surfaces_159.py'),str(p),str(o/'bone'),'--max-seconds','100'],check=True,stdout=subprocess.DEVNULL);b=json.loads((o/'bone/report.json').read_text());pairs=sum((r.get('result')or{}).get('count',0)for r in b['banks'])
 if not b['complete']or b['bank_count']!=185 or pairs:
  print(json.dumps({'status':'REJECTED_BONE_INTERSECTIONS','pairs':pairs,'native_admission':False}));return 3
 selfaudit=module('ordered_self',R/'boolean-prototype-001/audit_self_envelope006.py');predicate=selfaudit.load_exact();vertices=tuple(tuple(Fraction.from_float(float(x))for x in point)for point in X);records=predicate._records(vertices,tuple(tuple(map(int,f))for f in F));result=predicate._audit_pair(records,records,same_surface=True);witnesses=[]
 for ai,bi in result['triangle_pairs']:
  ai,bi=int(ai),int(bi);points=predicate.triangle_intersection_points(records[ai][0],records[bi][0]);witnesses.append({'face_rows':[ai,bi],'triangle_vertex_rows':[F[ai].tolist(),F[bi].tolist()],'triangle_a_f32':X[F[ai]].tolist(),'triangle_b_f32':X[F[bi]].tolist(),'intersection_points_fraction':[[str(x)for x in v]for v in points]})
 s={'status':'COMPLETE_RATIONAL_SAME_SURFACE_AUDIT','candidate_path':str(p),'candidate_sha256':h,'exact_predicate_owner':str(selfaudit.EXACT_OWNER),'exact_predicate_sha256':sha(selfaudit.EXACT_OWNER),'triangle_count':len(F),'exact_triangle_pair_count':int(result['count']),'aabb_candidate_pairs':int(result['aabb_candidate_pairs']),'allowed_shared_vertex_or_edge_pairs':int(result['allowed_shared_vertex_or_edge_pairs']),'triangle_pairs':witnesses,'complete':True,'geometry_or_native_admission':False};(o/'self.json').write_text(json.dumps(s,indent=2))
 if not s['complete']or s['exact_triangle_pair_count']:
  print(json.dumps({'status':'REJECTED_SELF_INTERSECTIONS','pairs':s['exact_triangle_pair_count'],'native_admission':False}));return 4
 query=module('ordered_query',R/'repair_connected.py');rows=query.query(X,o/'containment');nodes=list(csv.DictReader((o/'containment/nodes.csv').open()));inside=[r for r in rows if r['classification']=='inside'];complete=all(int(r['total_bone_banks'])==185 and sum(int(r[k])for k in ['aabb_proven_outside_banks','parity_outside_banks','inside_banks','ambiguous_banks','invalid_banks'])==185 for r in nodes);ambiguous=sum(int(r['ambiguous_banks'])for r in nodes);invalid=sum(int(r['invalid_banks'])for r in nodes);c={'candidate_sha256':h,'query_rows':len(X),'inside_node_bank_pairs':len(inside),'ambiguous_rows':ambiguous,'invalid_rows':invalid,'complete_bank_accounting':complete,'inside_witnesses':inside,'physical_steps':0,'geometry_or_native_admission':False};(o/'containment/report.json').write_text(json.dumps(c,indent=2))
 for name in ['points.csv','pairs.csv','nodes.csv']:
  f=o/'containment'/name
  with f.open('rb')as src,gzip.open(str(f)+'.gz','wb',compresslevel=1)as dst:
   import shutil;shutil.copyfileobj(src,dst)
  with gzip.open(str(f)+'.gz','rb')as restored:assert hashlib.sha256(restored.read()).hexdigest()==sha(f)
  f.unlink()
 if not complete or inside or ambiguous or invalid:
  print(json.dumps({'status':'REJECTED_CONTAINMENT','inside_pairs':len(inside),'ambiguous':ambiguous,'invalid':invalid,'native_admission':False}));return 5
 print(json.dumps({'status':'PASS_REFERENCE_GEOMETRY_ONLY','candidate_sha256':h,'banks':185,'bone_pairs':0,'self_pairs':0,'inside_pairs':0,'topology_matches_baseline':True,'native_admission':False,'physical_steps':0}));return 0
if __name__=='__main__':sys.exit(main())
