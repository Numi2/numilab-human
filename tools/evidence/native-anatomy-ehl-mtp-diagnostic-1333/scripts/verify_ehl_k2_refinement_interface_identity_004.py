from pathlib import Path
import json,numpy as np,hashlib
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276');O=A/'ehl-k2-refinement-interface-identity-004';O.mkdir(exist_ok=False)
RP=A/'ehl-current-anatomy-mtp-two-full-audit-003/report.json';r=json.loads(RP.read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
out={'scope':'Compare restored EHL and conforming refinement at identical K2 accepted poses. This checks new target pairs and exact unchanged crossing triangles, not validity of existing interfaces.','pins':{str(RP):sha(RP),str(Path(__file__)):sha(Path(__file__))},'rows':[]}
for sid in (23,24):
 d=A/f'ehl-restored-boundary-batch-001/stable-{sid}';bp=d/f'micrometre-resolution-003/stable-{sid}-geometric-reference-row-patch.npz';cp=d/f'posed-conforming-refinement-001/stable-{sid}-unadmitted-row-patch.npz'
 b=dict(np.load(bp));c=dict(np.load(cp));out['pins'].update({str(p):sha(p) for p in (bp,cp)})
 n=len(b['vertices6']);assert np.array_equal(c['vertices6'][:n,:3],b['vertices6'][:,:3]) and np.array_equal(c['binding_indices'][:n],b['binding_indices']) and np.array_equal(c['weights'][:n],b['weights'])
 lookup={tuple(sorted(map(int,f))):i for i,f in enumerate(b['faces'])};assert len(lookup)==len(b['faces'])
 mapping={i:lookup.get(tuple(sorted(map(int,f)))) for i,f in enumerate(c['faces'])}
 row={'stable_id':sid,'all_original_coordinates_and_maps_exact':True,'unchanged_triangle_count':sum(x is not None for x in mapping.values()),'new_triangle_count':sum(x is None for x in mapping.values()),'poses':[]};out['rows'].append(row)
 for ent in r['poses']:
  br=next(x for x in ent['rows'] if x['id']==sid and x['variant']=='restored');cr=next(x for x in ent['rows'] if x['id']==sid and x['variant']=='refined')
  bs={(tuple(x['key']),int(i),int(j)) for x in br['target_pairs'] for i,j in x['pairs']}
  cs={(tuple(x['key']),mapping[int(i)],int(j)) for x in cr['target_pairs'] for i,j in x['pairs']}
  unmap=[(x['key'],int(i),int(j)) for x in cr['target_pairs'] for i,j in x['pairs'] if mapping[int(i)] is None]
  row['poses'].append({'step':ent['step'],'unmapped_crossing_triangles':unmap,'exact_crossing_triangle_pairs_unchanged':not unmap and bs==cs,'baseline_count':len(bs),'candidate_count':len(cs)})
out.update(complete=True,inputs_unchanged=all(sha(p)==v for p,v in out['pins'].items()),all_original_crossing_triangles_and_target_pairs_exact=all(x['exact_crossing_triangle_pairs_unchanged'] for r in out['rows'] for x in r['poses']))
(O/'report.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({'all_crossing_triangles_exact':out['all_original_crossing_triangles_and_target_pairs_exact'],'rows':[{'id':r['stable_id'],'unchanged':r['unchanged_triangle_count'],'new':r['new_triangle_count'],'generated_crossing_counts':[len(x['unmapped_crossing_triangles']) for x in r['poses']]} for r in out['rows']]}))
