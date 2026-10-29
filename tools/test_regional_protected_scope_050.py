from pathlib import Path
import importlib.util,json,hashlib
import numpy as np
P=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209')
A=P/'reduced-physical-boundary-400/attempt-044-minimal-coupled-bone32-buffer'
TOOL=A/'authoring-source-chain/audit_local_mask_preservation_scoped_050.py'
REPORT=P/'reduced-physical-boundary-400/attempt-045-outerO-local-fine-joint-cap/run-001-ring1/mask-preservation-audit-001/report.json'
OUT=A/'regional-protected-scope-control-007'
def main():
 if OUT.exists():raise FileExistsError(OUT)
 spec=importlib.util.spec_from_file_location('scoped050',TOOL);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 b=m.arrays(m.BASE);report=json.loads(REPORT.read_text())
 rows={r['source_node']:r for material in report['protected_vertex_inventory'] for r in material['rows']}
 assert len(rows)==149
 protect=set(rows);source_roles={n:(r['role'],0) for n,r in rows.items()}
 cf={int(n):i for i,n in enumerate(b['parent_volume_node_row']) if n>=0}
 full=m.protected_material_scope(b,protect,source_roles,cf,False)
 regional=m.protected_material_scope(b,protect,source_roles,cf,True)
 results={}
 for name in ['skin','fat']:
  inv={r['source_node']:r for v in report['protected_vertex_inventory'] if v['material']==name for r in v['rows']}
  req=set(regional[name]['required_source_nodes']);exc=set(regional[name]['outside_regional_source_nodes'])
  assert not req&exc and req|exc==set(full[name]['required_source_nodes'])
  material=15 if name=='skin' else 16
  selected=(b['material_index']==material)&(b['geometric_donor_body']==23)&(b['anatomical_region_id']==23)
  expected_used=set(map(int,np.unique(b['volume_tetrahedra'][selected])))
  assert req=={n for n in inv if cf[n] in expected_used}
  missed=[n for n in req if not inv[n]['exact_f32_boundary_point_present']]
  outmissing=[n for n in exc if not inv[n]['exact_f32_boundary_point_present']]
  results[name]={'required_count':len(req),'out_of_scope_count':len(exc),'genuine_missing_source_nodes':sorted(missed),'outside_region_missing_source_nodes':sorted(outmissing)}
 assert results['skin']['genuine_missing_source_nodes']==[]
 assert results['fat']['genuine_missing_source_nodes']==[5958,5994]
 assert len(results['fat']['outside_region_missing_source_nodes'])==11
 bad=dict(b);bad['anatomical_region_id']=np.full_like(b['anatomical_region_id'],-1)
 try:m.protected_material_scope(bad,protect,source_roles,cf,True)
 except ValueError as e:assert 'no source cells' in str(e)
 else:raise AssertionError('empty regional material was admitted')
 try:m.protected_material_scope(b,protect,source_roles,{},True)
 except ValueError as e:assert 'no baseline child' in str(e)
 else:raise AssertionError('missing source child was admitted')
 bad=dict(b);bad['material_index']=b['material_index'][:-1]
 try:m.protected_material_scope(bad,protect,source_roles,cf,True)
 except ValueError as e:assert 'malformed' in str(e)
 else:raise AssertionError('inconsistent array lengths were admitted')
 OUT.mkdir()
 receipt={'status':'PASS','scope':'source-only protected-point audit scoping regression','cases':['default whole-body scope preserved','regional scope derived from complete same-label source-tet group','two true regional failures remain failures','eleven unrelated-body points classified outside partial domain','empty source group rejects','missing source-parent node rejects','malformed domains reject'],'results':results,'runtime_created':0,'accepted_steps':0,'physical_admission':False,'inputs_sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [TOOL,REPORT,m.BASE,Path(__file__)]}}
 (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(receipt))
if __name__=='__main__':main()
