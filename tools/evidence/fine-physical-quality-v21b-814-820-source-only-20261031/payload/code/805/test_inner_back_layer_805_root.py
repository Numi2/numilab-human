from pathlib import Path
import hashlib,importlib.util,sys,json,numpy as np
h=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
def imp(n,p):
 sp=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(sp);sys.modules[n]=m;sp.loader.exec_module(m);return m
new=imp('audit805_control',h/'combined-surface-diff-audit-270/audit_combined_bounded_inner_layer_surface_diff_805_root.py')
old=imp('audit775_control',h/'combined-surface-diff-audit-270/audit_combined_bounded_point_surface_diff_775_root.py')
st=json.loads((h/'all-quality-799-root/round002/trial01/verified-state.json').read_text())
scope=np.load(h/'current-back412-scope-597-root/run003/output-001/protected-back412-aabb-current-O558-I50-D264-rows.npz')
routes=json.loads((h/'verified-volume-preparation-010-merged038005/rim/protected149-current-routes.json').read_text())['surface_routes']
tests=[]
def check(name,fun,accept):
 try:fun();actual=True
 except ValueError as e:actual=False;detail=str(e)
 assert actual==accept,(name,actual,accept)
 tests.append({'case':name,'accepted':actual,**({} if actual else {'reason':detail})})
for role in ['I','D','O']:
 a=np.load(st['source_surfaces'][role]['path']);p=a['points_world_m_f32'].copy();f=a['faces'];edge=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1);e,ct=np.unique(edge,axis=0,return_counts=True);fixed=set(map(int,e[ct==1].ravel()))
 for rr in routes:
  if rr['role']==role:fixed.update(map(int,np.flatnonzero(np.all(p==np.asarray(rr['target_world_m_f32'],np.float32),axis=1))))
 rows=[int(x) for x in scope[role+'_vertex_rows'] if int(x) not in fixed]
 assert rows
 row=rows[0];c=p.copy();c[row,0]=np.nextafter(c[row,0],np.float32(np.inf))
 entry={'maximum_displacement_m':5.e-5,'inner_back_layer_repair':True}
 check(role+'_new_identity',lambda:new.bounded_point_identity(p,p,role,entry,f),role!='O')
 check(role+'_old_rejects_core_movement',lambda:old.bounded_point_identity(p,c,role,{'maximum_displacement_m':5.e-5},f),False)
 check(role+'_new_bounded_core_movement',lambda:new.bounded_point_identity(p,c,role,entry,f),role!='O')
 if role!='O':
  check(role+'_missing_flag_preserves_old_protection',lambda:new.bounded_point_identity(p,c,role,{'maximum_displacement_m':5.e-5},f),False)
  check(role+'_over_policy_cap_rejected',lambda:new.bounded_point_identity(p,c,role,{'maximum_displacement_m':.0005,'inner_back_layer_repair':True},f),False)
  check(role+'_nonboolean_flag_rejected',lambda:new.bounded_point_identity(p,c,role,{'maximum_displacement_m':5.e-5,'inner_back_layer_repair':1},f),False)
 if fixed:
  rr=sorted(fixed)[0];bad=p.copy();bad[rr,0]=np.nextafter(bad[rr,0],np.float32(np.inf))
  check(role+'_rim_or_route_stays_fixed',lambda:new.bounded_point_identity(p,bad,role,entry,f),False)
out=h/'inner-back-layer-805-controls.json'
d={'status':'PASS_POLICY_CONTROLS','audit':{'path':new.__file__,'sha256':hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest()},'tests':tests,'scope':'point-policy checks only; not geometric or physical qualification','physical_admission':False,'native_steps':0}
out.write_text(json.dumps(d,indent=2)+'\n')
print(out,len(tests),hashlib.sha256(out.read_bytes()).hexdigest())
