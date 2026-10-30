from pathlib import Path
import hashlib,numpy as np,json,importlib.util,sys
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706');p=H/'combine_reconstructed_children_129_root.py';sp=importlib.util.spec_from_file_location('combine129',p);m=importlib.util.module_from_spec(sp);sys.modules[sp.name]=m;sp.loader.exec_module(m)
out=H/'reconstructed-child-adapter-controls-134-root';out.mkdir(exist_ok=False);checks=[]
def put(name,p,t,mat):
 path=out/(name+'.npz');np.savez(path,points_world_m=p,tetrahedra=t,material_indices=mat);return path,m.sha(path)
P=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],np.float32);T=np.array([[0,1,2,3]],np.int64)
path,pin=put('positive',P,T,np.array([15],np.int32));m.load_positive(path,pin,15);checks.append('positive_F32_control')
for name,pp,tt,mat in [('negative',P,T[:,[0,2,1,3]],np.array([15])),('zero',np.zeros((4,3),np.float32),T,np.array([15])),('wrong_material',P,T,np.array([0])),('invalid_index',P,np.array([[0,1,2,4]]),np.array([15]))]:
 path,pin=put(name,pp,tt,mat)
 try:m.load_positive(path,pin,15)
 except ValueError:checks.append(name+'_rejected')
 else:raise AssertionError(name+' accepted')
path,pin=put('attributed_legacy_region',P,T,np.array([17],np.int32))
sc=out/'sidecar.npz';np.savez(sc,material_index=np.array([16]),current_candidate_region_attribute=np.array([17]),output_cell_row=np.array([0]));m.load_positive(path,pin,16,sc);checks.append('scoped_material_sidecar_used')
bad=out/'badsidecar.npz';np.savez(bad,material_index=np.array([16]),current_candidate_region_attribute=np.array([17]),output_cell_row=np.array([1]))
try:m.load_positive(path,pin,16,bad)
except ValueError:checks.append('sidecar_cell_domain_mismatch_rejected')
else:raise AssertionError('badsidecar')
m.load_positive(m.SKIN,m.SKIN_SHA,15);checks.append('actual_743497_cell_Skin072_passes_F32_gate')
report={'status':'PASS','adapter':str(p),'adapter_sha256':m.sha(p),'checks':checks,'runtime':0,'physical_admission':False}
(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
