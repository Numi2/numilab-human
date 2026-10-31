
from pathlib import Path
import hashlib,py_compile,json
H=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
fn="""
def fixed_current_registered_points(p):
 report=H/'fat-inferred-identity-mask-336-root/report.json'
 r=json.loads(report.read_text())
 assert r['status']=='PASS_FAT_REGISTERED_IDENTITY_AND_PENDING_CAVITY_MASK'
 assert r['source_input_sha256']=='4602edc3b1db3617ab610c21f91309db10892c0e28ff7d6ba5c381491d13df2c'
 assert sha(r['source_input'])==r['source_input_sha256']
 mask=H/'fat-inferred-identity-mask-336-root/identity-and-pending-cavity-mask.npz'
 assert sha(mask)==r['mask_sha256']=='b80ced216b95f0d7bdb1245252e11b6742f93dda31793dfe121b1204f61058f5'
 assert sha(r['registered_map'])==r['registered_map_sha256']
 with np.load(r['source_input']) as z:oldp=z['points_world_m']
 with np.load(mask) as z:protected=z['protected_point_rows'];assert str(z['source_input_sha256'].item())==r['source_input_sha256']
 lookup={T.key(x):i for i,x in enumerate(p)};assert len(lookup)==len(p)
 # Existing336 explicit semantic identity ownership; never infer registration
 # merely because a point matches an arbitrary historical interior.
 current=np.asarray([lookup[T.key(oldp[i])] for i in protected],np.int64)
 assert len(current)==len(np.unique(current))==r['protected_existing_point_rows']
 assert np.array_equal(p[current].view(np.uint32),oldp[protected].view(np.uint32))
 fixed=np.zeros(len(p),bool);fixed[current]=True
 return fixed,{'existing336_report':str(report),'existing336_report_sha256':sha(report),'existing336_mask':str(mask),'existing336_mask_sha256':sha(mask),'exact_registered_points_mapped':len(current),'identity_counts':r['identity_counts'],'mapping':'exactF32coordinatebijectionofauthenticated336registeredrows'}
"""
s=(H/'condition_fixed_fat_interior_LP_486_root.py').read_text()
a=""" source=H/'combined-child-handoff-142/fat.node';registration=H/'combined-child-handoff-142/registered-node-map-177-143-713-source008/registered-node-map.csv'
 sp=np.loadtxt(source,skiprows=1,usecols=(1,2,3)).astype(np.float32);reg,counts=registered_fat_source_rows(registration,sp)
 assert np.array_equal(p[:len(sp)][reg].view(np.uint32),sp[reg].view(np.uint32))
 boundary=T.boundary(t);fixed=np.zeros(len(p),bool);fixed[np.unique(boundary)]=True;fixed[:len(sp)]|=reg"""
b=""" registration=H/'combined-child-handoff-142/registered-node-map-177-143-713-source008/registered-node-map.csv'
 fixed,identity_proof=fixed_current_registered_points(p)
 boundary=T.boundary(t);fixed[np.unique(boundary)]=True"""
assert s.count(a)==1;s=s.replace(a,b).replace('def run(a):',fn+'\ndef run(a):')
s=s.replace("'protected_cell_maps':protection_records,","'registered_identity_crosswalk':identity_proof,'protected_cell_maps':protection_records,")
out=H/'condition_fixed_fat_identity_LP_490_root.py';assert not out.exists();out.write_text(s);py_compile.compile(str(out),doraise=True)
v=(H/'verify_fixed_fat_LP_487_root.py').read_text()
a="""source=H/'combined-child-handoff-142/fat.node';sp=np.loadtxt(source,skiprows=1,usecols=(1,2,3)).astype(np.float32)
assert sha(r['registration_map'])==r['registration_map_sha256']
reg,counts=registered_fat_source_rows(Path(r['registration_map']),sp);fixed[:len(sp)]|=reg"""
b="""assert sha(r['registration_map'])==r['registration_map_sha256']
registered,identity_proof=fixed_current_registered_points(p);fixed|=registered
assert identity_proof==r['registered_identity_crosswalk']"""
assert v.count(a)==1;v=v.replace(a,b)
v=v.replace("assert np.array_equal(p[:len(sp)][reg].view(np.uint32),sp[reg].view(np.uint32))","")
v=v.replace('a=argparse.ArgumentParser();',fn+'\na=argparse.ArgumentParser();')
outv=H/'verify_fixed_fat_identity_LP_491_root.py';assert not outv.exists();outv.write_text(v);py_compile.compile(str(outv),doraise=True)
run=(H/'run_fixed_fat_LP_488_root.py').read_text().replace('fixed-fat-interior-LP-488-root','fixed-fat-identity-LP-492-root').replace('condition_fixed_fat_interior_LP_486_root.py','condition_fixed_fat_identity_LP_490_root.py').replace('verify_fixed_fat_LP_487_root.py','verify_fixed_fat_identity_LP_491_root.py')
runner=H/'run_fixed_fat_identity_LP_492_root.py';assert not runner.exists();runner.write_text(run);py_compile.compile(str(runner),doraise=True)
print(json.dumps({'status':'PREPARED_EXACT_CURRENT_IDENTITY_CROSSWALK_CORRECTION','constructor':str(out),'constructor_sha256':sha(out),'verifier':str(outv),'verifier_sha256':sha(outv),'runner':str(runner),'runner_sha256':sha(runner),'previous_failed_trial_retained':str(H/'fixed-fat-interior-LP-488-root'),'physical_admission':False,'native_steps':0}))
