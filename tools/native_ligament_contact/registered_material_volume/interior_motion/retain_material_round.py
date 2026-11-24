"""Use the existing verified witness archive as the base for exact material and motion recovery."""
from pathlib import Path
import argparse,gzip,hashlib,json,os,shutil,subprocess
from archive_cavity_trial import archive_files,sha

def make_delta(f,base,folder):
    h=sha(f);before=sha(base) if base and base.is_file() else None
    info=dict(sha256=h,bytes=f.stat().st_size,base_sha256=before)
    if before==h:
        info['encoding']='identical';return info
    if before and f.stat().st_size>=1<<20:
        delta=folder/(h+'.zst')
        if not delta.exists():
            subprocess.run(['zstd','-q','-1','-T1','--patch-from='+str(base),str(f),'-o',str(delta)],check=True)
        with subprocess.Popen(['zstd','-qdc','--memory=1024MB','--patch-from='+str(base),str(delta)],stdout=subprocess.PIPE) as proc:
            recovered=hashlib.file_digest(proc.stdout,'sha256').hexdigest()
            if proc.wait() or recovered!=h:raise ValueError('Delta did not reproduce '+str(f))
        info.update(encoding='zstd --patch-from',payload=delta.name);return info
    plain=folder/h
    if not plain.exists():subprocess.run(['/bin/cp','-cp',f,plain],check=True)
    info.update(encoding='raw',payload=h);return info

def run(work,parent):
    current=work/'integrated-04/witness-material';prior=parent/'witness-material'
    out=work/'material-recovery';out.mkdir(exist_ok=True)
    base_receipt=json.loads((parent/'witness-archive.json').read_bytes())
    prior_inventory=json.loads((parent/'witness-recovery/reconstruction.json').read_bytes())['entries']
    mapping={}
    for group in ('volume','binding','current-p1','recorded-mass'):
        start=current/group
        for f in sorted(start.rglob('*')):
            if not f.is_file() or f.is_symlink() or f.suffix=='.o' or '__pycache__' in f.parts:continue
            key=str(Path(group)/f.relative_to(start));base=prior/key
            if key not in prior_inventory or not base.is_file() or sha(base)!=prior_inventory[key]['sha256']:base=None
            entry=make_delta(f,base,out)
            if base:entry['base_material_member']=key
            mapping[key]=entry
    runs={'native-interior-step':'accepted-material-07','native-interior-four':'native-witness-05'}
    for name,relative in runs.items():
        for f in sorted((work/name).rglob('*')):
            if not f.is_file() or f.is_symlink():continue
            rel=f.relative_to(work/name);s=str(rel)
            if len(rel.parts)>1 and not (s.endswith(('.csv','.json')) or rel.parts[1]=='accepted-geometry'):continue
            if f.name=='skin-source-influences.bin' or f.name=='registered-skin-native.csv':continue
            oldkey=str(Path(relative)/rel);base=parent/oldkey
            if not base.is_file() or base_receipt['verified_member_sha256'].get(oldkey)!=sha(base):base=None
            entry=make_delta(f,base,out)
            if base:entry['base_release_member']=oldkey
            mapping[str(Path(name)/rel)]=entry
    h=out/'reconstruction.json'
    h.write_text(json.dumps(dict(parent_release=base_receipt['url'],parent_release_sha256=base_receipt['archive_sha256'],parent_material_index='witness-recovery/reconstruction.json',entries=mapping),indent=2)+'\n')
    files={str(Path('delta')/f.name):f for f in out.iterdir() if f.is_file()}
    for n in ('physical-motion.json','full-bone-audit/summary.json','full-bone-audit/volume/result.json','connected-motion.json','interior-motion-03.json','interior-certificate-03.json','interior-certificate-04.json','interior-volume-02/native-import.json','interior-volume-03/result.json','interior-integration/witness-material/current-p1/result.json','integrated-04/p1.log','integrated-04/allocation.log'):
        files[n]=work/n
    for n in ('condition_connected_motion.py','fit_interior_motion.py','certify_interior_motion.py','import_conditioned_interior.py','reuse_interior_owners.py','bind_interior_scene.py','advance_interior_scene.py','check_full_motion.py','measure_matched_bones.py','retain_material_round.py'):
        files['sources/'+n]=work/n
    receipt=work/'physical-recovery.json'
    archive_files(files,'registered-interior-motion-'+sha(h)[:18]+'.tar.zst',receipt,'Exact same-topology volumetric material, source allocation, native registered owner, and accepted/rejected execution evidence. Reconstruct changed payloads with the named independently verified parent material or release members.')
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('work','parent'):p.add_argument(k,type=lambda x:Path(x).resolve())
    run(**vars(p.parse_args()))
