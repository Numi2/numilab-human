"""Make standard Zstandard patches against the previously verified native material archives."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from compile_connected_joint import digest

def run(work):
    q=work.parent;stage=work/'witness-material';folder=work/'witness-recovery';folder.mkdir(exist_ok=True)
    roots={'volume':q/'qualified-material','binding':work/'binding','current-p1':work/'current-p1','recorded-mass':work/'recorded-mass'}
    entries={}
    for group,prior in roots.items():
        src=stage/group
        for f in sorted(src.rglob('*')):
            if not f.is_file() or f.is_symlink() or f.suffix=='.o' or '__pycache__' in f.parts:continue
            rel=f.relative_to(src);key=str(Path(group)/rel)
            base=prior/rel
            if group=='volume' and rel.as_posix()=='current.node':base=work/'selected-pose/current.node'
            h=digest(f);old=digest(base)if base.is_file()else None
            result=dict(sha256=h,bytes=f.stat().st_size)
            if old:result.update(base_sha256=old,base_group=group,base_path=str(rel))
            if h==old:result['encoding']='identical'
            elif f.stat().st_size>=1024*1024 and old:
                new=folder/(h+'.zst')
                cmd=['zstd','-q','--threads=1','--patch-from='+str(base),'-1',str(f),'-o',str(new)]
                if not new.exists():subprocess.run(cmd,check=True)
                with subprocess.Popen(['zstd','-qdc','--memory=1024MB','--patch-from='+str(base),str(new)],stdout=subprocess.PIPE)as proc:
                    check=hashlib.file_digest(proc.stdout,'sha256').hexdigest()
                    if proc.wait() or check!=h:raise ValueError('Delta failed reconstruction '+key)
                result.update(encoding='zstd -d --memory=1024MB --patch-from',payload=new.name)
            else:
                new=folder/h
                if not new.exists():shutil.copy2(f,new)
                result.update(encoding='raw',payload=new.name)
            entries[key]=result
    (folder/'reconstruction.json').write_text(json.dumps(dict(entries=entries,reference_volume_archive=json.loads((q/'material-recovery.json').read_bytes())['url'],reference_source_carriers_archive=json.loads((work/'recovery.json').read_bytes())['url'],reference_build_archive=json.loads((work/'native-recovery.json').read_bytes())['url'],loaded_node_reference='volume/current.node uses selected-pose/current.node from the source carriers archive.'),indent=2)+'\n')
if __name__=='__main__':run(Path(__file__).resolve().parent)
