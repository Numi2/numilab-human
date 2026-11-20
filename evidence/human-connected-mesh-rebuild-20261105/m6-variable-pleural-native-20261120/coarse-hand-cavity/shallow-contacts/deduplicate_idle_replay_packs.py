"""Share physical blocks of byte-identical, idle native replay packs on APFS."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from tempfile import TemporaryDirectory

MIN_BYTES=256*1024**2
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def identity(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def available():
    a=os.statvfs(ROOT)
    return a.f_bavail*a.f_frsize


def idle(path):
    result=subprocess.run(["/usr/sbin/lsof",str(path)],capture_output=True)
    return result.returncode==1 and not result.stdout and not result.stderr


def user_metadata(path):
    check=subprocess.run(["/usr/bin/xattr",str(path)],check=True,capture_output=True,text=True)
    # The resource fork and compression attribute belong to transparent
    # filesystem storage. Leave any other attributes and file flags alone.
    return all(s in ("com.apple.decmpfs","com.apple.ResourceFork")
               for s in check.stdout.splitlines())


def clone_verified(source,target,digest,expected):
    saved=target.lstat()
    if identity(saved)!=expected or saved.st_nlink!=1 or not idle(target):
        return None
    if not user_metadata(source) or not user_metadata(target):
        return None
    flags = getattr(stat,"UF_COMPRESSED",0x20)
    if saved.st_flags & ~flags:
        return None
    if sha(target)!=digest:
        raise ValueError(f"Destination changed after grouping: {target}")
    clonefile=ctypes.CDLL(None,use_errno=True).clonefile
    clonefile.argtypes=(ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int)
    clonefile.restype=ctypes.c_int
    with TemporaryDirectory(dir=target.parent) as directory:
        copy=Path(directory)/target.name
        if clonefile(os.fsencode(source),os.fsencode(copy),0)!=0:
            return {"clone_errno":ctypes.get_errno()}
        assert copy.stat().st_size==saved.st_size and sha(copy)==digest
        os.chmod(copy,stat.S_IMODE(saved.st_mode))
        assert copy.stat().st_uid==saved.st_uid and copy.stat().st_gid==saved.st_gid
        os.utime(copy,ns=(saved.st_atime_ns,saved.st_mtime_ns))
        assert identity(target.lstat())==expected and idle(target)
        os.replace(copy,target)
        assert sha(target)==digest
    return {"old_allocated_bytes":saved.st_blocks*512,"size_bytes":saved.st_size}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt",type=Path)
    parser.add_argument("--apply",action="store_true")
    args=parser.parse_args()
    assert not args.receipt.exists()
    candidates=sorted(ROOT.glob("**/resting-human.mrvpack"))
    groups,hashes={},{}
    for p in candidates:
        s=p.lstat()
        if not p.resolve().is_relative_to(ROOT) or not stat.S_ISREG(s.st_mode) or s.st_size<MIN_BYTES:
            continue
        key=identity(s)
        if key not in hashes:
            hashes[key]=sha(p)
            assert identity(p.lstat())==key
        groups.setdefault((s.st_size,hashes[key]),[]).append((p,s,key))
    report={"mode":"apply" if args.apply else "preview",
            "unique_inodes_hashed":len(hashes),"free_before_bytes":available(),"groups":[]}
    with args.receipt.open("x") as f:
        for (_,digest),group in groups.items():
            if len({row[2] for row in group})<2:
                continue
            # Use the least allocated representation. Prefer a long-lived
            # shared source when several representations tie.
            source,_,src_id=min(group,key=lambda r:(r[1].st_blocks,-r[1].st_nlink,str(r[0])))
            if args.apply and not idle(source):
                continue
            row={"sha256":digest,"kept_source":str(source),"details":[]}
            for p,s,ident in group:
                if src_id==ident or s.st_nlink!=1:
                    continue
                if args.apply:
                    assert identity(source.lstat())==src_id
                    result=clone_verified(source,p,digest,ident)
                else:
                    result={"logical_bytes":s.st_size,"allocated_bytes":s.st_blocks*512}
                if result:
                    row["details"].append({"path":str(p),**result})
            if row["details"]:
                report["groups"].append(row)
        report["free_after_bytes"]=available()
        json.dump(report,f,indent=2)
        f.write("\n")
    print(json.dumps({"groups":len(report["groups"]),"unique_inodes":len(hashes),
                      "free_before_bytes":report["free_before_bytes"],
                      "free_after_bytes":report["free_after_bytes"]}))


if __name__=="__main__":
    main()
