"""Share verified identical profile captures without removing their paths."""
from argparse import ArgumentParser
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from tempfile import TemporaryDirectory

root=Path(__file__).resolve().parent
source=root.parent/"terminal-contact-replay/native/resting-human.mrvpack"
clone=ctypes.CDLL(None,use_errno=True).clonefile
clone.argtypes=(ctypes.c_char_p,ctypes.c_char_p,ctypes.c_int)
clone.restype=ctypes.c_int
def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()

def free():
    statfs=os.statvfs(root)
    return statfs.f_bavail*statfs.f_frsize

parser=ArgumentParser(description=__doc__)
parser.add_argument("receipt",type=Path)
parser.add_argument("trials",nargs="+")
args=parser.parse_args()
assert not args.receipt.exists()
expected=sha(source)
details={"source_sha256":expected,"free_before_bytes":free(),"copies":[]}
for name in args.trials:
    trial=(root/name).resolve()
    assert trial.parent==root
    assert json.loads((trial/"execution.json").read_text())["exit"]==0
    target=trial/"native/resting-human.mrvpack"
    info=target.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_nlink==1
    assert sha(target)==expected
    result=subprocess.run(["/usr/sbin/lsof",str(target)],capture_output=True)
    assert result.returncode==1 and not result.stdout
    with TemporaryDirectory(dir=target.parent) as temp:
        copy=Path(temp)/target.name
        result=clone(os.fsencode(source),os.fsencode(copy),0)
        if result!=0:
            details["copies"].append({"name":name,"clone_errno":ctypes.get_errno()})
            continue
        assert sha(copy)==expected and copy.stat().st_size==info.st_size
        os.chmod(copy,stat.S_IMODE(info.st_mode))
        os.utime(copy,ns=(info.st_atime_ns,info.st_mtime_ns))
        after=target.stat()
        assert (after.st_ino,after.st_ctime_ns,after.st_mtime_ns)==(info.st_ino,info.st_ctime_ns,info.st_mtime_ns)
        os.replace(copy,target)
        assert sha(target)==expected
        details["copies"].append({"name":name,"verified":True})
details["free_after_bytes"]=free()
with args.receipt.open("x") as stream:
    json.dump(details,stream,indent=2)
    stream.write("\n")
print(json.dumps(details))
