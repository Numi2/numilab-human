"""Losslessly compact superseded generated baseline sidecars on SSH Numi."""
from hashlib import file_digest
import json
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent.parent
run = Path(__file__).resolve().parent
zstd = "/opt/homebrew/bin/zstd"

def sha(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()

def restored_sha(path):
    with subprocess.Popen([zstd,"-qdc",path],stdout=subprocess.PIPE) as proc:
        result = file_digest(proc.stdout,"sha256").hexdigest()
        if proc.wait():
            raise RuntimeError(f"Could not restore {path}")
    return result

candidates = {}
roles = root/"paired-live-parent-native-role-flow"
for role in ("skin","fat"):
    pin = json.loads((roles/f"{role}-recorded-flow-receipt.json").read_text())
    for name,data in pin["flow_csv"].items():
        candidates[roles/name] = data["sha256"]
reg = root/"paired-repaired-registration-live-parent"
pin = json.loads((reg/"receipt.json").read_text())
for name in ("combined-node-map.csv","node-parent-registration.csv","registered-I.csv"):
    candidates[reg/name] = pin["outputs"][name]
diagnostic = root/"floor-native-live-parent-017/native"
pack = diagnostic/"accepted-geometry/step-12.mrvpack"
audit = json.loads((diagnostic.parent/"exact-skin-surface.json").read_text())
candidates[pack] = audit["accepted_pack_sha256"]
# Preserve the validation artifacts, including their original digest.
for scene in ("016", "017"):
    field = root/f"floor-native-live-parent-{scene}/native/common-field-map-f32.bin"
    if field.is_file():
        candidates[field] = sha(field)
before = os.statvfs(root)
for path, expected in candidates.items():
    archive = Path(str(path)+".zst")
    temporary = Path(str(archive)+".incomplete")
    if archive.exists():
        assert restored_sha(archive) == expected
    else:
        assert sha(path) == expected
        subprocess.run([zstd,"-q","-3","-T2",path,"-o",temporary],check=True)
        assert restored_sha(temporary) == expected
        temporary.rename(archive)
    if path.exists():
        assert sha(path) == expected
        path.unlink()
    print(f"{expected}  {path}  restore={archive}",flush=True)
after = os.statvfs(root)
print(f"free_before_bytes={before.f_bavail * before.f_frsize} "
      f"free_after_bytes={after.f_bavail * after.f_frsize}",flush=True)
