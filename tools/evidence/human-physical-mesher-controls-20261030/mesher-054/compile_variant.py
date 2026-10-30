#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, subprocess, time
P=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209")
H=P/"fine-physical-volume-706"
BASE=H/"tetgen-explicit-no-improvement-053"
OUT=H/"tetgen-explicit-no-improvement-054"
VENDOR=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/arm-local-pocket-remesh-116/local-remesh-118/tetgen-v1.6.0")
def sha(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): d.update(b)
    return d.hexdigest()
base_receipt=json.loads((BASE/"compile.json").read_text())
parent=BASE/"tetgen-explicit-no-improvement.cxx"
source=OUT/"tetgen-explicit-no-improvement-validation.cxx"
expected_variant_source_sha256="255bd025b7b7730ce81348c30df45fe5ad98b7160d5cea29c3bb03c91dae8b1a"
pins={
 "parent_053_source_sha256":"bd0c21ee558ae8de2b9cb790e9b85fcb6224fe505e36bebfc902900cdc9777ac",
 "vendor_tetgen_cxx_sha256":"a10f5e74c3ec45ffda3d9609b1cb6e864f1a628b4c2e3e61cabd49ddfccc36bc",
 "vendor_tetgen_h_sha256":"69b012da7c0b01e6327f5f281ac2d10fc3fb42503d22366c7bd0bc9181bd833b",
 "predicates_o_sha256":"d15c7e1a4dc798fa0b48204e346a9fe031070ccb7789d8b74b910dfafba26f4c",
 "license_sha256":"f15e3301d0f40116aa2e3567963c3dcc8dd45453337dedf592caa73a5390549f",
}
actual={
 "parent_053_source_sha256":sha(parent),
 "vendor_tetgen_cxx_sha256":sha(VENDOR/"tetgen.cxx"),
 "vendor_tetgen_h_sha256":sha(VENDOR/"tetgen.h"),
 "predicates_o_sha256":sha(VENDOR/"predicates.o"),
 "license_sha256":sha(VENDOR/"LICENSE"),
}
if actual != pins:
    raise SystemExit("Pinned source or license mismatch: "+json.dumps({"expected":pins,"actual":actual},sort_keys=True))
if base_receipt.get("source_sha256")!=pins["parent_053_source_sha256"]:
    raise SystemExit("053 compile receipt does not bind expected parent source")
if sha(source)!=expected_variant_source_sha256:
    raise SystemExit("054 source differs from the reviewed validation variant")
binary=OUT/"tetgen"
cmd=["/usr/bin/clang++","-O3","-I",str(VENDOR),str(source),str(VENDOR/"predicates.o"),"-lm","-o",str(binary)]
start=time.monotonic()
r=subprocess.run(cmd,capture_output=True,text=True)
(OUT/"compile.stdout.log").write_text(r.stdout)
(OUT/"compile.stderr.log").write_text(r.stderr)
receipt={
 "argv":cmd,"exit_code":r.returncode,"wall_s":time.monotonic()-start,
 "input_pins":pins,"verified_input_pins":actual,
 "parent_053_compile_receipt_sha256":sha(BASE/"compile.json"),
 "source_sha256":sha(source),"binary_sha256":sha(binary) if binary.exists() else None,
 "changes":{
  "parent_053":"The parser fix and optional-improvement guard are inherited unchanged.",
  "new_validation":"When PLC/quality work is requested but both improvement counts are zero, run improve_mesh's all-cell get_tetqual degeneracy scan without allocating queues or changing mesh cells. Preserve the -c exterior skip and initialize cos_large_dihed.",
  "defaults":"Default flip/iteration settings are unchanged; default path still calls improve_mesh."
 },
 "physical_admission":False,"runtime":0,"accepted_steps":0
}
(OUT/"compile.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+chr(10))
print(json.dumps(receipt,sort_keys=True))
if r.returncode: raise SystemExit(r.returncode)
