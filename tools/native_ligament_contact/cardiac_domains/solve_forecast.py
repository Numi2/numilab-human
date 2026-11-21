from pathlib import Path
import hashlib,json,struct,subprocess,sys
import numpy as np

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
base,entry,forecast,witness,out=map(Path,sys.argv[1:])
source=base/'cardiac-coordinates-live.bin'
raw=forecast.read_bytes();raw=raw[:len(raw)//16*16];assert len(raw)>8201*16
first=(base/'cardiac-targets-live.csv.volumes.bin').read_bytes()
assert raw[:len(first)]==first and len(first)==8000*16
f=json.loads(witness.read_text())['first_failure_latch']
q=np.array(f['common_first']+f['common_second'][:3],dtype='f4')
points=out/'targets.bin';points.write_bytes(raw)
probe=base/'cardiac-coordinates-probe';shader=base/'cardiac-coordinates-probe.metallib'
seen=set()
for path in sorted(entry.glob('resting-clearance-and-atrium-00[23456]/*/native/common-field-parameters-f32.bin')):
    boxes=path.parent/'common-field-domain-boxes-f32.bin';identity=(sha(path),sha(boxes))
    if identity in seen:continue
    seen.add(identity);dest=out/'coordinates.bin'
    subprocess.run([probe,shader,points,path,boxes,dest],check=True,stdout=subprocess.DEVNULL)
    buf=dest.read_bytes()
    if buf[:8000*64]!=source.read_bytes():continue
    state=np.frombuffer(buf,'<f4').reshape(-1,16)
    first_new=state[int(f['step'])]
    if not np.array_equal(first_new[:7],q):raise ValueError('Live failure coordinates disagree with the forecast')
    meta=dict(steps=len(state),geometry_map=str(path),geometry_map_sha256=identity[0],domain_boxes=str(boxes),domain_boxes_sha256=identity[1],forecast_targets_sha256=sha(points),forecast_coordinates_sha256=sha(dest),source_reference_sha256=sha(source),matched_reference_states=8000,matched_native_failure_step=8200,probe_sha256=sha(probe),shader_sha256=sha(shader))
    (out/'forecast-parity.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(meta,flush=True);break
else:raise ValueError('No byte-matched native source')
