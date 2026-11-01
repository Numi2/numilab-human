from pathlib import Path
import ast,hashlib,io,json,subprocess,tarfile,zipfile,struct
import numpy as np
H=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
OUT=H/"quality-volume-archive-chunk-027-plan";PLAN=json.loads((OUT/"selection-plan.json").read_text())
TAG="human-quality-volume-failed-unadmitted-20261101-chunk-027";ASSET="quality-volume-archive-chunk-027.tar.zst";EXPECTED="1b35914b6789aa4aaf7d00ce2356b1e811a59f1b8360baa11469d9e085c0ad40";SIZE=80034326
def sha(b):return hashlib.sha256(b).hexdigest()
def read_header(fp):
 v=np.lib.format.read_magic(fp)
 if v==(1,0):return np.lib.format.read_array_header_1_0(fp)
 if v==(2,0):return np.lib.format.read_array_header_2_0(fp)
 if v==(3,0):
  n=struct.unpack("<I",fp.read(4))[0]
  d=ast.literal_eval(fp.read(n).decode().strip());return tuple(d["shape"]),bool(d["fortran_order"]),np.dtype(d["descr"])
 raise RuntimeError("bad NPY")
remote=subprocess.run(["gh","release","download",TAG,"--repo","Numi2/numilab-human","--pattern",ASSET,"--output","-"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
if remote.returncode:raise RuntimeError(remote.stderr.decode("utf8","replace")[-1000:])
arc=remote.stdout;assert len(arc)==SIZE and sha(arc)==EXPECTED
dec=subprocess.run(["/opt/homebrew/bin/zstd","-dc"],input=arc,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=240)
if dec.returncode:raise RuntimeError(dec.stderr.decode("utf8","replace")[-1000:])
target=next(r for r in PLAN["files"] if r["role"]=="Skin");need=target["expected_sha256_from_source_receipt"];blob=None;idx=None
with tarfile.open(fileobj=io.BytesIO(dec.stdout),mode="r:") as tf:
 for m in tf:
  if m.name=="archive-index.json":idx=json.load(tf.extractfile(m))
  elif m.name=="blobs/"+need:
   blob=tf.extractfile(m).read();assert len(blob)==m.size and sha(blob)==need
assert blob is not None and idx is not None and sha(blob)==need and len(blob)==target["size_bytes_from_stat"]
with zipfile.ZipFile(io.BytesIO(blob)) as z:
 assert z.testzip() is None
 headers=[]
 for name in z.namelist():
  if name.endswith(".npy"):
   with z.open(name) as f:
    shape,fort,dtype=read_header(f);headers.append({"name":name,"shape":list(shape),"fortran_order":bool(fort),"dtype":dtype.str})
fresh=io.BytesIO();fresh.write(bytes(blob));fresh.seek(0);restored=fresh.read();assert sha(restored)==need and len(restored)==len(blob)
arrays={}
with np.load(io.BytesIO(restored),allow_pickle=False) as z:
 assert set(("points_world_m","points_world_m_f32","tetrahedra","material_indices","boundary_faces")).issubset(z.files)
 for name in z.files:
  a=z[name];arrays[name]={"shape":list(a.shape),"dtype":a.dtype.str,"sha256":sha(a.tobytes(order="C"))};del a
out={"schema":"numi-human.volume-archive-all-blobs-exact-ram-restore-supplement.v1","status":"PASS_INDEPENDENT_GH_ARCHIVE_STREAM_AND_EXACT_RESTORE_ALL_TWO_FILES","chunk":"027","github_archive_sha256":sha(arc),"github_archive_size_bytes":len(arc),"selection_plan_sha256":sha((OUT/"selection-plan.json").read_bytes()),"restored_member_path":target["path"],"restored_member_sha256":sha(restored),"restored_member_bytes":len(restored),"zip_crc":"PASS","npy_header_count":len(headers),"npy_headers":headers,"loaded_arrays":arrays,"physical_admission":False,"native_steps":0}
p=OUT/"all-files-exact-restore-supplement.json";p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":out["status"],"archive_sha256":sha(arc),"restored_path":target["path"],"restored_sha256":sha(restored),"bytes":len(restored),"headers":len(headers),"arrays":len(arrays),"receipt":str(p)},sort_keys=True))