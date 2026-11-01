#!/usr/bin/env python3
from pathlib import Path
import tempfile, zipfile, numpy as np, importlib.util, hashlib
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location("npz_complete_926b",HERE/"npz_cow_append_complete_926b.py")
M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def sha(p): return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def arrays(path):
 with np.load(path,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def main():
 with tempfile.TemporaryDirectory(dir=HERE) as td:
  d=Path(td); p=d/"parent.npz"
  vals={"scalar":np.asarray(5,dtype=np.int16),
        "matrix":np.asfortranarray(np.arange(12,dtype=np.float64).reshape(3,4)),
        "signed_zero":np.asarray([0.,-0.],dtype=np.float64),
        "text":np.asarray("日本語",dtype="<U4"),
        "remove":np.asarray([1,2,3],dtype=np.int32)}
  np.savez(p,**vals); parent_sha=sha(p)
  # Identical complete payload emits no replacement/deletion and retains every member.
  r1=M.write_npz_clone_complete(p,d/"same.npz",arrays(p))
  assert r1["changed_existing_keys"]==[] and r1["new_keys"]==[] and r1["deleted_keys"]==[]
  assert len(r1["unchanged_payload_keys"])==5 and r1["unchanged_compressed_members_bitwise_verified"]==5,r1
  assert sha(p)==parent_sha
  # A one-scalar change retains all unrelated compressed members.
  changed=arrays(p); changed["scalar"]=np.asarray(7,dtype=np.int16)
  r2=M.write_npz_clone_complete(p,d/"changed.npz",changed)
  assert r2["changed_existing_keys"]==["scalar"] and len(r2["unchanged_payload_keys"])==4,r2
  assert r2["unchanged_compressed_members_bitwise_verified"]==4,r2
  # Complete-payload omission means deletion, not an implicit keep.
  omitted=arrays(p); omitted.pop("remove")
  r3=M.write_npz_clone_complete(p,d/"deleted.npz",omitted)
  assert r3["deleted_keys"]==["remove"] and r3["output_member_count"]==4,r3
  with zipfile.ZipFile(d/"deleted.npz") as z:
   assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
  with np.load(d/"changed.npz",allow_pickle=False) as z:
   assert z["scalar"].item()==7 and z["matrix"].flags.f_contiguous
   assert z["signed_zero"].tobytes()==vals["signed_zero"].tobytes()
   assert z["text"].item()=="日本語"
 print({"status":"PASS_COMPLETE_PAYLOAD_DIFF_CONTROLS","identical_noop":"PASS",
        "one_scalar_change":"PASS","omitted_key_deleted":"PASS","parent_unchanged":True})
if __name__=="__main__":main()

