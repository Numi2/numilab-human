#!/usr/bin/env python3
from pathlib import Path
import importlib.util, json, hashlib, numpy as np
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location("npz_complete_actual_926b",HERE/"npz_cow_append_complete_926b.py")
M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def sha(p): return hashlib.file_digest(open(p,"rb"),"sha256").hexdigest()
def main():
 src=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/all-quality-923-fixed-boundary-root/flips/fat-fixed-boundary-quality-queue.npz")
 td=HERE/"actual-923-complete-test-only";td.mkdir(exist_ok=True)
 dst=td/"fat-full-payload-cow-test.npz"
 if dst.exists(): raise FileExistsError(dst)
 before=src.stat(); before_sha=sha(src)
 with np.load(src,allow_pickle=False) as z: payload={k:z[k] for k in z.files}
 payload["npz_emitter_test_scalar"]=np.asarray(926,dtype=np.int64)
 rep=M.write_npz_clone_complete(src,dst,payload)
 assert sha(src)==before_sha and rep["parent_unchanged"]
 assert rep["readback_arrays_bitwise_verified"]==167 and rep["unchanged_compressed_members_bitwise_verified"]==166,rep
 assert rep["changed_existing_keys"]==[] and rep["new_keys"]==["npz_emitter_test_scalar"],rep
 outstat=dst.stat(); clone_sha=rep["output_sha256"]
 with np.load(dst,allow_pickle=False) as z:
  assert len(z.files)==167 and z["npz_emitter_test_scalar"].item()==926
 dst.unlink();td.rmdir()
 receipt={"schema":"numi.npz-cow-emitter-complete-payload-actual-test.v1",
  "status":"PASS_ACTUAL_923_FAT_COMPLETE_PAYLOAD_ONE_SCALAR",
  "parent_path":str(src),"parent_sha256_before":before_sha,"parent_sha256_after":sha(src),
  "parent_size_bytes":int(before.st_size),"test_clone_sha256":clone_sha,
  "test_clone_size_bytes":int(outstat.st_size),"test_clone_deleted":not dst.exists(),
  "input_payload_array_count":rep["input_payload_array_count"],
  "output_array_count":rep["readback_arrays_bitwise_verified"],
  "unchanged_compressed_members_verified":rep["unchanged_compressed_members_bitwise_verified"],
  "changed_existing_keys":rep["changed_existing_keys"],"new_keys":rep["new_keys"],
  "zip_crc":rep["zip_crc"],"metadata_change":"added disposable int64 scalar npz_emitter_test_scalar=926"}
 path=HERE/"actual_923_complete_payload_receipt.json";path.write_text(json.dumps(receipt,indent=2)+"\n")
 print(json.dumps(receipt,sort_keys=True))
if __name__=="__main__":main()

