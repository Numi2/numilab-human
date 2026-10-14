#!/usr/bin/env python3
"""Late N22 preparation using source rows proven unchanged from N27."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
D=A/'full-anatomy-delivery-audit-1378'
OLD=D/'intercostal-retained-motion-preparation-028-001/prepare.py'
O=D/'intercostal-late-settled-preparation-028-002'
N22=Path('/Users/n/numi-human-resting-evidence-20261005/native-lung-free-apex-two-family-composition-1178/composition-024-attempt5/final/resting-thorax.nhanatomy')
PACK=A/'back412-skin-native-022-001-60s/baseline/native-run/accepted-geometry/step-14975.mrvpack'
RC=PACK.with_suffix('.receipt.json')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 assert sha(OLD)=='ead6bc67a66d0d865ddcf0426e8c1fa31081e90bccfea061a5ff396496e87619'
 assert sha(N22)=='1c0c37af76ab3f8e86870fd6cd3abab00b7bcdae51fe934e3461722ca306c241'
 assert sha(PACK)=='b1c25b89e828ecf62b5b7dfd58ac3a38c9f1d420306e5f9b789d04ec29859c3d'
 assert sha(RC)=='84a98b1d4d4098c762798517e91248b6f3f47df3b841ce344c04d593f9b58b79'
 spec=importlib.util.spec_from_file_location('retained_intercostal_preparation',OLD)
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 import sys;sys.path.insert(0,str(mod.ROOT/'src'))
 from numilab_human import resting_anatomy_interface_patch as codec
 t=time.monotonic();_,old=codec.parse_payload(N22);_,baseline=codec.parse_payload(mod.SOURCE)
 rows={}
 for sid in (313,315,317):
  assert np.array_equal(old[sid]['vertices6'],baseline[sid]['vertices6'])
  assert np.array_equal(old[sid]['faces'],baseline[sid]['faces'])
  assert (old[sid]['body_index'],old[sid]['layer'],old[sid]['flags'])==(20,1,0)
  rows[str(sid)]={'vertices_normals_faces_and_owners_byte_identical_to_N27':True}
 del old,baseline
 manifest=O/'late-capture-input.json'
 manifest.write_text(json.dumps({'source':{'path':str(N22),'sha256':sha(N22)},'capture_rows':[{'step':14975,'pack_path':str(PACK),'receipt_path':str(RC),'recorded_pack_sha256':sha(PACK),'receipt_sha256':sha(RC)}],'scope':'One retained late settled N22 pose at29.950001422548667simulatedseconds; not a new native run.'},indent=2)+'\n')
 mod.CAPTURE=manifest;mod.O=O;mod.PINS=dict(mod.PINS);mod.PINS.pop(str(D/'../accepted-capture-inputs-1398-001.json'),None)
 # Replace only the prior capture manifest pin; keep all source/repair pins.
 mod.PINS={p:h for p,h in mod.PINS.items() if Path(p).name!='accepted-capture-inputs-1398-001.json'}
 mod.PINS[str(manifest)]=sha(manifest);mod.PINS[str(N22)]=sha(N22)
 mod.PINS[str(PACK)]=sha(PACK);mod.PINS[str(RC)]=sha(RC)
 (O/'source-binding-preparation.json').write_text(json.dumps({'status':'PASS_EXACT_N22_TO_N27_SOURCE_ROW_BINDING','script_sha256':sha(__file__),'reused_preparation_code_sha256':sha(OLD),'n22_source_sha256':sha(N22),'source_rows':rows,'elapsed_s':time.monotonic()-t,'candidate_admission':False},indent=2)+'\n')
 mod.run()
 report_sha=sha(O/'report.json')
 (O/'coverage-addendum.json').write_text(json.dumps({'report_sha256':report_sha,'actual_capture_steps':[14975],'correction':'Inherited status label is not accurate for this input list. This successor evaluated only one late settled N22 capture, not eight N27 poses.','source_rows_exactly_bound_to_N27':True,'native_run':False,'candidate_admission':False},indent=2)+'\n')
 print(json.dumps({'report_sha256':report_sha,'scope':'one late retained preparation capture only'}))
if __name__=='__main__':main()
