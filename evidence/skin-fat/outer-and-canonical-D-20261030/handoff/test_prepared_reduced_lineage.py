from pathlib import Path
import hashlib,importlib.util,json,tempfile
import numpy as np
HERE=Path(__file__).parent
P=HERE.parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
h=load("reduced_native_handoff",HERE/"prepare_tetgen_handoff_697_typedD.py")
r=load("reduced_native_runner",HERE/"run_native_domain_217_typedD_697.py")
source=P/"reduced-surface-lineage-702/O081-lineage-prepared.npz"
with np.load(source,allow_pickle=False) as z:a={k:np.asarray(z[k]) for k in z.files}
base=h.load_owner()
s=h.load_surface_217(base,source,"O")
native=r.load_typed_owner().load_surface(source,"O")
assert len(s["faces"])==76937 and len(s["points"])==38509
assert len(s["reduced_source_group_offsets"])==63
assert sum(map(len,s["ancestry"]))==760900
assert np.all(s["exact_source_rows"]==-1)
for k in ("faces","points","source_rows","source_layers","source_bodies","source_bones",
          "source_original_ids","source_identity0","source_identity1"):
    np.testing.assert_array_equal(s[k],native[k])
for dest,src in (("ancestry","face_ancestry_source_face_rows"),
                 ("ancestry_original_ids","face_ancestry_source_original_ids"),
                 ("ancestry_parent_rows","face_ancestry_parent_face_rows")):
    np.testing.assert_array_equal(np.concatenate(s[dest]),a[src])
    assert native[dest]==s[dest]
sample=dict(s);sel=np.asarray([0,13944,13945,74298,76801,76936])
for k in ("source_rows","exact_source_rows","source_original_ids","source_layers","source_bodies",
          "source_bones","source_run_original_ids","source_face_ids","source_identity0","source_identity1"):
    sample[k]=s[k][sel]
sample["faces"]=s["faces"][sel]
for k in ("ancestry","ancestry_original_ids","ancestry_parent_rows"):sample[k]=[s[k][i] for i in sel]
records=base._make_face_records(sample,1,"outer_O")
merged=base._merge_face_records([records])
np.testing.assert_array_equal(merged["face_ancestry_source_rows"],np.concatenate(sample["ancestry"]))
np.testing.assert_array_equal(merged["face_ancestry_source_original_ids"],np.concatenate(sample["ancestry_original_ids"]))
np.testing.assert_array_equal(merged["face_ancestry_parent_face_rows"],np.concatenate(sample["ancestry_parent_rows"]))
tests=["real_O081_full_typed_derivation_CSR_preserved","native_runner_uses_same_prepared_loader",
       "sampled_native_face_record_export_preserves_all_three_CSR_lanes"]
with tempfile.TemporaryDirectory(prefix="reduced-lineage-native-") as td:
    path=Path(td)/"corrupt.npz"
    def reject(name,mutate):
        c={k:v.copy() for k,v in a.items()};mutate(c);np.savez_compressed(path,**c)
        try:h.load_surface_217(base,path,"O")
        except (ValueError,KeyError,IndexError):tests.append(name)
        else:raise AssertionError("accepted "+name)
    reject("reject_stale_construction_group_offsets",lambda c:c.update(source_group_offsets=np.r_[c["source_group_offsets"],len(c["source_group_face_rows"])]))
    reject("reject_stale_face_group",lambda c:c["face_group_id"].__setitem__(0,1))
    reject("reject_partial_typed_CSR",lambda c:c.pop("face_ancestry_source_original_ids"))
    reject("reject_exact_parent_fiction",lambda c:c["face_exact_source_row"].__setitem__(0,0))
    reject("reject_duplicate_source_group_inventory",lambda c:c["source_group_face_rows"].__setitem__(0,c["source_group_face_rows"][1]))
receipt={"status":"PASS","tests":tests,"test_count":len(tests),"physical_admission":False,"runtime":0,
         "candidate_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
         "handoff_sha256":hashlib.sha256((HERE/"prepare_tetgen_handoff_697_typedD.py").read_bytes()).hexdigest(),
         "native_runner_sha256":hashlib.sha256((HERE/"run_native_domain_217_typedD_697.py").read_bytes()).hexdigest()}
(HERE/"test_prepared_reduced_lineage.receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt))
