from pathlib import Path
import importlib.util,json,hashlib,numpy as np
HERE=Path(__file__).parent
P=HERE.parents[1]
spec=importlib.util.spec_from_file_location("typed_precedence_697",HERE/"prepare_tetgen_handoff_697_typedD.py")
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
source=P/"canonical-D-boundary-binding-696/run-007-collar-expanded-safe-replay/source-D-canonical14-stars-conditioned.npz"
owner=h.load_owner()
surface=h.load_surface_217(owner,source,"D")
with np.load(source,allow_pickle=False) as z:
 offsets=np.asarray(z["union_face_ancestry_offsets"])
 assert len(surface["faces"])==257463
 assert "face_source_row" in z.files
 assert len(surface["ancestry"])==len(surface["faces"])
 for dest,key in (("ancestry","union_face_ancestry_rows"),
                  ("ancestry_original_ids","union_face_ancestry_run_original_ids"),
                  ("ancestry_parent_rows","union_face_ancestry_face_ids")):
  a=np.asarray(z[key])
  np.testing.assert_array_equal(np.concatenate(surface[dest]),a)
  assert len(a)==303421
  assert all(len(surface[dest][i])==offsets[i+1]-offsets[i] for i in range(len(surface["faces"])))
 assert np.count_nonzero(offsets[1:]-offsets[:-1]>1)>0
 assert np.all(surface["exact_source_rows"]==-1)
result={"status":"PASS","physical_admission":False,"runtime":0,
        "input_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "handoff_sha256":hashlib.sha256((HERE/"prepare_tetgen_handoff_697_typedD.py").read_bytes()).hexdigest(),
        "faces":len(surface["faces"]),"preserved_union_ancestry_tuples":sum(map(len,surface["ancestry"])),
        "regression":"scalar face_source_row must not shadow full typed union ancestry"}
(HERE/"test_conditioned_D_typed_precedence.receipt.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
