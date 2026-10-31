#!/usr/bin/env python3
"""Current-parent metadata binding on the pinned 790 tissue-quality constructor."""
from pathlib import Path
import importlib.util,sys,hashlib,json
import numpy as np
BASE=Path(__file__).with_name("coupled_anchor_collapse_790_inner_layer_root.py")
BASE_SHA="097e4c7256d23ddee251e782fa8cc81b56275aea6f516adc4ea298fbd4c4070a"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(BASE)==BASE_SHA,"790 owner changed"
sp=importlib.util.spec_from_file_location("root791base790",BASE);G=importlib.util.module_from_spec(sp);sys.modules[sp.name]=G;sp.loader.exec_module(G)
M=G.M;PREVIOUS_TYPED=M.typed_source_output;PREVIOUS_RUN=M.run_from_config
def typed_current_parent(parent,ops,points,faces,faceparents):
    data=PREVIOUS_TYPED(parent,ops,points,faces,faceparents)
    active=G.V786.ACTIVE
    if not isinstance(active,dict) or not isinstance(active.get("parent_surface"),dict):raise ValueError("unbound immediate typed I parent")
    ref=active["parent_surface"];p=Path(ref["path"]).resolve()
    if sha(p)!=ref["sha256"]:raise ValueError("current typed I parent hash changed")
    data["typed_parent_candidate_path"]=np.asarray(str(p))
    data["typed_parent_candidate_sha256"]=np.asarray(ref["sha256"])
    return data
M.typed_source_output=typed_current_parent
def run_current_parent(config,out,edges_override=None,prune_receipt=None):
    result=PREVIOUS_RUN(config,out,edges_override=edges_override,prune_receipt=prune_receipt)
    p=Path(out)/"state.json"
    if p.exists():
        state=json.loads(p.read_text());state["current_typed_parent_owner"]={"path":str(Path(__file__).resolve()),"sha256":sha(__file__),"base":{"path":str(BASE),"sha256":BASE_SHA},"policy":"only immediate typed-parent path/hash rebound to hash-authenticated actual run input; complete source CSR and geometry unchanged"}
        M._write_json(p,state);result=state
    return result
M.run_from_config=run_current_parent
if __name__=="__main__":G.main()
