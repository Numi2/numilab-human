#!/usr/bin/env python3
from pathlib import Path
import ast,hashlib
ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/coupled-anchor-collapse-780-root")
F=ROOT/"ordinary-allrole-v25f-blocklocal/repair_coupled_quality_points_722_v25f_ordinary_blocklocal.py"
E=ROOT/"ordinary-allrole-v25e/repair_coupled_quality_points_722_v25e_inner_back_remainder_blocklocal.py"
G=ROOT/"ordinary-allrole-v25g-blocklocal/repair_coupled_quality_points_722_v25g_ordinary_blocklocal.py"
EXPECTED={"f":"e05d5662d64edd97d25c5a0e42e00a58ba06ef2850b3afc6fd9b23a3111fc376",
          "e":"59b78e7f4766de9ed000d4a35bd048938c7b49cdc04e183ed560a6015767fb7c",
          "g":"0571c28aa8966516a204017bbeaaf1b01eb8fc460d296974d57a14464bdac189"}
def pin(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert pin(F)==EXPECTED["f"] and pin(E)==EXPECTED["e"] and pin(G)==EXPECTED["g"]
def func(path):
 s=path.read_text();lines=s.splitlines(keepends=True);tree=ast.parse(s)
 n=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="_single_node_quality_candidate")
 return s,lines,n
fs,fl,fn=func(F);es,el,en=func(E);gs,gl,gn=func(G)
assert ast.dump(fn.args,include_attributes=False)==ast.dump(en.args,include_attributes=False)==ast.dump(gn.args,include_attributes=False)
assert ast.dump(gn,include_attributes=False)==ast.dump(en,include_attributes=False)
restored="".join(gl[:gn.lineno-1]+fl[fn.lineno-1:fn.end_lineno]+gl[gn.end_lineno:])
assert restored==fs
print("PASS v25g is v25f byte-for-byte outside the single-node fallback; API/body exactly match v25e")
