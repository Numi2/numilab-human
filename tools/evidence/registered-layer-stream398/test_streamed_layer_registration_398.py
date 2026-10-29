#!/usr/bin/env python3
"""Check export byte parity, bounded result lifetime and failed-staging cleanup."""
from pathlib import Path
from contextlib import ExitStack, redirect_stdout
from unittest.mock import patch
import argparse, importlib.util, io, json, sys, tempfile, weakref
ROOT=Path(__file__).resolve().parent
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sys.path.insert(0,str(ROOT))
candidate_path=ROOT/"export_authoring_registered_surface_sidecars_230.py"
if not candidate_path.is_file():candidate_path=ROOT.parents[1]/candidate_path.name
sys.path.insert(0,str(candidate_path.parent))
candidate=load("streamed",candidate_path)
baseline=load("baseline",ROOT/"baseline_export_authoring_registered_surface_sidecars_230.py")
class Tracked(dict):pass
def run(m,root,out,fail=False):
 live=weakref.WeakValueDictionary();counts=[]
 args=argparse.Namespace(outer=root/"O",inner=root/"I",deep=root/"D",parent=root/"parent",poses=root/"poses",parent_full86_weights=root/"weights",out=out,maximum_parent_distance_m=.02,tie_tolerance_m=1e-8)
 def surface(path,layer,require_f32=True):
  return dict(layer=layer,path=str(path),sha256="1"*64,vertexCount=1,faceCount=1,sourceOriginalFaceIDs=[1],faceAncestry={},sourceSchema={})
 def register(s,*a):
  l=s["layer"];counts.append(len(live))
  if fail and l=="I":raise ValueError("controlled registration rejection")
  def rows(header):return [{k:("row-"+l+"-"+k) for k in header}]
  result=Tracked(nodes=rows(m.NODE_HEADER),faces=rows(m.FACE_HEADER),weightRows=rows(m.WEIGHT_HEADER),weightOwnerRows=rows(m.WEIGHT_OWNER_HEADER),registrations=rows(m.PARENT_HEADER),metrics={"count":1},sourceDomain={"layer":l})
  live[l]=result
  return result
 with ExitStack() as stack:
  for name,value in {
   "validate_candidate":surface,"load_frames":lambda p:({},{"sha256":"2"*64},list(range(86))),
   "load_parent_current":lambda p:({},{},{},{"nodeSHA256":"3"*64}),
   "load_parent_full86_weights":lambda *a:(None,{}),"register_layer":register,"frame_csv_rows":lambda *a:[]}.items():
   stack.enter_context(patch.object(m,name,value))
  with redirect_stdout(io.StringIO()):
   if fail:
    try:m.export(args)
    except ValueError as e:assert str(e)=="controlled registration rejection"
    else:raise AssertionError("failed registration was accepted")
    assert not out.exists() and not out.with_name(out.name+".staging").exists()
   else:m.export(args)
 return counts
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 for name in ["O","I","D","parent","poses","weights"]:(root/name).touch()
 before=root/"baseline";after=root/"streamed"
 oldcounts=run(baseline,root,before);newcounts=run(candidate,root,after)
 names=sorted(p.name for p in before.iterdir())
 assert names==sorted(p.name for p in after.iterdir())
 for name in names:assert (before/name).read_bytes()==(after/name).read_bytes(),name
 assert oldcounts==[0,1,2] and newcounts==[0,0,0],(oldcounts,newcounts)
 run(candidate,root,root/"failed",fail=True)
 print(json.dumps({"status":"PASS","identical_output_files":len(names),"baseline_retained_previous_results":oldcounts,"streamed_retained_previous_results":newcounts,"controlled_failure_cleanup":True,"geometry_and_registration_math_changed":False}))
