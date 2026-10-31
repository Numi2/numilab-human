#!/usr/bin/env python3
import hashlib,json,os
from pathlib import Path
R=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
P=Path("/Users/numi/Work/human-volume-proof-publication-20261031")
PK=P/"tools/evidence/connected-skin-fat-physical-transfer-20261031h"
ST=Path("/Users/numi/Work/.human-quality678-archive-stage")
START=R/"all-quality-678-root/round001/trial01/verified-state.json"
STOP="checkpoint-current622-root.json"
GM=P/"tools/evidence/connected-skin-fat-physical-transfer-20261031g/manifest.json"
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8*1024*1024),b""):h.update(b)
 return h.hexdigest()
ST.mkdir(parents=True,exist_ok=True);(ST/"blobs").mkdir(exist_ok=True)
files={};blobs={};states=[];external=[];p=START;seen=set();idx=0
def add(path,kind,expected=None,state_index=None):
 pth=Path(path)
 if not pth.is_file():raise FileNotFoundError(f"{kind}: {pth}")
 h=sha(pth)
 if expected and h!=expected:raise ValueError(f"SHA mismatch {pth}: {h} != {expected}")
 size=pth.stat().st_size; key=(str(pth),h); member="blobs/"+h
 if key not in files:files[key]={"source_path":str(pth),"source_relative_path":str(pth.relative_to(R)) if pth.is_relative_to(R) else None,"archive_member":member,"sha256":h,"size_bytes":size,"kinds":[kind],"state_index":state_index}
 else:files[key]["kinds"].append(kind)
 if h not in blobs:
  dst=ST/"blobs"/h
  if not dst.exists():os.link(pth,dst)
  if dst.stat().st_size!=size or sha(dst)!=h:raise ValueError("bad staged blob "+str(dst))
  blobs[h]={"size_bytes":size,"source_path":str(pth)}
 return h,size
while True:
 if p in seen:raise ValueError("parent cycle")
 seen.add(p);d=json.loads(p.read_text())
 if p.name==STOP:
  h=sha(p);g=json.loads(GM.read_text());hits=[x for x in g["files"] if x.get("payload")=="payload/checkpoint-current622-root.json"]
  if len(hits)!=1 or hits[0]["sha256"]!=h:raise ValueError("checkpoint622 does not match .g")
  external.append({"path":str(p),"sha256":h,"size_bytes":p.stat().st_size,"dependency":"tools/evidence/connected-skin-fat-physical-transfer-20261031g","dependency_manifest_sha256":sha(GM),"reason":"existing .g supplies checkpoint622; geometry is not duplicated"})
  break
 sh,sn=add(p,"state",state_index=idx);row={"index":idx,"path":str(p),"sha256":sh,"size_bytes":sn,"parent":d.get("source_state"),"operations":{}}
 for role,x in sorted(d.get("source_surfaces",{}).items()):
  h,n=add(x["path"],"typed_surface_"+role,x.get("sha256"),idx);row.setdefault("surfaces",{})[role]={"sha256":h,"size_bytes":n}
 for role in ("Skin","Fat"):
  x=d.get(role+"_current")
  if x:
   h,n=add(x["path"],"volume_"+role,x.get("sha256"),idx)
   row[role]={k:x.get(k) for k in ("cells","nodes","native_quality_failures","nonpositive","missing_target_faces","extra_actual_faces")}
   row[role].update({"sha256":h,"size_bytes":n})
 for key in ("producer","local_owner","exact_context","independent_replay"):
  x=d.get(key)
  if isinstance(x,dict) and x.get("path"):
   add(x["path"],key,x.get("sha256"),idx)
   if key=="exact_context":
    report=json.loads(Path(x["path"]).read_text());refs=[]
    def walk(o):
     if isinstance(o,dict):
      if isinstance(o.get("path"),str) and isinstance(o.get("sha256"),str):refs.append(o)
      for v in o.values():walk(v)
     elif isinstance(o,list):
      for v in o:walk(v)
    walk(report)
    for ref in refs:
     q=Path(ref["path"])
     if q.is_file():add(q,"context_dependency",ref["sha256"],idx)
 row["operations"]={k:(len(v) if isinstance(v,list) else v) for k,v in d.get("operations",{}).items()}
 row["quality_failures"]={k:d.get(k+"_current",{}).get("native_quality_failures") for k in ("Skin","Fat")}
 row["surface_crossings"]=d.get("surface_crossings");row["physical_admission"]=d.get("physical_admission");row["native_steps"]=d.get("native_steps")
 states.append(row)
 parent=d.get("source_state",{}).get("path")
 if not parent:raise ValueError("no source_state parent at "+str(p))
 p=Path(parent);idx+=1
F=R/"bounded625-local-cavity-repair/metadata-finalized-678-v2-001"
for name,kind in [("state.json","final_metadata_state"),("metadata-finalizer-report.json","metadata_finalizer_report"),("finalizer-readback.json","finalizer_readback"),("skin-paired-target-metadata-finalized.npz","finalized_Skin_volume"),("fat-paired-target-metadata-finalized.npz","finalized_Fat_volume")]:add(F/name,kind)
add(R/"bounded625-local-cavity-repair/metadata-finalized-677-001/skin-paired-target-metadata-finalized.npz","failed_partial_677_artifact")
entries=sorted(files.values(),key=lambda x:(x["archive_member"],x["source_path"]))
index={"schema":"numi-human.quality678-deduplicated-lineage-archive.v1","snapshot":{"verified_state":str(START),"verified_state_sha256":sha(START),"metadata_finalized_state":str(F/"state.json"),"metadata_finalized_state_sha256":sha(F/"state.json"),"not_live_checkpoint":True,"quality_failures":{"Skin":448,"Fat":2531},"surface_crossings":0,"physical_admission":False,"native_steps":0,"mass_status":"PENDING"},"parent_evidence_package":{"path":"tools/evidence/connected-skin-fat-physical-transfer-20261031g","manifest_sha256":sha(GM),"checkpoint622_sha256":external[0]["sha256"]},"states_in_order":states,"external_dependencies":external,"files":entries,"unique_blobs":[{"archive_member":"blobs/"+h,"sha256":h,"size_bytes":v["size_bytes"]} for h,v in sorted(blobs.items())],"unique_blob_count":len(blobs),"unique_blob_bytes":sum(x["size_bytes"] for x in blobs.values()),"file_reference_count":len(entries)}
IP=PK/"archive-index.json";IP.write_text(json.dumps(index,indent=2,sort_keys=True)+"\n")
(ST/"archive-index.json").write_bytes(IP.read_bytes())
print("states",len(states),"file refs",len(entries),"unique blobs",len(blobs),"bytes",index["unique_blob_bytes"])
print("index sha",sha(IP),"parent622",external[0]["sha256"],"parent .g manifest",sha(GM))
