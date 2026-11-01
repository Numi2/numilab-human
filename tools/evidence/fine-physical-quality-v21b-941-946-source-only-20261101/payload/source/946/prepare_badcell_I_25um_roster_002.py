#!/usr/bin/env python3
"""Focused read-only I edge roster from only the current bad-cell stars."""
from pathlib import Path
import hashlib, importlib.util, json, sys, time
import numpy as np

ROOT=Path("/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706")
HERE=ROOT/"quality-endpoint-collapse-001"
STATE=ROOT/"all-quality-943-expanded-joint-root/qualified/verified-state.json"
STATE_SHA="093cbef3d1d379852cecac183a15b04d286974b5e8b82f2420479d10c39f1c7e"
BAD_REPORT=ROOT/"all-quality-945-fixed-cavity-connected-root/batch-report.json"
BAD_REPORT_SHA="70269e8c148de024f5e413f3689287e5a50be47ea7d10b72cd7df3085c75c283"
Q_PATH=HERE/"batch_inner_endpoint_quotient_010_prepared_guard.py"
Q_SHA="8da4676413c2801c896d386454978c0517d16c0b1e6c25bd0f7591ef8367d343"
OUT=ROOT/"all-quality-946b-I-short-badcell-roster-root"
CAP=25e-6

def sha(p): return hashlib.file_digest(Path(p).open("rb"),"sha256").hexdigest()
def imp(name,path,expected):
    got=sha(path)
    if got!=expected: raise RuntimeError(f"{name} SHA mismatch: {got}")
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m)
    return m
def fkey(x): return np.ascontiguousarray(np.asarray(x,np.float32).reshape(3)).view(np.uint32).tobytes()
def main():
    tic=time.monotonic()
    if sha(STATE)!=STATE_SHA: raise RuntimeError("current state pin mismatch")
    if sha(BAD_REPORT)!=BAD_REPORT_SHA: raise RuntimeError("current bad-cell report pin mismatch")
    q=imp("endpoint_batch_010_focused_roster",Q_PATH,Q_SHA)
    c6=json.loads(BAD_REPORT.read_text())
    if c6.get("input_state",{}).get("sha256")!=STATE_SHA: raise RuntimeError("C6 census is not from exact current state")
    bad_cells={r:sorted(int(x) for comp in c6["roles"][r]["components"] for x in comp["bad_cells"]) for r in ("Skin","Fat")}
    state,surfaces,srefs,volumes,vrefs=q.load_state_inputs(STATE)
    OUT.mkdir(parents=True,exist_ok=False)
    fixed,back,protection,P=q.protected_rows(STATE,state,"I",surfaces,volumes,OUT)
    source=surfaces["I"];faces=np.asarray(source["faces"],np.int64);sp=np.asarray(source["points_world_m_f32"],np.float32)
    active=np.unique(faces.reshape(-1)).astype(np.int64)
    active_set=set(map(int,active)); fixed=set(map(int,fixed)); back=set(map(int,back))
    targets=("Skin","Fat")
    reverse={};map_detail={}
    candidate_hits={};cell_witnesses={"Skin":{},"Fat":{}}
    for vr in targets:
        prefix="skin" if vr=="Skin" else "fat"
        v=volumes[vr];vp=np.asarray(v["points_world_m_f32"] if "points_world_m_f32" in v else v["points_world_m"],np.float32)
        pm=np.asarray(v[prefix+"_current_target_rebind_I_point_to_volume_rows"],np.int64)
        if pm.shape!=(len(sp),) or np.any(pm[active]<0) or np.any(pm[active]>=len(vp)):
            raise RuntimeError(f"{vr} current I point-to-volume mapping incomplete")
        if len(np.unique(pm[active]))!=len(active): raise RuntimeError(f"{vr} active I point mapping not injective")
        if not np.array_equal(sp[active],vp[pm[active]]): raise RuntimeError(f"{vr} active I F32 coordinate map mismatch")
        rev=np.full(len(vp),-1,np.int64);rev[pm[active]]=active;reverse[vr]=rev
        map_detail[vr]={"active_source_points":len(active),"active_unmapped_rows":0,"active_coordinate_collisions":0,
                        "volume_coordinate_collisions_among_active_source_points":0,"mapping_lane":prefix+"_current_target_rebind_I_point_to_volume_rows"}
        t=np.asarray(v["tetrahedra"],np.int64)
        for cell in bad_cells[vr]:
            sr=rev[t[cell]];mapped=sorted(set(map(int,sr[sr>=0])))
            cell_witnesses[vr][cell]=mapped
            for i,a in enumerate(mapped):
                for b in mapped[i+1:]:
                    e=(a,b)
                    L=float(np.linalg.norm(sp[a].astype(np.float64)-sp[b].astype(np.float64)))
                    if L<=CAP:
                        row=candidate_hits.setdefault(e,{"hits":{},"length_m":L})
                        row["hits"].setdefault(vr,set()).add(int(cell))
    # Enumerate only source faces incident to vertices in the bad-cell candidate pairs.
    edge_candidates=set(candidate_hits)
    candidate_nodes=np.unique(np.asarray(sorted({x for e in edge_candidates for x in e}),np.int64)) if edge_candidates else np.empty(0,np.int64)
    incident_mask=np.any(np.isin(faces,candidate_nodes),axis=1) if len(candidate_nodes) else np.zeros(len(faces),bool)
    incident_rows=np.flatnonzero(incident_mask).astype(np.int64); local_faces=faces[incident_rows]
    efaces={e:[] for e in edge_candidates};neighbors={int(v):set() for v in candidate_nodes}
    for fi,tri in zip(incident_rows,local_faces):
        a,b,c=map(int,tri)
        if a in neighbors: neighbors[a].update((b,c))
        if b in neighbors: neighbors[b].update((a,c))
        if c in neighbors: neighbors[c].update((a,b))
        for x,y,z in ((a,b,c),(b,c,a),(c,a,b)):
            e=tuple(sorted((x,y)))
            if e in efaces: efaces[e].append((int(fi),int(z)))
    candidates=[];excluded=[];star_data=[]
    for e in sorted(edge_candidates,key=lambda e:(candidate_hits[e]["length_m"],e)):
        hit=candidate_hits[e];fr=sorted(x[0] for x in efaces[e]);opps=[x[1] for x in efaces[e]]
        interior=len(fr)==2
        common=(neighbors[e[0]]&neighbors[e[1]])-{e[0],e[1]}
        link=interior and common==set(opps)
        dirs=[{"drop":a,"keep":b,"eligible_endpoint":a not in fixed} for a,b in (e,e[::-1])]
        eligible=[d for d in dirs if d["eligible_endpoint"]]
        maps={vr:bool(np.all(reverse[vr][np.asarray(e,np.int64)]>=0)) for vr in targets}
        bad_by_role={r:sorted(map(int,hit["hits"].get(r,set()))) for r in targets if hit["hits"].get(r)}
        reasons=[]
        if not interior: reasons.append("not_an_interior_I_source_edge")
        if not link: reasons.append("source_link_condition_fail")
        if not eligible: reasons.append("both_endpoints_immutable_rim_or_route")
        if not all(maps.values()): reasons.append("endpoint_missing_from_paired_volume")
        rec={"role":"I","edge_rows":[int(e[0]),int(e[1])],"length_m":hit["length_m"],"source_face_rows":fr,
             "source_edge_incidence":len(fr),"link_condition_pass":bool(link),"bad_cell_hits":sum(len(x) for x in hit["hits"].values()),
             "bad_cells_by_volume_role":bad_by_role,"directions":dirs,"eligible_directions":eligible,
             "target_volume_endpoint_map":maps,"excluded_reasons":reasons}
        (candidates if interior and link and eligible and all(maps.values()) else excluded).append(rec)
        # Keep complete endpoint source/volume stars and parent cell rows for disjoint batching review.
        source_star=np.flatnonzero(np.any(np.isin(faces,e),axis=1)).astype(np.int64)
        src_vertices=np.unique(faces[source_star]).astype(np.int64)
        star_data.append({"edge":e,"source_face_rows":source_star,"source_vertex_rows":src_vertices})
        for vr in targets:
            vv=np.asarray([int(volumes[vr][("skin" if vr=="Skin" else "fat")+"_current_target_rebind_I_point_to_volume_rows"][x]) for x in e],np.int64)
            tet=np.asarray(volumes[vr]["tetrahedra"],np.int64)
            cells=np.flatnonzero(np.any(np.isin(tet,vv),axis=1)).astype(np.int64)
            nodes=np.unique(tet[cells]).astype(np.int64)
            rec.setdefault("volume_stars",{})[vr]={"endpoint_volume_rows":vv.tolist(),"cell_count":len(cells),"point_count":len(nodes)}
            star_data[-1][vr+"_cell_rows"]=cells;star_data[-1][vr+"_point_rows"]=nodes
    candidates.sort(key=lambda x:(x["length_m"],-x["bad_cell_hits"],x["edge_rows"]))
    # Write small exact star sidecar, keeping full point/cell identities out of JSON.
    payload={"edge_rows":np.asarray([x["edge"] for x in star_data],np.int64).reshape((-1,2)),
             "source_face_offsets":np.r_[0,np.cumsum([len(x["source_face_rows"]) for x in star_data])].astype(np.int64),
             "source_face_rows":np.concatenate([x["source_face_rows"] for x in star_data]) if star_data else np.empty(0,np.int64),
             "source_vertex_offsets":np.r_[0,np.cumsum([len(x["source_vertex_rows"]) for x in star_data])].astype(np.int64),
             "source_vertex_rows":np.concatenate([x["source_vertex_rows"] for x in star_data]) if star_data else np.empty(0,np.int64)}
    for vr in targets:
        payload[vr.lower()+"_cell_offsets"]=np.r_[0,np.cumsum([len(x[vr+"_cell_rows"]) for x in star_data])].astype(np.int64)
        payload[vr.lower()+"_cell_rows"]=np.concatenate([x[vr+"_cell_rows"] for x in star_data]) if star_data else np.empty(0,np.int64)
        payload[vr.lower()+"_point_offsets"]=np.r_[0,np.cumsum([len(x[vr+"_point_rows"]) for x in star_data])].astype(np.int64)
        payload[vr.lower()+"_point_rows"]=np.concatenate([x[vr+"_point_rows"] for x in star_data]) if star_data else np.empty(0,np.int64)
    star_path=OUT/"candidate-full-stars.npz";np.savez_compressed(star_path,**payload)
    # Pairwise overlap counts show whether the eligible edges can be retained together.
    overlap=[]
    for i in range(len(star_data)):
        for j in range(i+1,len(star_data)):
            a,b=star_data[i],star_data[j]
            overlap.append({"edge_rows":[list(a["edge"]),list(b["edge"])],
                "shared_source_faces":len(set(map(int,a["source_face_rows"]))&set(map(int,b["source_face_rows"]))),
                "shared_source_vertices":len(set(map(int,a["source_vertex_rows"]))&set(map(int,b["source_vertex_rows"]))),
                "shared_volume_cells":{vr:len(set(map(int,a[vr+"_cell_rows"]))&set(map(int,b[vr+"_cell_rows"]))) for vr in targets},
                "shared_volume_points":{vr:len(set(map(int,a[vr+"_point_rows"]))&set(map(int,b[vr+"_point_rows"]))) for vr in targets}})
    report={"schema":"h706.bad-cell-focused-I-short-edge-roster.v1","status":"PASS_READ_ONLY_ACTIVE_ROW_BAD_CELL_CENSUS",
       "physical_admission":False,"native_steps":0,"mass_mapping":"PENDING",
       "input_state":{"path":str(STATE),"sha256":STATE_SHA},"c6_bad_cell_report":{"path":str(BAD_REPORT),"sha256":BAD_REPORT_SHA},
       "role":"I","maximum_edge_m":CAP,"source_surface":srefs["I"],"target_volumes":{
          r:{**vrefs[r],"native_bad_cells":len(bad_cells[r]),"bad_cell_rows":bad_cells[r]} for r in targets},
       "active_coordinate_mapping":map_detail,
       "protected_scope":{"owner_path":str(Q_PATH),"owner_sha256":Q_SHA,"immutable_rim_route_source_rows":len(fixed),
          "actual_back_source_rows":len(back),"back_endpoints_distance_capped_by_existing_50um_guard":True},
       "candidate_edge_count_from_bad_cells_only":len(edge_candidates),"eligible_bad_cell_hit_edges":candidates,
       "excluded_candidate_edges":excluded,"eligible_count":len(candidates),"excluded_count":len(excluded),
       "candidate_star_sidecar":{"path":str(star_path),"sha256":sha(star_path),"domain":"current 943 input source and volume row domains"},
       "pairwise_star_overlap":overlap,"source_edge_scan":"candidate pairs came only from exact current bad-cell corners; incidence/link checked in incident source-face subset",
       "next_driver":{"path":str(Q_PATH),"sha256":Q_SHA,"command_template":"PY RUNNER --state STATE --roster ROSTER --role I --max-edge-m 2.5e-5 --output FRESH_OUT --max-ops 64 --max-trials 300 --seconds 90"},
       "elapsed_s":time.monotonic()-tic}
    out=OUT/"roster.json";out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":report["status"],"candidate_edge_count":len(edge_candidates),"eligible_count":len(candidates),
      "excluded_count":len(excluded),"edges":[{"edge":x["edge_rows"],"length_um":x["length_m"]*1e6,"hits":x["bad_cells_by_volume_role"]} for x in candidates],
      "elapsed_s":report["elapsed_s"],"report":str(out)},separators=(",",":")),flush=True)
if __name__=="__main__":main()
