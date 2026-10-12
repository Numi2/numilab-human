#!/usr/bin/env python3
"""Audit native right forearm captures and exact unchanged-scene parity."""
from pathlib import Path
import argparse,ast,csv,hashlib,importlib.util,json,sys,time
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
OLD=A/'forearm-shared-field-native-012-diagonal097'
V=OLD/'verify_forearm_shared_field_native_012.py'
H=A/'neck-back-retained-pose-audit-002/forward_all_positive_weights.py'
PUB=Path('/Users/n/numi-human-left-pq-publication-1344/src')
TARGETS=(131,133)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v:raise RuntimeError(m)
def save(p,j):p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=Path,required=True);ap.add_argument('--bundle',type=Path,required=True);args=ap.parse_args()
    scene=args.scene.resolve();bundle=args.bundle.resolve();run=scene/'baseline/native-run'
    out=scene/'native-audit-001';out.mkdir(exist_ok=False)
    sys.path.insert(0,str(PUB))
    from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
    from numilab_human.cardiac_cavity_geometry import analyze_topology
    spec=importlib.util.spec_from_file_location('native13forward',H);h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h)
    source=V.read_text()
    tree=ast.parse(source);names={'parity','volume_signs','sign','build_union','exact_audit','summarize_topology'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    text=ast.unparse(ast.Module(body=nodes,type_ignores=[])).replace('gids[132]','gids[131]').replace('gids[134]','gids[133]')
    ns={'np':np,'TARGETS':TARGETS};exec(compile(text,str(V),'exec'),ns)
    pins={str(p):sha(p) for p in (Path(__file__),V,H,Path(ci.__file__),Path(ca.__file__),scene/'preparation.json',scene/'baseline/run-declaration.json',scene/'baseline/execution.json',run/'invocation.json',run/'run-metadata.json',bundle/'preparation.json')}
    def pin(p):pins[str(p)]=sha(p)
    bp=json.loads((bundle/'preparation.json').read_text())
    decl=json.loads((scene/'baseline/run-declaration.json').read_text())
    ex=json.loads((scene/'baseline/execution.json').read_text())
    meta=json.loads((run/'run-metadata.json').read_text());inv=json.loads((run/'invocation.json').read_text())
    report={'status':'running','complete':False,'scope':'Actual native GPU right131/133 child/union checks; all866 other surfaces including left097,157bodyposes and fullphysiology compared exactly to012. No external-interface or final310s admission.','checks':{},'captures':{},'pins':pins}
    def check(k,v,**kw):report['checks'][k]={'passed':bool(v),**kw};save(out/'report.json',report)
    check('execution',ex.get('returncode')==0 and ex.get('timed_out') is False and meta.get('exit_code')==0,wall_seconds=ex.get('wall_seconds'))
    check('invocation',inv.get('argv')==decl['owner_cli_preview']['native_argv'] and inv.get('environment')==decl['owner_cli_preview']['recorded_invocation_environment'] and not ex.get('changed_inputs') and not meta.get('source_files_changed_during_run'))
    badpins=[p for p,d in decl['immutable_assets'].items() if sha(p)!=d]
    check('immutable_inputs',not badpins,count=len(decl['immutable_assets']),mismatches=badpins)
    src={};gids={};rest={};weights={}
    child=h.load_tissue(scene/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
    oldchild=h.load_tissue(OLD/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue')
    for sid in TARGETS:
        item=bp['geometry_rows'][str(sid)];p=Path(item['path']);require(sha(p)==item['sha256'],'bundlegeometrychanged');pin(p)
        with np.load(p,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
        gids[sid]=list(map(int,d['global_vertex_ids']));rest[sid]=d['vertices6'][:,:3];weights[sid]=d['core_weights_float64']
        src[sid]={'faces':d['faces_local']}
        row=h.row_data(child,sid)
        check('source_geometry_'+str(sid),np.array_equal(row['positions'],rest[sid]) and np.array_equal(row['faces'],src[sid]['faces']))
    bad=[]
    for r in oldchild['records']:
        sid=int(r[6])
        if sid in TARGETS:continue
        a=h.row_data(child,sid);b=h.row_data(oldchild,sid)
        if a['vertex_bytes']!=b['vertex_bytes'] or not np.array_equal(a['faces'],b['faces']) or a['binding_bytes']!=b['binding_bytes']:bad.append(sid)
    check('untargeted_source_rows',not bad,expected=156,mismatches=bad)
    shared=sorted(set(gids[131])&set(gids[133]));ix={sid:{g:i for i,g in enumerate(gids[sid])} for sid in TARGETS}
    check('shared_reference_field',all(rest[131][ix[131][g]].tobytes()==rest[133][ix[133][g]].tobytes() and weights[131][ix[131][g]].tobytes()==weights[133][ix[133][g]].tobytes() for g in shared),shared_vertices=len(shared))
    uxyz,ufaces,usources,umeta=ns['build_union'](src,gids,rest)
    require(not umeta['interface_errors'],'source interface error')
    reference={}
    for label,xyz,faces in [(str(s),rest[s],src[s]['faces']) for s in TARGETS]+[('union',uxyz,ufaces)]:
        top=analyze_topology(xyz.astype(float).tolist(),faces.tolist())
        reference[label]={'components':len(top['face_components']),'signs':ns['sign'](ns['volume_signs'](xyz,faces,top['face_components']))}
        require(top['closed_oriented_manifold_candidate'] and 'zero' not in reference[label]['signs'],'invalid reference topology')
    def csvrows(p):
        with p.open(newline='') as f:return list(csv.DictReader(f))
    tracep=run/'resting-coupled.csv';oldtracep=OLD/'baseline/native-run/resting-coupled.csv';pin(tracep);pin(oldtracep)
    trace=csvrows(tracep);oldtrace=csvrows(oldtracep)
    check('physiology_exact_012',trace==oldtrace and len(trace)==1000,rows=len(trace),last_step=int(float(trace[-1]['step'])),last_time_s=float(trace[-1]['time_s']))
    req,*_=ca._load_target_inventory(h.INV);req=set(req)|{(51005,s) for s in (151,152,159,160,161,162,163,164)}
    inventory=req|{(51007,1)};targets={(51005,s) for s in TARGETS}
    def audit(label,xyz,faces):
        topo=analyze_topology(xyz.astype(float).tolist(),faces.tolist())
        count,dg,pairs=ns['exact_audit'](h,xyz,faces,ci)
        signs=ns['sign'](ns['volume_signs'](xyz,faces,topo['face_components']))
        ok=count==0 and not dg and topo['closed_oriented_manifold_candidate'] and len(topo['face_components'])==reference[label]['components'] and signs==reference[label]['signs']
        return {'passed':bool(ok),'exact_self_pair_count':count,'exact_self_pairs':[list(map(int,p)) for p in pairs],'degenerate_face_ids':list(map(int,dg)),'topology':ns['summarize_topology'](topo),'component_signs':signs}
    for step in [0,8000,3807,4767,5023,5599,6207,7423]:
        start=time.monotonic();p=run/'accepted-geometry'/f'step-{step}.mrvpack';rp=p.with_suffix('.receipt.json')
        op=OLD/'baseline/native-run/accepted-geometry'/p.name;orp=op.with_suffix('.receipt.json')
        for path in (p,rp,op,orp):pin(path)
        rc=json.loads(rp.read_text());orc=json.loads(orp.read_text())
        pos,surfs,_=ca._pack_surfaces(p,req);opos,osurfs,_=ca._pack_surfaces(op,req)
        require(set(surfs)==inventory and set(osurfs)==inventory and len(surfs)==868,'surface inventory differs')
        otherbad=[]
        for k in inventory-targets:
            f=np.asarray(surfs[k]['faces'],dtype=np.int64);of=np.asarray(osurfs[k]['faces'],dtype=np.int64)
            if f.shape!=of.shape or not np.array_equal(pos[f],opos[of]):otherbad.append(list(k))
        bodyok=len(rc.get('accepted_body_poses',[]))==157 and rc.get('accepted_body_poses')==orc.get('accepted_body_poses')
        valid=rc.get('physical_endpoint')=='accepted' and rc.get('pack_file_sha256')==sha(p)
        world={};results={}
        for sid in TARGETS:
            f=np.asarray(surfs[(51005,sid)]['faces'],dtype=np.int64);local=src[sid]['faces']
            require(f.shape==local.shape,'native face layout changed')
            delta=f-local;require(np.all(delta==delta.flat[0]),'native face offsets inconsistent')
            first=int(delta.flat[0]);world[sid]=pos[first:first+len(gids[sid])].astype('<f4').copy()
            results[str(sid)]=audit(str(sid),world[sid],local)
        ux,uf,us,um=ns['build_union'](src,gids,world)
        results['union']=audit('union',ux,uf)
        results['union']['source_face_witnesses']=[[[int(us[int(i)][0]),int(us[int(i)][1])] for i in pair] for pair in results['union']['exact_self_pairs']]
        results['union']['interface_errors']=um['interface_errors']
        results['union']['shared_face_pairs']=um['shared_face_pairs']
        shared_bad=[g for g in shared if world[131][ix[131][g]].tobytes()!=world[133][ix[133][g]].tobytes()]
        seam=not shared_bad and not um['interface_errors'] and um['shared_face_pairs']==umeta['shared_face_pairs']
        results['union']['all_shared_vertices_bitwise_equal']=not shared_bad
        results['union']['shared_vertex_count']=len(shared)
        results['union']['shared_vertex_coordinate_mismatches']=shared_bad
        ok=valid and bodyok and not otherbad and seam and all(v['passed'] for v in results.values())
        result={'passed':bool(ok),'valid_accepted_receipt':valid,'accepted_body_poses_exact_012':bodyok,'unchanged_surface_count':866-len(otherbad),'unchanged_surface_mismatches':otherbad,'meshes':results,'wall_seconds':time.monotonic()-start}
        report['captures'][str(step)]=result;save(out/f'step-{step}.json',result);save(out/'report.json',report)
        print(json.dumps({'step':step,'passed':bool(ok),'pairs':{k:v['exact_self_pair_count'] for k,v in results.items()},'unchanged_surfaces':866-len(otherbad),'seconds':result['wall_seconds']}),flush=True)
    report['complete']=True
    report['status']='passed_bounded_native_not_admitted' if all(c['passed'] for c in report['checks'].values()) and all(c['passed'] for c in report['captures'].values()) else 'failed_bounded_native_not_admitted'
    require(all(sha(p)==d for p,d in pins.items()),'audit input changed')
    save(out/'report.json',report);print(json.dumps({'status':report['status'],'report':str(out/'report.json')}),flush=True)
if __name__=='__main__':main()

