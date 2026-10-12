#!/usr/bin/env python3
"""Audit appended back rows from accepted native geometry; preserve failures."""
from pathlib import Path
import argparse,ast,hashlib,importlib.util,json,sys,time
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
OLD=A/'forearm-shared-field-native-012-diagonal097'
V=OLD/'verify_forearm_shared_field_native_012.py'
H=A/'neck-back-retained-pose-audit-002/forward_all_positive_weights.py'
OWNER=Path('/Users/n/numi-human-left-pq-publication-1344/src')
IDS=tuple(range(153,159))
EXPECTED={(153,155):816,(154,156):815,(155,157):457,(156,158):450,(157,158):254}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,m):
    if not x:raise RuntimeError(m)
def save(p,j):p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scene',type=Path,required=True);a=ap.parse_args()
    scene=a.scene.resolve();run=scene/'baseline/native-run';out=scene/'native-back-audit-001';out.mkdir(exist_ok=False)
    sys.path.insert(0,str(OWNER))
    from numilab_human import cardiac_cavity_intersections as ci,common_atlas_skin_clearance as ca
    from numilab_human.cardiac_cavity_geometry import analyze_topology
    sp=importlib.util.spec_from_file_location('backnativeforward',H);h=importlib.util.module_from_spec(sp);sys.modules[sp.name]=h;sp.loader.exec_module(h)
    nodes=[n for n in ast.parse(V.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in {'exact_audit','volume_signs','sign','summarize_topology'}]
    ns={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(V),'exec'),ns)
    pins={}
    def pin(p):pins[str(p)]=sha(p)
    for p in (Path(__file__),V,H,Path(ci.__file__),Path(ca.__file__),scene/'preparation.json',scene/'baseline/run-declaration.json',scene/'baseline/execution.json',run/'run-metadata.json',run/'invocation.json',scene/'passive-tissue-parent-bindings.json'):pin(p)
    decl=json.load(open(scene/'baseline/run-declaration.json'));ex=json.load(open(scene/'baseline/execution.json'));meta=json.load(open(run/'run-metadata.json'));inv=json.load(open(run/'invocation.json'))
    require(ex['returncode']==0 and not ex['timed_out'] and meta['exit_code']==0,'native execution failure')
    require(not ex['changed_inputs'] and inv['argv']==decl['owner_cli_preview']['native_argv'] and inv['environment']==decl['owner_cli_preview']['recorded_invocation_environment'],'invocation changed')
    require(all(sha(p)==s for p,s in decl['immutable_assets'].items()),'immutable asset changed')
    cp=scene/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue';op=OLD/'assets'/cp.name;pin(cp);pin(op)
    child=h.load_tissue(cp);old=h.load_tissue(op);require(len(child['records'])==164 and len(old['records'])==158,'child source inventory')
    unchanged=[]
    for r in old['records']:
        sid=int(r[6]);x=h.row_data(child,sid);y=h.row_data(old,sid)
        require(x['vertex_bytes']==y['vertex_bytes'] and x['binding_bytes']==y['binding_bytes'] and np.array_equal(x['faces'],y['faces']),'prior source row changed');unchanged.append(sid)
    def referenced_topology(xyz,faces):
        used,inverse=np.unique(np.asarray(faces,dtype=np.int64).reshape(-1),return_inverse=True)
        top=analyze_topology(np.asarray(xyz)[used].astype(float).tolist(),inverse.reshape(-1,3).tolist())
        top['unreferenced_source_vertex_ids']=sorted(set(range(len(xyz)))-set(map(int,used)))
        return top
    rows={s:h.row_data(child,s) for s in IDS};reference={}
    for s,r in rows.items():
        top=referenced_topology(r['positions'],r['faces'])
        reference[s]={'unreferenced_source_vertex_ids':top['unreferenced_source_vertex_ids'],'components':len(top['face_components']),'signs':ns['sign'](ns['volume_signs'](r['positions'],r['faces'],top['face_components']))}
        require(top['closed_oriented_manifold_candidate'] and 'zero' not in reference[s]['signs'],'invalid source topology')
    ext=json.load(open(scene/'passive-tissue-parent-bindings.json'))
    surfaces={x['stable_id']:x for x in ext['surfaces']}
    keys={}
    for s,r in rows.items():
        vertices=surfaces[s]['vertices'];require(len(vertices)==len(r['positions']),'term cardinality')
        require(all(v and all(t[3]>0 for t in v) and abs(sum(t[3] for t in v)-1)<1e-12 for v in vertices),'invalid parent partition')
        keys[s]=[tuple(tuple(t) for t in v) for v in vertices]
    pairs={}
    def parity(values):return sum(values[i]>values[j] for i in range(3) for j in range(i+1,3))%2
    for (aa,bb),expected in EXPECTED.items():
        face_maps={}
        for s in (aa,bb):
            fm={}
            for fi,f in enumerate(rows[s]['faces']):
                ordered=tuple(keys[s][int(v)] for v in f);key=tuple(sorted(ordered))
                require(key not in fm,'duplicate source interface face key');fm[key]=(fi,ordered)
            face_maps[s]=fm
        shared=set(face_maps[aa])&set(face_maps[bb]);require(len(shared)==expected,'shared face cardinality changed')
        records=[]
        for key in sorted(shared):
            fa,ka=face_maps[aa][key];fb,kb=face_maps[bb][key];require(parity(ka)!=parity(kb),'shared winding mismatch')
            va={keys[aa][int(v)]:int(v) for v in rows[aa]['faces'][fa]};vb={keys[bb][int(v)]:int(v) for v in rows[bb]['faces'][fb]}
            records.extend((va[k],vb[k]) for k in key)
        pairs[(aa,bb)]=sorted(set(records))
    tp=run/'resting-coupled.csv';otp=OLD/'baseline/native-run/resting-coupled.csv';pin(tp);pin(otp)
    require(tp.read_bytes().splitlines()[:len(otp.read_bytes().splitlines())]==otp.read_bytes().splitlines(),'physiology first 16s prefix changed')
    req,*_=ca._load_target_inventory(h.INV);oldreq=set(req)|{(51005,s) for s in (151,152,159,160,161,162,163,164)}
    newreq=oldreq|{(51005,s) for s in IDS}
    report={'scope':'Native whole-body back sliver repair diagnostic; raw contacts remain unadjudicated. No anatomical admission.','complete':False,'native_admission':False,'wall_seconds':ex['wall_seconds'],'immutable_pin_count':len(decl['immutable_assets']),'physiology_first_16s_prefix_byte_exact_012':True,'original_158_rows_byte_exact':True,'source_referenced_topology':reference,'captures':{},'pins':pins}
    for step in decl['capture_steps']:
        start=time.monotonic();p=run/'accepted-geometry'/f'step-{step}.mrvpack';rp=p.with_suffix('.receipt.json');oldp=OLD/'baseline/native-run/accepted-geometry'/p.name;oldrp=oldp.with_suffix('.receipt.json')
        for x in (p,rp):pin(x)
        rc=json.load(open(rp));require(rc['physical_endpoint']=='accepted' and rc['pack_file_sha256']==sha(p),'not accepted geometry')
        require(len(rc['accepted_body_poses'])==157,'body pose inventory changed')
        pos,surfs,_=ca._pack_surfaces(p,newreq)
        require(len(surfs)==874,'native inventory differs')
        baseline_exact=None
        if step==0:
            pin(oldp);pin(oldrp)
            orc=json.load(open(oldrp));opos,osurf,_=ca._pack_surfaces(oldp,oldreq)
            require(rc['accepted_body_poses']==orc['accepted_body_poses'],'initial physical pose changed')
            require(len(osurf)==868 and set(surfs)-set(osurf)=={(51005,s) for s in IDS},'initial inventory differs')
            require(all(np.array_equal(pos[np.asarray(surfs[k]['faces'])],opos[np.asarray(v['faces'])]) for k,v in osurf.items()),'initial original surface changed')
            baseline_exact=True
        worlds={};checks={}
        for s,r in rows.items():
            f=np.asarray(surfs[(51005,s)]['faces'],dtype=np.int64);local=r['faces'].astype(np.int64);require(f.shape==local.shape,'native face shape mismatch')
            delta=f-local;require(np.all(delta==delta.flat[0]),'native face remap mismatch');first=int(delta.flat[0]);n=len(r['positions']);require(0<=first and first+n<=len(pos),'native vertex span')
            xyz=pos[first:first+n].astype('<f4');worlds[s]=xyz
            top=referenced_topology(xyz,local);signs=ns['sign'](ns['volume_signs'](xyz,local,top['face_components']))
            if step in (0,decl['accepted_steps']):count,dg,witness=ns['exact_audit'](h,xyz,local,ci)
            else:
                from fractions import Fraction
                dg=[]
                for fi,face in enumerate(local):
                    tri=[[Fraction.from_float(float(v)) for v in xyz[idx]] for idx in face]
                    ab=[tri[1][k]-tri[0][k] for k in range(3)];ac=[tri[2][k]-tri[0][k] for k in range(3)]
                    if all(ab[(k+1)%3]*ac[(k+2)%3]-ab[(k+2)%3]*ac[(k+1)%3]==0 for k in range(3)):dg.append(fi)
                count=None;witness=[]
            okay=not dg and top['closed_oriented_manifold_candidate'] and len(top['face_components'])==reference[s]['components'] and signs==reference[s]['signs']
            checks[str(s)]={'nondegenerate_topology_preserved':bool(okay),'degenerate_face_ids':list(map(int,dg)),'raw_self_pair_count':count,'raw_self_pairs':[list(map(int,x)) for x in witness],'component_signs':signs,'topology':ns['summarize_topology'](top)}
            np.savez_compressed(out/f'step-{step}-row-{s}.npz',positions=xyz,faces=local)
            save(out/f'step-{step}-row-{s}.json',checks[str(s)])
        interfaces={}
        for (aa,bb),ps in pairs.items():
            mismatch=[(i,j) for i,j in ps if worlds[aa][i].tobytes()!=worlds[bb][j].tobytes()]
            interfaces[f'{aa}+{bb}']={'expected_paired_faces':EXPECTED[(aa,bb)],'unique_vertex_pairs':len(ps),'mismatch_count':len(mismatch),'mismatches':mismatch}
        result={'rows':checks,'interfaces':interfaces,'original_868_surfaces_exact_to_012':baseline_exact,'body_poses_exact_to_012':baseline_exact,'elapsed_s':time.monotonic()-start}
        report['captures'][str(step)]=result;save(out/'report.json',report)
        print(json.dumps({'step':step,'collapsed_faces':{s:len(c['degenerate_face_ids']) for s,c in checks.items()},'self_pairs':{s:c['raw_self_pair_count'] for s,c in checks.items()},'interface_mismatches':sum(x['mismatch_count'] for x in interfaces.values())}),flush=True)
    report['complete']=True;report['no_collapsed_faces_at_declared_captures']=all(c['nondegenerate_topology_preserved'] for p in report['captures'].values() for c in p['rows'].values())
    report['all_shared_interfaces_exact']=all(c['mismatch_count']==0 for p in report['captures'].values() for c in p['interfaces'].values())
    require(all(sha(p)==v for p,v in pins.items()),'audit input changed');save(out/'report.json',report)
if __name__=='__main__':main()

