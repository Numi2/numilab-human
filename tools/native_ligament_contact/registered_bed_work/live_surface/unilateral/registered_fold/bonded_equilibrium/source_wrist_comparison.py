"""Localize forearm crossings to registered skin input or the installed material envelope."""
import json,runpy,sys
from pathlib import Path
import numpy as np
from material_context import sha
from opening import read_opening

def read(p):return json.loads(p.read_text())

def compare(case,reader,decoder):
    run=case/'three';job=read(run/'launch.json');arg=job['argv']
    s=Path(arg[arg.index('--skin-payload')+1]);m=Path(arg[arg.index('--soft-tissue-payload')+1])
    manifest=read(m.with_suffix('.manifest.json'))
    if sha(s)!=job['pins'][str(s)] or sha(m)!=job['pins'][str(m)] or manifest['payload']['sha256']!=sha(m):raise ValueError('Changed source input')
    skin=runpy.run_path(str(decoder))['decode'](s.read_bytes())
    src=np.loadtxt(run/'registered-skin-native.csv',skiprows=1,delimiter=',')
    if len(src)!=skin['nv'] or not np.array_equal(src[:,0],np.arange(len(src))):raise ValueError('Different source vertex identity')
    origin=read(run/'registered-initial-bodies.json')['all_bodies'];actual=read(run/'native-initial-bodies.json')['all_bodies']
    def column(data,name):return np.array([v['current'][name] for v in data])
    rotations=column(origin,'local_to_world_row_major_f32').reshape(-1,3,3)
    if not np.array_equal(rotations,column(actual,'local_to_world_row_major_f32').reshape(-1,3,3)):raise ValueError('Need the general articulated replay')
    pos=column(origin,'position_world_m');delta=column(actual,'position_world_m')-pos
    owners=src[:,1].astype(int)
    if np.linalg.norm(pos[owners]+(rotations[owners]@src[:,8:11,None])[:,:,0]-src[:,2:5],axis=1).max()>2e-7:raise ValueError('Source vertex and receipt disagree')
    spread=float(np.linalg.norm(delta-delta[0],axis=1).max())
    if spread>1e-7:raise ValueError('Translation-only source comparison is not accurate in this pose')
    weights=skin['weights'].astype(float);weights/=weights.sum(1)[:,None]
    x=(src[:,2:5]+weights@delta[skin['binds']['body']]).astype('f4')
    faces=skin['inds']
    sys.path.insert(0,str(reader.parent/'src'));audit=runpy.run_path(str(reader));c=audit['clearance'];ci=audit['intersections']
    def index(coords,tri):
        rec=c._exact_surface_records(coords,tri)
        return ci._prepare_surface_aabb(rec),c._float32_skin_aabb_index(rec)
    def compute_hits(idx,broad,y,g):
        target,checks=c._candidate_target_rows_for_skin_aabb(g,y,broad);chosen=np.asarray(target,dtype=int)
        ids=np.unique(g[chosen]);r=c._exact_surface_records(y[ids],np.searchsorted(ids,g[chosen]))
        h=ci._audit_pair_prepared_first(idx,[(a,b,q,int(chosen[k]),e) for a,b,q,k,e in r])
        if h['aabb_candidate_pairs']!=checks:raise ValueError('Broad phase mismatch')
        return {tuple(t) for t in h['triangle_pairs']}
    tree,broad=index(x,faces)
    e=job['environment'];prefix='NUMI_HUMAN_PHYSICAL_SKIN_REGISTERED_VOLUME_PREFLIGHT_SURFACE_O_'
    fn=Path(e[prefix+'NODES_CSV']);ff=Path(e[prefix+'FACES_CSV'])
    if sha(fn)!=e[prefix+'NODES_SHA256'] or sha(ff)!=e[prefix+'FACES_SHA256']:raise ValueError('Different registered envelope')
    coords=np.loadtxt(fn,delimiter=',',skiprows=1,usecols=(0,1,2,3,4))
    of=np.loadtxt(ff,delimiter=',',skiprows=1,usecols=(1,2,3),dtype=np.int32)
    if not np.array_equal(coords[:,0],np.arange(len(coords))):raise ValueError('Registered O rows are unordered')
    reference=(coords[:,2:5]+delta[coords[:,1].astype(int)]).astype('f4')
    registered_tree,registered_broad=index(reference,of)
    poses={r['body_index']:{'position_m':r['current']['position_world_m'],'quaternion_xyzw':r['current']['orientation_world_xyzw']} for r in actual}
    exact=case/'executed-envelope/target-witnesses.jsonl'
    hits=[json.loads(l) for l in exact.read_text().splitlines()]
    select={r['stable_id']:r for r in sorted((r for r in hits if r['semantic']==51005),key=lambda r:-r['count'])[:4]}
    names={r['stable_id']:r.get('label',r.get('member_id')) for r in manifest['source']['surfaces'] if r['layer']=='muscle'}
    results=[]
    with read_opening(case,read(case/'opening.json')['pack']) as raw:
        for (sem,stable),i,y,g in audit['primitives'](raw,poses):
            if sem!=51005 or stable not in select:continue
            original=select[stable]
            if original['primitive']!=i:raise ValueError('Different muscle bank')
            src_pairs=compute_hits(tree,broad,y,g);registered=compute_hits(registered_tree,registered_broad,y,g)
            installed_pairs={tuple(t) for t in original['triangle_pairs']}
            srcfaces={b for a,b in src_pairs};installed={b for a,b in installed_pairs}
            regfaces={b for a,b in registered}
            results.append(dict(stable=stable,name=names[stable],source_triangle_pairs=len(src_pairs),registered_reference_pairs=len(registered),
                 registered_reference_muscle_faces=len(regfaces),common_registered_source_muscle_faces=len(regfaces&srcfaces),
                 common_registered_installed_muscle_faces=len(regfaces&installed),common_registered_installed_pairs=len(registered&installed_pairs),
                 installed_triangle_pairs=original['count'],
                 source_muscle_faces=len(srcfaces),installed_muscle_faces=len(installed),common_muscle_faces=len(srcfaces&installed),
                 introduced_muscle_faces=len(installed-srcfaces),cleared_muscle_faces=len(srcfaces-installed)))
    if len(results)!=4:raise ValueError('Incomplete dominant bank comparison')
    return dict(source_skin=sha(s),source_tissue=sha(m),executed_pose=sha(run/'native-initial-bodies.json'),source_observation=sha(run/'registered-skin-native.csv'),
        maximum_body_translation_spread_m=spread,registered_outer_nodes=sha(fn),registered_outer_faces=sha(ff),changed_rotation_entries=0,source_vertices=len(x),installed_witness_sha256=sha(exact),
        comparison='Same target muscle triangles tested against source, registered outer boundary, and installed outer boundary at the executed initial pose.',banks=sorted(results,key=lambda r:-r['installed_triangle_pairs']))
if __name__=='__main__':print(json.dumps(compare(*(Path(p).resolve() for p in sys.argv[1:])),indent=2))
