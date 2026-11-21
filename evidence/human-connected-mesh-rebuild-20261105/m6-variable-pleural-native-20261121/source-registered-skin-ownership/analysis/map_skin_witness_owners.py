"""Trace source-stable skin triangles to the actual skeletal binding contributors."""
import hashlib,json,sys
from collections import Counter
from pathlib import Path
import numpy as np
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def map_owners(work,receipt,capture,report,output):
    sys.path.insert(0,str(work/'resting-clearance-and-atrium-002/publication/scripts'))
    from rebind_human_skin_reference import decode
    r=json.loads(receipt.read_text());c=json.loads((receipt.parent/'launch.json').read_text());audit=json.loads(report.read_text())
    source=Path(c['argv'][c['argv'].index('--skin-payload')+1]);tissue=Path(c['argv'][c['argv'].index('--soft-tissue-payload')+1])
    man=tissue.with_suffix('.manifest.json');m=json.loads(man.read_text())
    if digest(tissue)!=m['payload']['sha256'] or digest(source)!=c['pins'][str(source)]:raise ValueError('Source skin or muscles changed')
    q=decode(source.read_bytes());f=q['inds'];w=q['weights'];p=np.load(capture)
    if (str(p['pack_sha256'])!=audit['accepted_pack_sha256'] or r['pack_file_sha256']!=str(p['pack_sha256']) or
        str(p['faces_sha256'])!=hashlib.sha256(f.astype('<u4').tobytes()).hexdigest()):raise ValueError('Stable triangle source correspondence required')
    name={};surfaces={s['stable_id']:s for s in m['source']['surfaces']}
    for s in surfaces.values():
        for b in s.get('body_bindings',[]):
            code=b['core_body_index'];text=b['myosim_body']
            if code in name and name[code]!=text:raise ValueError('Inconsistent skeletal owner')
            name[code]=text
    bone=q['binds']['body']
    # A face receives its original three vertices' mean source weights.
    face_w=w[f].mean(1);donor=bone[face_w.argmax(1)]
    def label(k):return name.get(int(k),'body '+str(k))
    def tops(c):return [{'owner':k,'pairs':v} for k,v in c.most_common(12)]
    def owners(face_ids):return Counter(label(x) for x in donor[face_ids])
    pairs=np.array(audit['self_intersections']['triangle_pairs'],int).reshape(-1,2)
    shared=Counter(str(len(set(f[i])&set(f[j]))) for i,j in pairs)
    shared_features=dict(shared)
    selfpairs=Counter(' / '.join(sorted([label(donor[a]),label(donor[b])])) for a,b in pairs)
    tissue_owners=Counter();shared=0;total=0;bybank=[]
    for s in audit['surfaces']:
        if not s['count'] or s['ocular_source_id']:continue
        ids=np.array(s['triangle_pairs'],int)[:,0]
        tissue_owners.update(owners(ids))
        if s['semantic']==51005:
            refs=surfaces[s['stable_id']];owners_id=[b['core_body_index'] for b in refs['body_bindings']]
            match=np.isin(donor[ids],owners_id);total+=len(ids);shared+=int(match.sum())
            bybank.append(dict(name=refs['label'],pairs=len(ids),skin_dominant_bone_is_muscle_binding=int(match.sum()),skin_body_owners=tops(owners(ids))))
    used=np.unique(f);weight4=np.sort(w[used],axis=1)[:,-4:].sum(1,dtype=float)
    result={'step':r['accepted_step'],'skin_sha256':digest(source),'pack_sha256':r['pack_file_sha256'],'source_face_order_verified':True,
        'self_pairs_by_number_of_shared_indexed_vertices':shared_features,'self_dominant_body_pairs':tops(selfpairs),'outer_skin_dominant_bodies':tops(tissue_owners),'muscle_pairs':total,'muscle_pairs_sharing_dominant_skin_bone':shared,
        'four_largest_weights_sum_quantiles_0_50_100':list(map(float,np.quantile(weight4,[0,.5,1]))),'muscles':sorted(bybank,key=lambda r:-r['pairs']),
        'limit':'Skin donors are ranked by the mean source weights of the three face vertices. Sharing a donor with a muscle does not establish shared deformation or physical contact.'}
    output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':map_owners(*map(Path,sys.argv[1:]))
