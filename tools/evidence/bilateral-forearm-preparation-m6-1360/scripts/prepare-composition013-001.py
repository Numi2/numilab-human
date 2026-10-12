#!/usr/bin/env python3
"""Compose right128 into native012 for loader preflight only; no launch/admission."""
from pathlib import Path
import hashlib, importlib.util, json, os, sys
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
D=A/'forearm-shared-field-native-012-diagonal097'
B=Path('/Users/n/numi-human-fds132-m6-1350/attempt-128-right-carrier-assets-104-to-122-002')
OUT=A/'bilateral-forearm-composition-preflight-013-001'
HELPER=A/'prepare_forearm_shared_field_native_012_diagonal097.py'
OWNER=Path('/Users/n/numi-human-left-pq-publication-1344/src')
OVERLAY=B.parent/'attempt-100-pq130-shared-route-field-screen-001/097-lineage-overlay-001.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v,m):
    if not v: raise RuntimeError(m)
def save(p,j): p.write_text(json.dumps(j,indent=2,sort_keys=True)+'\n')
def main():
    require(not OUT.exists(),'preserve previous output')
    pins={}
    def pin(p,expected=None):
        p=Path(p);digest=sha(p)
        require(expected is None or digest==expected,'input SHA mismatch '+str(p))
        pins[str(p)]=digest
    pin(D/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue','88342f64d7f76b467ecc72d32f54248e37c01ab0c81c0c096f465158411879b2')
    pin(D/'passive-tissue-parent-bindings.json','3527a4fe0a3b1be1e99d39a385af50a428320c3881810bd30529ca6725adb678')
    pin(OVERLAY,'68a69c1e987a75dcca8da138f102d7a705df00ddc4e63cd24c80029d420d83bc')
    pin(HELPER);pin(B/'preparation.json');pin(Path(__file__))
    prep=json.loads((B/'preparation.json').read_text())
    for path,digest in prep['inputs_sha256'].items(): pin(path,digest)
    spec=importlib.util.spec_from_file_location('prep012',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    h=helper.helper_module(helper.FORWARD)
    sys.path.insert(0,str(OWNER))
    from numilab_human import passive_attachment_composition as pc
    pin(Path(pc.__file__));pin(helper.RIGID);pin(helper.ARCHIVE);pin(helper.FORWARD)
    parentpath=D/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
    parent=h['load_tissue'](parentpath)
    carrierpath=Path(prep['carrier_payload']['path']);pin(carrierpath,prep['carrier_payload']['sha256'])
    carrier=h['load_tissue'](carrierpath)
    poses,poseproof=helper.decode_default_poses(helper.RIGID,helper.ARCHIVE)
    OUT.mkdir();(OUT/'patches').mkdir()
    rule=('Explicit inferred load-time compatibility field: select the four largest full21 reference coefficients among cores available in the unchanged original child binding table, break ties by canonical core order, normalize, and serialize in ascending core order. This field only supplies the loader childRest reference; the full21 source-parent extension exclusively owns runtime motion. It does not replace physical muscle or tendon bindings.')
    replacements=[];rest={};data={}
    for sid in (131,133):
        item=prep['geometry_rows'][str(sid)];path=Path(item['path']);pin(path,item['sha256'])
        with np.load(path,allow_pickle=False) as z: d={k:z[k].copy() for k in z.files}
        n=len(d['vertices6']);old=h['row_data'](parent,sid)
        binds=parent['bindings'][old['fb']:old['fb']+old['bc']]
        cores=list(map(int,d['core_body_indices']))
        corelocal={int(b['core']):i for i,b in enumerate(binds)}
        require(len(corelocal)==len(binds),'duplicate child binding core')
        available=[i for i,c in enumerate(cores) if c in corelocal]
        require(len(available)>=4,'insufficient compatible cores')
        raw=d['core_weights_float64'][:,available]
        require(np.isfinite(raw).all() and np.all(raw>0),'invalid reference coefficients')
        order=np.argsort(-raw,axis=1,kind='stable')[:,:4]
        cols=np.asarray(available)[order];cols.sort(axis=1)
        selected=np.take_along_axis(d['core_weights_float64'],cols,axis=1)
        weights=(selected/selected.sum(axis=1)[:,None]).astype('<f4')
        local=np.asarray([[corelocal[cores[int(c)]] for c in row] for row in cols],dtype='<u4')
        # Every compatible binding is byte-identical to the corresponding full-field carrier.
        cb=carrier['bindings']
        cbmap={int(b['core']):b.tobytes() for b in cb}
        require(all(b.tobytes()==cbmap[int(b['core'])] for b in binds),'noncanonical child binding')
        patch=OUT/'patches'/('row-'+str(sid)+'.npz')
        np.savez_compressed(patch,vertices6=d['vertices6'],faces=d['faces_local'],binding_indices=local,weights=weights)
        proof=OUT/'patches'/('row-'+str(sid)+'.json')
        save(proof,{'status':'preparation_not_admitted','source_path':str(path),'source_sha256':sha(path),'binding_rule':rule,'stable_id':sid,'patch_sha256':sha(patch)})
        replacements.append((sid,patch,proof))
        childworld=h['forward'](d['vertices6'][:,:3],local,weights,old,parent,poses)
        parentworld=np.zeros((n,3),dtype='<f4')
        crow=h['row_data'](carrier,sid)
        ccores={int(b['core']):i for i,b in enumerate(carrier['bindings'][crow['fb']:crow['fb']+crow['bc']])}
        one=np.zeros((n,4),dtype='<f4');one[:,0]=1
        clocal=np.zeros((n,4),dtype='<u4')
        for col,core in enumerate(cores):
            clocal[:,0]=ccores[core]
            world=h['forward'](d['vertices6'][:,:3],clocal,one,crow,carrier,poses)
            coeff=d['core_weights_float64'][:,col].astype('<f4')
            parentworld=np.add(parentworld,np.multiply(coeff[:,None],world,dtype='<f4'),dtype='<f4')
        rest[str(sid)]=helper.stats(parentworld,childworld)
        data[sid]=d
        print(json.dumps({'row':sid,'rest':rest[str(sid)]}),flush=True)
    save(OUT/'rest-registration-preflight.json',{'status':'passed' if all(v['native_float32_gate_failures']==0 for v in rest.values()) else 'failed','rows':rest,'binding_rule':rule,'poseproof':poseproof,'gpu_used':False,'anatomical_admission_claimed':False})
    require(all(v['native_float32_gate_failures']==0 and v['max_um']<=2 for v in rest.values()),'fixed loader rest gate failed')
    composition=pc.compose(parentpath,OUT/'assets',replacements,reference_surface_rows=(131,133))
    childpath=OUT/'assets'/parentpath.name
    child=h['load_tissue'](childpath)
    for row in parent['records']:
        sid=int(row[6])
        if sid not in (131,133):
            before=h['row_data'](parent,sid);after=h['row_data'](child,sid)
            require(before['vertex_bytes']==after['vertex_bytes'] and np.array_equal(before['faces'],after['faces']),'untargeted child changed')
    ext=json.loads((D/'passive-tissue-parent-bindings.json').read_text())
    leftsource=D/ext['source_payloads'][0]['path']
    pin(leftsource,ext['source_payloads'][0]['sha256'])
    os.link(leftsource,OUT/'assets/left-full21-carriers.nhtissue')
    os.link(carrierpath,OUT/'assets/right-full21-carriers.nhtissue')
    ext['source_payloads']=[{'path':'assets/left-full21-carriers.nhtissue','sha256':sha(leftsource)},{'path':'assets/right-full21-carriers.nhtissue','sha256':sha(carrierpath)}]
    termsfile=B/'assets/right-parent-surface-terms-native013-source-index-1.json'
    pin(termsfile,'0271c31cf6aabf9738c0c5171e69e1f4928cb20bb9a58b30d6009f9fc9aee75e')
    terms=json.loads(termsfile.read_text())['surfaces']
    require([s['stable_id'] for s in terms]==[131,133],'right surface order')
    for surface in terms:
        sid=surface['stable_id'];d=data[sid]
        require(len(surface['vertices'])==len(d['vertices6']),'child/term cardinality')
        for vi,vs in enumerate(surface['vertices']):
            require(len(vs)==21,'truncated runtime field')
            require(all(t[0]==1 and t[1]==sid and float(t[3])==float(d['core_weights_float64'][vi,col]) for col,t in enumerate(vs)),'runtime coefficients changed')
    ext['surfaces']+=terms;ext['child_payload_sha256']=sha(childpath)
    ext['provenance']['right_preparation_path']=str(B/'preparation.json')
    ext['provenance']['right_preparation_sha256']=sha(B/'preparation.json')
    ext['provenance']['left097_pooled_derived_face_lineage_overlay']={'path':str(OVERLAY),'sha256':sha(OVERLAY)}
    extpath=OUT/'passive-tissue-parent-bindings.json';save(extpath,ext)
    count=sum(len(vs) for s in ext['surfaces'] for vs in s['vertices'])
    require(count<=4194304 and extpath.stat().st_size<=128*1024*1024,'native parent extension capacity')
    require(all(sha(p)==digest for p,digest in pins.items()),'input changed')
    save(OUT/'preparation.json',{'status':'loader_preflight_passed_anatomical_failure_retained_no_launch','input_sha256':pins,'rows':rest,'binding_rule':rule,'composition':composition,'child_payload_sha256':sha(childpath),'extension_sha256':sha(extpath),'full_runtime_influences':count,'extension_bytes':extpath.stat().st_size,'left012_untargeted_rows_byte_exact':True,'known_blocker':'Right122 has source/pose Float32 residual intersections and four current8000 Float64 external intersections. This composed package is not anatomically admitted and has no launch declaration.','gpu_used':False})
    print(json.dumps({'status':'loader_preflight_passed_anatomical_failure_retained_no_launch','output':str(OUT),'influences':count}),flush=True)
if __name__=='__main__': main()

