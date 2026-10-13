#!/usr/bin/env python3
"""Prepare existing native resting scene with passive vessel attachment terms."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,struct,sys
import numpy as np
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def req(v,m):
    if not v:raise ValueError(m)
def save(p,d):p.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
def registered_vessel_rest_preflight(loaded, declaration, pins):
    # Asset-load verification only. The existing Metal path remains the
    # sole owner of simulation updates.
    import importlib.util
    hp=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276/prepare_forearm_shared_field_native_012_diagonal097.py')
    spec=importlib.util.spec_from_file_location('vascular047_reference_rest',hp)
    helper=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    rigid=Path(declaration['owner_cli_preview']['native_argv'][1]).resolve()
    req(rigid==helper.RIGID.resolve(),'vascular registered-rest rigid owner differs')
    h=helper.helper_module(helper.FORWARD)
    for path in (hp,helper.FORWARD,helper.RIGID,helper.ARCHIVE):
        pins[str(path.resolve())]=sha(path)
    poses,poseproof=helper.decode_default_poses(helper.RIGID,helper.ARCHIVE)
    rows={}
    for item in loaded:
        z=item['z']; n=item['rr'][2]; body=item['body']
        p,q=poses[body]
        expected=np.asarray(p[:,None]+h['rotate32'](q,z['source_local_positions'].T),dtype='<f4').T
        world=np.zeros((n,3),dtype='<f4')
        for parent in item['bodies']:
            bp,bq=poses[parent]
            transformed=np.asarray(bp[:,None]+h['rotate32'](bq,z['body'+str(parent)+'_local_positions'].T),dtype='<f4').T
            world=np.add(world,np.multiply(z['weight_body'+str(parent)][:,None],transformed,dtype='<f4'),dtype='<f4')
        stats=helper.stats(world,expected)
        req(stats['native_float32_gate_failures']==0 and stats['max_um']<=2,
            'fixed native vessel registered-rest gate failed')
        cached_blend=float(np.linalg.norm(world.astype(float)-z['registered_blended_rest_positions'].astype(float),axis=1).max())
        cached_source=float(np.linalg.norm(expected.astype(float)-z['registered_rest_world_positions'].astype(float),axis=1).max())
        req(cached_blend<=2e-6 and cached_source<=2e-6,'vessel cached registered-rest metadata differs')
        stats.update(recomputed_vs_cached_blend_max_um=cached_blend*1e6,
                     recomputed_vs_cached_source_max_um=cached_source*1e6)
        rows[str(item['sid'])]=stats
    return rows,poseproof

def main():
    ap=argparse.ArgumentParser()
    for k in ('parent','source-preparation','output'):ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--field',type=Path,action='append',required=True)
    ap.add_argument('--tissue-composition',type=Path)
    a=ap.parse_args();parent=a.parent.resolve();out=a.output.resolve()
    req(not out.exists(),'preserve prior candidate')
    d=json.loads((parent/'baseline/run-declaration.json').read_text())
    ex=json.loads((parent/'baseline/execution.json').read_text())
    req(ex['returncode']==0 and not ex['timed_out'] and not ex['changed_inputs'],'parent failed')
    pins=dict(d['immutable_assets'])
    req(all(sha(p)==h for p,h in pins.items()),'parent pin drift')
    src=d['argv'];mi=src.index('numilab_human.resting_run');tail=src[mi+1:].copy()
    rpold=Path(tail[tail.index('--anatomy-receipt')+1]).resolve()
    receipt=json.loads(rpold.read_text())
    oldmuscle=Path(receipt['provenance']['native_muscle_surfaces']['payload_path']).resolve()
    newmuscle=oldmuscle
    old_anatomy=Path(receipt['payload']['path']).resolve()
    req(sha(old_anatomy)==receipt['payload']['sha256'],'anatomy drift')
    source_report=a.source_preparation/'preparation.json'
    source_proof=json.loads(source_report.read_text())
    req(source_proof['input_sha256'][str(old_anatomy.resolve())]==sha(old_anatomy),
        'paired source parent differs')
    req(source_proof['records_indices_registration_byte_exact'] and
        len(source_proof['unchanged_source_rows'])+len(source_proof['changed_rows'])==523,'paired source proof differs')
    source_pins={str(Path(p).resolve()):h for p,h in source_proof['input_sha256'].items()}
    req(all(sha(p)==h for p,h in source_pins.items()),'vascular preparation input drift')
    pins.update(source_pins)
    anatomy=a.source_preparation/'resting-thorax.nhanatomy'
    req(sha(anatomy)==source_proof['payload_sha256'],'paired source payload drift')
    oldsha=receipt['payload']['sha256'];newsha=sha(anatomy)
    receipt['payload'].update(path=str(anatomy.resolve()),sha256=newsha,input_payload_sha256=oldsha)
    receipt['functional_bindings']['anatomy_payload_sha256']=newsha
    receipt['provenance']['cardiac_geometry_binding']['common_field']['anatomy_payload_sha256']=newsha
    receipt['passive_tissue_parent_bindings']['anatomy_payload_sha256']=newsha
    raw=anatomy.read_bytes()
    header=struct.unpack_from('<8s5I32s',raw,0)
    records=[struct.unpack_from('<8I',raw,60+32*i) for i in range(header[2])]
    vertexoffset=60+32*header[2];indexoffset=vertexoffset+24*header[3]
    loaded=[]
    for field in a.field:
        z=np.load(field,allow_pickle=False)
        sid=int(z['stable_surface_id'][0]);body=int(z['source_body'][0])
        rr=next(r for r in records if r[5]==sid)
        req(rr[0]==body and rr[6]==2,'source vessel ownership differs')
        bodies=list(map(int,z['target_body_ids']))
        req(0<len(bodies)<=21 and len(set(bodies))==len(bodies) and
            all(0<=b<157 for b in bodies),'invalid attachment body inventory')
        vertices=np.frombuffer(raw,dtype='<f4',offset=vertexoffset+24*rr[1],count=6*rr[2]).reshape((-1,6))
        faces=np.frombuffer(raw,dtype='<u4',offset=indexoffset+4*rr[3],count=rr[4]).reshape((-1,3))-rr[1]
        req(vertices[:,:3].tobytes()==z['source_local_positions'].tobytes() and
            vertices[:,3:].tobytes()==z['source_local_normals'].tobytes() and
            faces.tobytes()==z['faces'].tobytes(),'field source anatomy differs')
        weights=np.column_stack([z['weight_body'+str(b)] for b in bodies])
        req(weights.shape==(rr[2],len(bodies)) and np.isfinite(weights).all() and
            np.all(weights>=0) and np.all(weights<=1),'invalid material weights')
        req(np.max(np.abs(weights.astype(float).sum(axis=1)-1))<1e-7,'invalid native partition')
        resterr=np.linalg.norm((z['registered_blended_rest_positions']-z['registered_rest_world_positions']).astype(float),axis=1)
        req(float(resterr.max())<=2e-6,'registered rest residual exceeds 2um')
        loaded.append(dict(sid=sid,body=body,bodies=bodies,z=z,rr=rr,faces=faces,
                           weights=weights,resterr=float(resterr.max()),field=field))
    req(len({x['sid'] for x in loaded})==len(loaded),'duplicate anatomical surface')
    vessel_rest_rows,poseproof=registered_vessel_rest_preflight(loaded,d,pins)
    out.mkdir();(out/'assets').mkdir();(out/'baseline').mkdir()
    for p in (parent/'assets').iterdir():
        if p.is_file():os.link(p,out/'assets'/p.name)
    ext=receipt['passive_tissue_parent_bindings']
    req([x['stable_id'] for x in ext['anatomy_surfaces']]==[477] and len(ext['source_payloads'])==5,'unexpected prior anatomy parents')
    req(all(t[0]==4 for vs in ext['anatomy_surfaces'][0]['vertices'] for t in vs),'prior vessel source differs')
    parent0=parent/ext['source_payloads'][0]['path']
    th=struct.unpack_from('<8s6I32s',parent0.read_bytes(),0)
    req(th[0]==b'NHTISS4\0' and th[1]==5 and th[6]==header[5] and th[7]==header[6],
        'carrier and anatomy registration differ')
    # One-body carrier rows use the already supported NHTISS ABI5. These
    # are load-time carriers only and create no rendered shell, force or mass.
    recs=[];bindings=[];vb=[];ix=[];nv=ni=nb=0
    for item in loaded:
        z=item['z'];n=item['rr'][2];faces=item['faces'];item['carrier_ids']=[]
        for body in item['bodies']:
            carrier_id=len(recs)+1;item['carrier_ids'].append(carrier_id)
            recs.append(struct.pack('<8I',nb,1,nv,n,ni,faces.size,carrier_id,1))
            bindings.append(struct.pack('<I8f',body,0,0,0,0,0,0,1,1))
            points=z['body'+str(body)+'_local_positions']
            normals=z['body'+str(body)+'_local_normals']
            req(points.shape==(n,3) and normals.shape==(n,3) and
                np.isfinite(points).all() and np.isfinite(normals).all(),'invalid carrier coordinates')
            req(np.max(np.abs(np.linalg.norm(normals.astype(float),axis=1)-1))<1e-3,'invalid unit normals')
            row=bytearray(n*56)
            for i in range(n):
                struct.pack_into('<6f4I4f',row,i*56,*points[i],*normals[i],0,0,0,0,1,0,0,0)
            vb.append(bytes(row));ix.append(np.asarray(faces+nv,dtype='<u4').tobytes())
            nv+=n;ni+=faces.size;nb+=1
    req(nv<=1_000_000 and len(recs)<=192,'existing parent payload bound exceeded')
    carrier=out/'assets/vascular-reference-carriers-042.nhtissue'
    carrier.write_bytes(struct.pack('<8s6I32s',b'NHTISS4\0',5,len(recs),nb,nv,ni,th[6],th[7])+
        b''.join(recs+bindings+vb+ix))
    req(carrier.stat().st_size<=128*1024*1024,'existing parent source byte bound exceeded')
    sourceidx=4
    ext['source_payloads'][sourceidx]={'path':'assets/'+carrier.name,'sha256':sha(carrier)}
    # Choose binary64 coefficients inside each native-float rounding interval.
    # This preserves every GPU coefficient exactly while satisfying the
    # existing loader's exact positive material partition.
    def partition(native):
        active=np.flatnonzero(native>0)
        c=native[active].astype(np.float64)
        lower=(native[active].astype(float)+np.nextafter(native[active],np.float32(0)).astype(float))/2
        upper=(native[active].astype(float)+np.nextafter(native[active],np.float32(np.inf)).astype(float))/2
        lower=np.nextafter(lower,c);upper=np.nextafter(upper,c)
        residual=1.-float(c.sum())
        for j in np.argsort(-(upper-lower)):
            change=min(max(residual,float(lower[j]-c[j])),float(upper[j]-c[j]))
            c[j]+=change;residual=1.-float(c.sum())
        req(abs(residual)<=1e-12 and np.array_equal(c.astype('<f4'),native[active]),
            'native material coefficients cannot form exact partition')
        return active,c
    prior={int(x['stable_id']):x for x in ext['anatomy_surfaces']}
    provenance=[]
    for item in loaded:
        terms=[]
        for row in item['weights']:
            active,c=partition(row)
            terms.append([[sourceidx,item['carrier_ids'][int(k)],len(terms),float(v)] for k,v in zip(active,c)])
        req(all(abs(sum(t[3] for t in ts)-1)<=1e-12 for ts in terms),'partition differs')
        prior[item['sid']]={'stable_id':item['sid'],'vertices':terms}
        provenance.append({'stable_id':item['sid'],'source_body':item['body'],
            'parent_bodies':item['bodies'],'field_path':str(item['field'].resolve()),
            'field_sha256':sha(item['field']),'registered_rest_max_residual_m':item['resterr']})
    ext['anatomy_payload_sha256']=receipt['payload']['sha256']
    ext['anatomy_surfaces']=[prior[k] for k in sorted(prior)]
    if a.tissue_composition:
        comp=a.tissue_composition.resolve()
        prep=json.loads((comp/'preparation.json').read_text())
        req(prep['status']=='loader_preflight_passed_native_pending','muscle source failed preflight')
        req(prep['preserved_back_candidate']=='412' and prep['unchanged_source_rows']==162,
            'back412/unrelated muscle source scope differs')
        req(all(row['max_um']<=2 and row['native_float32_gate_failures']==0
                for row in prep['rest_rows'].values()),'muscle registered rest failed')
        composition_pins={str(Path(p).resolve()):h for p,h in prep['input_sha256'].items()}
        req(all(sha(p)==h for p,h in composition_pins.items()),'muscle composition drift')
        pins.update(composition_pins)
        rest=json.loads((comp/'rest-preflight.json').read_text())
        req(rest['rows']==prep['rest_rows'] and rest['gpu_used'] is False,
            'muscle rest preflight evidence differs')
        cp=comp/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
        req(sha(cp)==prep['child_payload_sha256'],'muscle child differs')
        ce=json.loads((comp/'passive-tissue-parent-bindings.json').read_text())
        req(sha(comp/'passive-tissue-parent-bindings.json')==prep['extension_sha256'] and
            ce['child_payload_sha256']==sha(cp),'muscle parent extension differs')
        sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
        from numilab_human import passive_attachment_composition as pc
        target=out/'assets'/cp.name
        req(sha(target)==sha(oldmuscle),'current native muscle owner differs')
        target.unlink();os.link(cp,target)
        manifest=target.with_suffix('.manifest.json')
        req(manifest.is_file(),'missing current muscle manifest')
        manifest.unlink();os.link(cp.with_suffix('.manifest.json'),manifest)
        br=out/'muscle-composition-proof/resting-anatomy-receipt.json'
        br.parent.mkdir()
        bound=pc.bind_anatomy_receipt(rpold,target,br)
        for key in ('native_muscle_surfaces','reference_surface_composition_binding',
                    'reference_surface_composition_binding_history'):
            if key in bound['provenance']:receipt['provenance'][key]=bound['provenance'][key]
        before={int(row['stable_id']):row for row in ext['surfaces']}
        after={int(row['stable_id']):row for row in ce['surfaces']}
        req(set(before)==set(after),'muscle child map inventory differs')
        req(all(before[k]==after[k] for k in before if k not in (132,134)),
            'unrelated muscle parent maps changed')
        mapping={}
        for i,rec in enumerate(ce['source_payloads']):
            sp=Path(rec['path']);sp=comp/sp if not sp.is_absolute() else sp
            req(sha(sp)==rec['sha256'],'muscle source carrier differs')
            matches=[j for j,existing in enumerate(ext['source_payloads'])
                     if existing['sha256']==rec['sha256']]
            if matches:
                req(len(matches)==1,'ambiguous shared carrier source');mapping[i]=matches[0]
            else:
                req(len(ext['source_payloads'])<8,'combined source bound exceeded')
                dest=out/'assets'/('forearm-combined-parent-'+str(i)+'.nhtissue')
                os.link(sp,dest)
                mapping[i]=len(ext['source_payloads'])
                ext['source_payloads'].append({'path':'assets/'+dest.name,'sha256':sha(dest)})
        for sid in (132,134):
            row=after[sid]
            for terms in row['vertices']:
                for term in terms:term[0]=mapping[term[0]]
            before[sid]=row
        ext['surfaces']=[before[int(row['stable_id'])] for row in ext['surfaces']]
        ext['child_payload_sha256']=sha(target)
        ext['provenance']['forearm_reference_compilation_042']=ce['provenance']
        receipt['provenance']['combined_anatomical_reference_042']={
            'composition_path':str(comp),'composition_sha256':sha(comp/'preparation.json'),
            'source_interfaces_and_physical_owners_preserved':True,
            'preserved_back_candidate':'412','native_admission':'pending'}
        newmuscle=target.resolve()
        for p in (Path(pc.__file__),comp/'preparation.json',comp/'rest-preflight.json',
                  comp/'passive-tissue-parent-bindings.json',cp,cp.with_suffix('.manifest.json'),
                  br,target,manifest):
            pins[str(p.resolve())]=sha(p)
    # Replacing the source requires an explicit field for every old user.
    req(set(prior)=={x['sid'] for x in loaded},'unreplaced vessel still refers to old source')
    receipt['provenance']['passive_vascular_reference_registration']={
        'surfaces':provenance,'source_payload_path':str(anatomy.resolve()),
        'source_payload_sha256':newsha,'source_proof_path':str(source_report.resolve()),
        'source_proof_sha256':sha(source_report),
        'interpretation':'Inferred mixed-source adult vascular attachment, compiled into the existing accepted-pose GPU parent buffer. Native coefficients match the pinned fields exactly. Sources, laterality, topology, blood-state ownership and physical force routes are preserved; carriers add no mass or forces. Native and anatomical admission remain pending.',
        'native_admission':'pending'}
    receipt['qualification']='Native paired passive vascular attachment diagnostic; back412, repaired skin and retired liver alias preserved. Complete anatomy pending forearm, vessel interfaces and integrated final run.'
    for key in ('map','polynomials','domain_boxes'):
        rec=receipt['provenance']['cardiac_geometry_binding']['common_field'][key]
        rel=Path(rec['path']);req(not rel.is_absolute() and '..' not in rel.parts,'invalid common path')
        req(sha(parent/rel)==rec['sha256'],'common geometry drift')
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True);os.link(parent/rel,target)
    rp=out/'resting-anatomy-receipt.json';rp.write_text(json.dumps(receipt,separators=(',',':'))+'\n')
    req(rp.stat().st_size<=128*1024*1024,'receipt exceeds native limit')
    oldout=Path(tail[tail.index('--output')+1]).resolve()
    tail[tail.index('--anatomy-receipt')+1]=str(rp);tail[tail.index('--output')+1]=str(out/'baseline/native-run')
    env=dict(x.split('=',1) for x in src[2:mi-2])
    env['NUMI_HUMAN_RESTING_COMMON_FAILURE_RECEIPT']=str(out/'baseline/native-run/common-field-failure.json')
    sys.path.insert(0,env['PYTHONPATH'])
    from numilab_human import resting_run as owner
    parser=argparse.ArgumentParser();owner.add_arguments(parser);native,assets=owner.command(parser.parse_args(tail))
    oldnative=d['owner_cli_preview']['native_argv']
    req(native==[str(rp) if v==str(rpold) else str(anatomy.resolve()) if v==str(old_anatomy) else str(newmuscle) if v==str(oldmuscle) else v.replace(str(oldout),str(out/'baseline/native-run')) for v in oldnative],
        'nonanatomy native invocation change')
    launcher=out/'launch_arm.py';shutil.copy2(parent/'launch_arm.py',launcher)
    pins.update(assets)
    vessel_rest_path=out/'vascular-rest-preflight.json'
    save(vessel_rest_path,{'rows':vessel_rest_rows,'poseproof':poseproof,'gpu_used':False,
        'native_run_performed':False,'anatomical_admission':False})
    pins[str(vessel_rest_path.resolve())]=sha(vessel_rest_path)

    for p in (Path(__file__),*a.field,rp,carrier,launcher,anatomy,source_report):pins[str(p.resolve())]=sha(p)
    for rec in ext['source_payloads']:
        p=Path(rec['path']);p=out/p if not p.is_absolute() else p
        req(sha(p)==rec['sha256'],'parent carrier drift');pins[str(p.resolve())]=sha(p)
    d.update(immutable_assets=pins,argv=['/usr/bin/env','-i',*[k+'='+v for k,v in sorted(env.items())],src[mi-2],'-m','numilab_human.resting_run',*tail],
        qualification=receipt['qualification'],launch_adapter={'path':str(launcher),'sha256':sha(launcher)},
        owner_cli_preview={'native_argv':native,'asset_sha256':assets,'recorded_invocation_environment':owner.invocation_environment(env)})
    dp=out/'baseline/run-declaration.json';save(dp,d)
    save(out/'launch-guard.json',{'launcher_sha256':sha(launcher),'arms':{'baseline':{'run_declaration_sha256':sha(dp),'native_output':str(out/'baseline/native-run')}}})
    save(out/'preparation.json',{'native_pending':True,'parent':str(parent),'field_sha256':{str(p.resolve()):sha(p) for p in a.field},
        'carrier_sha256':sha(carrier),'surface_stable_ids':[x['sid'] for x in loaded],'source_geometry_unchanged':not bool(source_proof['changed_rows']),'unrelated_source_rows_byte_exact':len(source_proof['unchanged_source_rows']),
        'registered_rest_max_residual_m':max(x['max_um'] for x in vessel_rest_rows.values())*1e-6,'vessel_rest_rows':vessel_rest_rows,'physiology_or_force_changes':False,'input_sha256':pins})
    req(all(sha(p)==h for p,h in pins.items()),'input drift')
    print(json.dumps({'output':str(out),'carrier_sha256':sha(carrier),'receipt_bytes':rp.stat().st_size,'pins':len(pins)}))
if __name__=='__main__':main()
