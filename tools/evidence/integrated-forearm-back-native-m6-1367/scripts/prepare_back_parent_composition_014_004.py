#!/usr/bin/env python3
"""Prepare passive back anatomy with existing append, composition and parent owners."""
from pathlib import Path
from fractions import Fraction as F
import argparse, hashlib, importlib.util, json, os, struct, sys
import numpy as np
A=Path('/Users/n/numi-human-retained-delivery-20261009/anatomy-completion-1276')
B=Path('/Users/n/numi-human-back-m6-1349')
OWNER=Path('/Users/n/numi-human-left-pq-publication-1344/src')
RAW=[B/'attempt-335-right-raw-source-nhtiss-001/trapezius-153-155-raw-candidate-parents.nhtissue',B/'attempt-334-left-raw-source-nhtiss-001/trapezius-154-156-157-158-raw-candidate-parents.nhtissue']
PID={153:0,155:0,154:1,156:1,157:1,158:1}
SUBSET=A.parent/'passive-neck-back-coverage-1281/candidate-005/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(x,m):
    if not x:raise RuntimeError(m)
def save(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
    ap=argparse.ArgumentParser()
    for x in ('base','descending','output'):ap.add_argument('--'+x,type=Path,required=True)
    ap.add_argument('--upper-candidate',type=Path);ap.add_argument('--width-m',type=float,default=.00387);ap.add_argument('--raw-parent-only',action='store_true');a=ap.parse_args()
    base=a.base.resolve();out=a.output.resolve();require(not out.exists(),'preserve old output');require(0<a.width_m<=.02,'width outside declared sensitivity')
    pins={}
    def pin(p,expected=None):
        p=Path(p);v=sha(p);require(expected is None or v==expected,'hash mismatch '+str(p));pins[str(p)]=v
    sys.path.insert(0,str(OWNER))
    from numilab_human import passive_attachment_composition as pc
    from numilab_human.passive_tissue_parent_carriers import CarrierRow,encode_carriers
    hp=A/'prepare_forearm_shared_field_native_012_diagonal097.py';sp=importlib.util.spec_from_file_location('prep012',hp);helper=importlib.util.module_from_spec(sp);sp.loader.exec_module(helper);h=helper.helper_module(helper.FORWARD)
    basecp=B/'attempt-366-cap-multi-ulp-conditioned-edgeflips-001/candidate-before-audit.json';cp=a.upper_candidate.resolve() if a.upper_candidate else basecp;require(a.upper_candidate is None or a.raw_parent_only,'conditioned upper geometry requires raw-parent-only; frozen field is not recalibrated');fp=B/'attempt-381-common-descending-seam-fourrow-001/frozen-x0-axial-field-table.json'
    for p in (Path(__file__),hp,Path(pc.__file__),OWNER/'numilab_human/passive_tissue_parent_carriers.py',basecp,cp,fp,a.descending,SUBSET,SUBSET.with_suffix('.manifest.json'),helper.RIGID,helper.ARCHIVE,helper.FORWARD,*RAW):pin(p)
    upper=json.loads(cp.read_text());c=upper.get('geometry',upper);dg=json.loads(a.descending.read_text())['geometry'];field=json.loads(fp.read_text())
    require(field['source_artifacts']['attempt366_candidate_json_sha256']==sha(basecp),'field source changed')
    knots=[float(v['z_m']) for v in field['bins']];alphas=[float(v['frozen_isotonic_alpha_26']) for v in field['bins']]
    raw=[h['load_tissue'](p) for p in RAW];rd={(pid,int(r[6])):h['row_data'](t,int(r[6])) for pid,t in enumerate(raw) for r in t['records']}
    canonical={}
    for t in raw:
        for b in t['bindings']:
            core=int(b['core'])
            if core in (20,26):
                require(core not in canonical or canonical[core]==b.tobytes(),'axial binding mismatch');canonical[core]=b.tobytes()
    require(set(canonical)=={20,26},'missing axial binding')
    out.mkdir();(out/'patches').mkdir()
    parentpath=base/'assets/bodyparts3d-myosim-fullbody-muscle-surfaces.nhtissue'
    for p in (parentpath,parentpath.with_suffix('.manifest.json'),base/'passive-tissue-parent-bindings.json'):pin(p)
    appended=out/'append';pc.append_passive_surfaces(parentpath,SUBSET,appended,stable_ids=tuple(range(153,159)))
    parent=h['load_tissue'](appended/parentpath.name);poses,poseproof=helper.decode_default_poses(helper.RIGID,helper.ARCHIVE)
    data={};carrierrows=[]
    for sid in range(153,159):
        src=c if sid<157 else dg
        q=np.array([[float(F(int(x[0]),int(x[1]))) for x in p] for p in src['points_exact'][str(sid)]])
        maps=[]
        for ts in src['parent_maps'][str(sid)]:
            terms=[]
            for t in ts:
                if sid<157 and len(t)==4:s,v,n,d=t;pid=PID[int(s)]
                else:
                    require(len(t)==5,'unsupported source-parent term shape');pid,s,v,n,d=t
                require(int(pid)==PID[int(s)],'raw source owner mismatch')
                terms.append((int(pid),int(s),int(v),F(int(n),int(d))))
            require(all(t[3]>0 for t in terms) and sum((t[3] for t in terms),F())==1,'invalid source partition');maps.append(sorted(terms))
        faces=np.asarray([f for f in src['faces'][str(sid)] if f is not None],dtype='<u4')
        norm=np.zeros_like(q);fn=np.cross(q[faces[:,1]]-q[faces[:,0]],q[faces[:,2]]-q[faces[:,0]])
        for col in range(3):np.add.at(norm,faces[:,col],fn)
        length=np.linalg.norm(norm,axis=1);norm[length==0]=[0,0,1];length[length==0]=1;norm/=length[:,None]
        v6=np.column_stack((q,norm)).astype('<f4')
        t=np.clip(np.abs(q[:,0])/a.width_m,0,1);lam=1-t*t*(3-2*t);alpha=np.clip(np.interp(q[:,2],knots,alphas),0,1)
        # Each generated carrier vertex has one body. Template .5/.5 coefficients
        # only allocate both copies; the extension below owns the frozen field.
        if a.raw_parent_only:lam=np.zeros_like(lam)
        carrierrows.append(CarrierRow(sid,v6,faces,canonical[20]+canonical[26],np.full((len(q),2),.5)))
        data[sid]={'q':q,'maps':maps,'faces':faces,'v6':v6,'lam':lam,'alpha':alpha}
    hdr=struct.unpack_from('<8s6I32s',raw[0]['raw'])
    payload,virtual,carrierproof=encode_carriers(carrierrows,registration_fingerprint=int(hdr[6]),source_digest=hdr[7])
    vp=out/'frozen-axial-carriers.nhtissue';vp.write_bytes(payload);vt={v['stable_id']:v['vertices'] for v in virtual};vc=h['load_tissue'](vp)
    ext=json.loads((base/'passive-tissue-parent-bindings.json').read_text());ns=len(ext['source_payloads']);require(ns+3<=8,'source count cap')
    posed={key:h['forward'](r['positions'],r['local'],r['weights'],r,raw[key[0]],poses) for key,r in rd.items()}
    surfaces=[];replacements=[];rests={}
    for sid,d in data.items():
        old=h['row_data'](parent,sid);binds=parent['bindings'][old['fb']:old['fb']+old['bc']];cl={int(b['core']):i for i,b in enumerate(binds)}
        aggregate=np.zeros((len(d['q']),len(binds)));world=np.zeros((len(d['q']),3),dtype='<f4');vs=[]
        vr=h['row_data'](vc,sid);vw=h['forward'](vr['positions'],vr['local'],vr['weights'],vr,vc,poses)
        for i,ts in enumerate(d['maps']):
            lam=float(d['lam'][i]);alpha=float(d['alpha'][i]);terms=[]
            for pid,s,v,c0 in ts:
                weight=(1-lam)*float(c0)
                if weight<=0:continue
                terms.append([ns+pid,s,v,weight]);r=rd[(pid,s)]
                for slot in range(4):
                    w=float(r['weights'][v,slot])
                    if w<=0:continue
                    core=int(raw[pid]['bindings'][r['fb']+int(r['local'][v,slot])]['core'])
                    if w>0 and core in cl:aggregate[i,cl[core]]+=weight*w
                world[i]=np.add(world[i],np.float32(weight)*posed[(pid,s)][v],dtype='<f4')
            for col,weight in enumerate((lam*(1-alpha),lam*alpha)):
                if weight<=0:continue
                vi=int(vt[sid][i][col][2]);terms.append([ns+2,sid,vi,weight])
                if (20,26)[col] in cl:aggregate[i,cl[(20,26)[col]]]+=weight
                world[i]=np.add(world[i],np.float32(weight)*vw[vi],dtype='<f4')
            require(0<len(terms)<=128 and abs(sum(t[3] for t in terms)-1)<1e-12,'invalid runtime partition');vs.append(terms)
        local=np.zeros((len(vs),4),dtype='<u4');weights=np.zeros((len(vs),4),dtype='<f4')
        for i,w in enumerate(aggregate):
            cols=sorted(int(k) for k in np.argsort(-w,kind='stable')[:4] if w[k]>0);require(cols,'no compatible loader support');total=sum(w[k] for k in cols)
            for j,k in enumerate(cols):local[i,j]=k;weights[i,j]=w[k]/total
        cw=h['forward'](d['v6'][:,:3],local,weights,old,parent,poses);rests[str(sid)]=helper.stats(world,cw)
        patch=out/'patches'/f'row-{sid}.npz';np.savez_compressed(patch,vertices6=d['v6'],faces=d['faces'],binding_indices=local,weights=weights)
        proof=out/'patches'/f'row-{sid}.json';save(proof,{'status':'unadmitted_reference_preparation','input_sha256':pins,'stable_id':sid,'field_width_m':a.width_m,'patch_sha256':sha(patch),'scope':'Passive reference anatomy; physical routes, mass and force owners unchanged.'})
        replacements.append((sid,patch,proof));surfaces.append({'stable_id':sid,'vertices':vs})
        print(json.dumps({'row':sid,'rest':rests[str(sid)],'vertices':len(vs),'max_terms':max(map(len,vs))}),flush=True)
    save(out/'rest-preflight.json',{'rows':rests,'poseproof':poseproof,'gpu_used':False})
    require(all(x['native_float32_gate_failures']==0 and x['max_um']<=2 for x in rests.values()),'fixed loader rest gate failed')
    composition=pc.compose(appended/parentpath.name,out/'assets',replacements,reference_surface_rows=tuple(range(153,159)))
    child=out/'assets'/parentpath.name
    for idx,s in enumerate(ext['source_payloads']):
        p=base/s['path'];pin(p,s['sha256']);name=f'prior-parent-{idx}.nhtissue';os.link(p,out/'assets'/name);s['path']='assets/'+name
    for idx,p in enumerate(RAW if a.raw_parent_only else [*RAW,vp]):
        name=f'back-parent-{idx}.nhtissue';os.link(p,out/'assets'/name);ext['source_payloads'].append({'path':'assets/'+name,'sha256':sha(p)})
    ext['surfaces']+=surfaces;ext['child_payload_sha256']=sha(child)
    ext.setdefault('provenance',{})['back_field']={'path':str(fp),'sha256':sha(fp),'width_m':a.width_m,'raw_parent_only':a.raw_parent_only,'upper_candidate':str(cp),'upper_candidate_sha256':sha(cp),'descending_candidate':str(a.descending),'descending_sha256':sha(a.descending),'runtime_owner':'existing positive parent-influence Metal path','anatomical_admission':False}
    ep=out/'passive-tissue-parent-bindings.json';ep.write_text(json.dumps(ext,separators=(',',':'))+'\n')
    influences=sum(len(v) for s in ext['surfaces'] for v in s['vertices']);require(influences<=4194304 and ep.stat().st_size<=128*1024*1024,'runtime capacity exceeded')
    require(all(sha(p)==v for p,v in pins.items()),'source changed')
    save(out/'preparation.json',{'status':'loader_preflight_passed_native_pending','input_sha256':pins,'base':str(base),'child_payload_sha256':sha(child),'extension_sha256':sha(ep),'full_runtime_influences':influences,'extension_bytes':ep.stat().st_size,'carrier':carrierproof,'rest_rows':rests,'composition':composition,'width_m':a.width_m,'raw_parent_only':a.raw_parent_only,'native_admission':False})
    print(json.dumps({'output':str(out),'influences':influences,'status':'loader_preflight_passed_native_pending'}),flush=True)
if __name__=='__main__':main()

