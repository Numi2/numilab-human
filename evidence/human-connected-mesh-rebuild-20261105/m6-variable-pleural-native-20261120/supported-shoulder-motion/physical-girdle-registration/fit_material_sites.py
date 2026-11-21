"""Fit side-specific material triangles inside named source-landmark regions on the corrected girdle."""
from pathlib import Path
import sys,json,zipfile,hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from compare_registration import prepare
from compare_atlas import vertices_and_faces
from shoulder_inputs import old,digest
from girdle_ik import contact_site

def unit(x):return x/np.linalg.norm(x,axis=-1,keepdims=True)
def run(launch,reference,motion,archive,bones,prior,out):
    cfg,cat,_,reg,poses,rest,data=prepare(launch,reference,motion,archive)
    old.audit.ag.BONES=bones
    _,raw,_,_,_,_,_,cs,_,vo,io=old.audit.ag.read_payloads()
    banks=old.load_local_meshes(raw,cs,vo,io)
    name={p[1]:p[2] for x in (archive.parent/'isa_element_parts.txt').read_text().splitlines() if len(p:=x.split('\t'))==3}
    poses={0:rest,**poses}
    known=json.loads(prior.read_text())
    rows=[];evidence=[];pub=dict(candidate_sha256=digest(bones),rows=rows,regions=evidence)
    def write():out.write_text(json.dumps(pub,indent=2,allow_nan=False)+'\n')
    for pair in reg['candidates']:
        si=pair['side'];side='right' if si else 'left';sc,hm=pair['stable_ids'];cl=8 if si else 9
        sv,cf,hv=[data[u][2] for u in (sc,cl,hm)]
        source_id={}
        with zipfile.ZipFile(archive) as z:
            def muscle(label):
                mid=name[label];buf=z.read(next(p for p in z.namelist() if p.endswith('/'+mid+'.obj')))
                source_id[label]=[mid,hashlib.sha256(buf).hexdigest()];return vertices_and_faces(buf)[0]
            corb=muscle(side+' coracobrachialis')
            bic=muscle('short head of '+side+' biceps brachii')
            supra=muscle(side+' supraspinatus')
            long=muscle('long head of '+side+' biceps brachii')
        def closest_upper(v):
            upper=v[v[:,2]>=v[:,2].max()-.006]
            distances,idx=cKDTree(sv).query(upper);return sv[idx[np.argmin(distances)]]
        tip=(closest_upper(bic)+closest_upper(corb))*.5
        gl=sv[known[si]['atlas']['scapular_vertex']];lt=hv[known[si]['atlas']['humeral_vertex']]
        # Source glenoid and coracoid processes are separated largely along the
        # anterior/posterior atlas axis. Locate the superior face of the base
        # on the path from the coracoid apex toward the scapular body.
        near=sv[(np.linalg.norm(sv-tip,axis=1)<.031)&(np.linalg.norm(sv-tip,axis=1)>.017)]
        direction=unit(gl-tip)
        base_guess=tip+direction*.022
        selected=near[np.linalg.norm(near-base_guess,axis=1)<.013]
        assert len(selected)
        sup=selected[np.argmax(selected[:,2])]
        ac=contact_site(cf,sv); medial=cf[np.argmax(np.linalg.norm(cf-ac,axis=1))]
        axis=unit(medial-ac);frac=(cf-ac)@axis/np.linalg.norm(medial-ac)
        band=cf[(frac>.15)&(frac<.25)];bottom=band[band[:,2]<np.quantile(band[:,2],.18)].mean(axis=0)
        # A greater-tubercle witness is located by the humeral contact of
        # supraspinatus near the bicipital groove.
        dh,ii=cKDTree(hv).query(supra);db=cKDTree(long).query(supra)[0]
        nearby=np.where(dh<max(.003,dh.min()+.001))[0];ix=nearby[np.argmin(db[nearby])];gt=hv[ii[ix]]
        cases=[('superior_glenohumeral_capsuloligament',sc,hm,gl,lt,.009,.009,
            'Reference bony proxy in superior glenoid rim and lesser tubercle regions located by the source long-head biceps and subscapularis. Kask PMID 20434927. Inferred effective fiber.'),
            ('coracohumeral_capsuloligament',sc,hm,base_guess,gt,.011,.010,
            'Source coracoid base region and supraspinatus greater tubercle neighborhood. Inferred coracohumeral fiber.'),
            ('coracoclavicular_complex',sc,cl,sup,bottom,.010,.009,
            'Superior coracoid base and inferior distal clavicle region. Inferred effective fiber for the combined complex. PMID 20440501 and 23674218.')]
        for label,a,b,g0,g1,r0,r1,description in cases:
            options=[]
            for sid,seed,radius in [(a,g0,r0),(b,g1,r1)]:
                original=data[sid][2];v,f,ns,*_=banks[sid]
                pts=original[f].mean(axis=1);dist=np.linalg.norm(pts-seed,axis=1)
                ids=np.flatnonzero(dist<=radius)
                options.append((ids,dist[ids],v[f[ids]].mean(1),unit(ns[f[ids]].mean(1)),unit(np.cross(v[f[ids,1]]-v[f[ids,0]],v[f[ids,2]]-v[f[ids,0]]))))
            (ii,da,p0,n0,fn0),(jj,db,p1,n1,fn1)=options
            evidence_row=dict(side=si,name=label,candidates=[len(ii),len(jj)],source_members=source_id,
                              centers_atlas_m=[g0.tolist(),g1.tolist()],bounds_m=[r0,r1])
            if not len(ii) or not len(jj):
                evidence.append(evidence_row);print(label,si,'empty',flush=True);write();continue
            # Retain anatomically close pairs with outward-facing facets and
            # interpolated normals in both articulated rest and observed poses.
            qualified=np.ones((len(ii),len(jj)),bool)
            for k in [0,-1,-2,-3,32,256,4000,8000]:
                if k not in poses:continue
                p=poses[k];pa,pb=p[cat[a]['body']],p[cat[b]['body']]
                ra,rb=R.from_quat(pa[3:]),R.from_quat(pb[3:]);first=ra.apply(p0)+pa[:3];second=rb.apply(p1)+pb[:3]
                vec=second[None,:,:]-first[:,None,:]
                for normal,sgn,ax,rr in [(n0,1,0,ra),(fn0,1,0,ra),(n1,-1,1,rb),(fn1,-1,1,rb)]:
                    dot=np.einsum('ijk,ik->ij' if ax==0 else 'ijk,jk->ij',vec,rr.apply(normal))*sgn
                    qualified&=dot>.00005
            ix,jx=np.nonzero(qualified)
            score=da[ix]**2+db[jx]**2
            ranking=np.argsort(score,kind='stable'); evidence_row['facing_pairs']=len(ix)
            segments={}
            def clear(k,i,j):
                # Test every nearby bank, including both endpoint banks, with
                # the force owner's open-segment convention.
                p=poses[k];xs=[]
                for sid,pt in ((a,p0[i]),(b,p1[j])):
                    bp=p[cat[sid]['body']];xs.append(R.from_quat(bp[3:]).apply(pt)+bp[:3])
                if k not in segments:
                    nearby={}
                    for sid,(vs,fs,*_) in banks.items():
                        bp=p[cat[sid]['body']];rr=R.from_quat(bp[3:]);tri=rr.apply(vs)+bp[:3]
                        if np.any(tri.max(0)<np.minimum(*xs)-.04) or np.any(tri.min(0)>np.maximum(*xs)+.04):continue
                        tri=tri[fs];nearby[sid]=(tri,tri.min(1),tri.max(1))
                    segments[k]=nearby
                for sid,(tri,lower,upper) in segments[k].items():
                    sel=np.all((lower<=np.maximum(*xs)+1e-9)&(upper>=np.minimum(*xs)-1e-9),axis=1)
                    if not np.any(sel):continue
                    ts=tri[sel].reshape(-1,3); faces=np.arange(len(ts)).reshape(-1,3)
                    if old.audit.lp.vector_hits(*xs,(ts,faces)):return False
                return True
            for count,rank in enumerate(ranking[:3000]):
                i,j=ix[rank],jx[rank]
                if all(clear(k,i,j) for k in (0,-1,-2,-3,32,8000)):
                    row=dict(name=label,side=si,stable_ids=[a,b],source=description,
                          attachments=[dict(triangle=int(u),barycentric=[1/3]*3) for u in (ii[i],jj[j])])
                    rows.append(row);evidence_row.update(candidate_rank=count,distances_to_seeds_m=[float(da[i]),float(db[j])]);break
            print(label,si,evidence_row.get('candidate_rank'),[len(ii),len(jj),len(ix)],flush=True)
            evidence.append(evidence_row);write()
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
