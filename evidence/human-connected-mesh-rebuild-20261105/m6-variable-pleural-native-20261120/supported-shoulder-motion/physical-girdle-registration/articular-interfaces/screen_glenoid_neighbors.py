"""Audit a shared local glenohumeral source frame against the captured whole-body anatomy."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from compare_registration import prepare
from connected_registration import points
from girdle_ik import contact_site
from screen_material_transport import inputs
from bone_overlap import intersections

def run(launch,sites,motion,archive,scene,loader,receipt,clav,out):
    state,banks,clavs,_,pins=inputs(scene,loader,receipt,clav)
    _,cat,ms,items,_,rest,data=prepare(launch,sites,motion,archive)
    originals={};updated={};rows=[]
    for att in items['candidates']:
        sb,hb=att['stable_ids'];sc,hu=data[sb],data[hb]
        src=(contact_site(sc[2],hu[2])+contact_site(hu[2],sc[2]))*.5
        center=(points(sc,src)+points(hu,src))*.5
        common=R.from_rotvec((hu[6]*sc[6].inv()).as_rotvec()*.5)*sc[6]
        scale=(sc[7]+hu[7])*.5
        for k,item in ((sb,sc),(hb,hu)):
            v=item[2];goal=center+common.apply(v-src)*scale;old=points(item,v)
            t=np.clip((np.linalg.norm(v-src,axis=1)-.044)/.075,0,1)
            w=1-((6*t-15)*t+10)*t**3
            q=rest[item[1]];local=R.from_quat(q[3:]).inv().apply(old+w[:,None]*(goal-old)-q[:3]);local[w==0]=ms[k][0][w==0]
            p=state['poses'][cat[k]['body']];pose=R.from_quat(p['quaternion_xyzw'])
            originals[k]=(p['position_m']+pose.apply(ms[k][0])).astype('f4')
            updated[k]=(p['position_m']+pose.apply(local)).astype('f4')
    for s,v in updated.items():
        x=originals[s];f=ms[s][1]
        for (sem,k),(other,g) in banks.items():
            if sem==51004 and (k==s or k in updated and k<s):continue
            o=other;y=other
            if sem==51004 and k in (8,9):o=y=clavs[k][1];g=clavs[k][2]
            if sem==51004 and k in updated:o=originals[k];y=updated[k];g=ms[k][1]
            counts=[];witness=[]
            for a,b in ((x,o),(v,y)):
                pairs=[] if np.any(a.min(0)>b.max(0)) or np.any(b.min(0)>a.max(0)) else intersections(a,f,b,g)
                counts.append(len(pairs));witness.append(pairs[:1])
            if any(counts):rows.append(dict(bone=s,semantic=sem,stable=k,before=counts[0],candidate=counts[1],first=witness))
    out.write_text(json.dumps(dict(inputs=pins,rows=rows),indent=2)+'\n')
if __name__=='__main__':run(*map(Path,sys.argv[1:]))
