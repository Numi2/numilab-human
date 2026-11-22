"""Preserve previously clear cortical geometry throughout an upward reference registration."""
import json,runpy,sys,subprocess,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from emit_skin_source_registration import sha
from audit_torso_plc import Surface
src,impl,pack,pose,baseline,out=map(Path,sys.argv[1:7]);ids=set(map(int,sys.argv[7:]))
s=np.load(src);x=s['x'].astype('f8');f=s['O'];tri=x[f]
b=json.loads(pose.read_text())['all_bodies'];poses={r['body_index']:{'position_m':r['current']['position_world_m'],'quaternion_xyzw':r['current']['orientation_world_xyzw']} for r in b}
ref=json.loads(baseline.read_text());assert sha(pose)==ref['pose_sha256']
raw=subprocess.check_output(['/opt/homebrew/bin/zstd','-qdc',str(pack)]);assert hashlib.sha256(raw).hexdigest()==ref['pack_sha256']
sys.path.insert(0,str(impl.parent/'src'));api=runpy.run_path(str(impl));data={};corners=[];weight=[];gap=[];bankrows=[];seen=set()
def cr(a,b):return a[0]*b[1]-a[1]*b[0]
def bary(p,q):
    m=q[1]-q[0];n=q[2]-q[0];d=cr(m,n)
    if abs(d)<1e-20:return np.array([np.nan]*3)
    b=cr(p-q[0],n)/d;c=cr(m,p-q[0])/d;return np.array([1-b-c,b,c])
for (kind,bank),i,bx,bf in api['primitives'](raw,poses):
    if kind!=51004 or bank not in ids:continue
    seen.add(bank);bt=bx[bf].astype('f8');blo=bt.min(1);bhi=bt.max(1);ctr=(blo+bhi)/2;rad=((bhi-blo)/2)[:,:2].max(1);tree=cKDTree(ctr[:,:2]);mx=rad.max()
    for n,a in enumerate(tri):
        lo=a.min(0);hi=a.max(0);center=(lo+hi)/2;r=max((hi[:2]-lo[:2])/2)
        cand=np.asarray(tree.query_ball_point(center[:2],r+mx+1e-12,p=np.inf),dtype=int)
        cand=cand[np.all(bhi[cand,:2]>=lo[:2],1)&np.all(blo[cand,:2]<=hi[:2],1)&(bhi[cand,2]>=lo[2])&(blo[cand,2]<=hi[2]+.032)]
        for j in cand:
            t=bt[j];aa=a[:,:2];bb=t[:,:2];pts=list(aa)+list(bb)
            for k in range(3):
                e=aa[(k+1)%3]-aa[k]
                for l in range(3):
                    ee=bb[(l+1)%3]-bb[l];det=cr(e,ee)
                    if abs(det)<1e-20:continue
                    v=bb[l]-aa[k];u=cr(v,ee)/det;z=cr(v,e)/det
                    if -1e-9<=u<=1+1e-9 and -1e-9<=z<=1+1e-9:pts.append(aa[k]+u*e)
            for pt in pts:
                wa,wb=bary(pt,aa),bary(pt,bb)
                if not np.isfinite(wa).all() or not np.isfinite(wb).all() or min(wa.min(),wb.min())< -1e-8:continue
                dz=float(wb@t[:,2]-wa@a[:,2])
                if dz<0 or dz>.032:continue
                wa=np.clip(wa,0,1);wa/=wa.sum()
                corners.append(f[n]);weight.append(wa);gap.append(max(0.,dz*1000-.0001));bankrows.append([bank,n,int(j)])
assert seen==ids
np.savez_compressed(out,corners=np.array(corners,dtype='i4').reshape(-1,3),weight=np.array(weight).reshape(-1,3),max_delta_mm=gap,bank_rows=np.array(bankrows,dtype='i4'))
print('banks',sorted(seen),'projected upper bounds',len(gap),flush=True)
