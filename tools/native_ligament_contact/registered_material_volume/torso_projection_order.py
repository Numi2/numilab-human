"""Linear inequalities preserving the native ordering on a projected triangle overlap."""
import numpy as np
from scipy import sparse

def section_order(s,x,files):
    groups={}
    for p in files:
        with np.load(p) as hits:
            for k in hits.files:
                a,b=k.split('_')
                if b.isdigit():continue
                for i,j in hits[k]:
                    key=(a,int(i),b,int(j));groups[key]=groups.get(key,0)+1
    n=len(x);rr=[];cc=[];vv=[];rhs=[]
    def add(nodes,weight,value):
        j=len(rhs);rr.extend([j]*len(nodes));cc.extend(nodes);vv.extend(weight);rhs.append(value)
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    def bary(p,t):
        e=t[1]-t[0];f=t[2]-t[0];v=cross(p-t[0],f)/cross(e,f);w=cross(e,p-t[0])/cross(e,f);return np.array([1-v-w,v,w])
    for (ga,i,gb,j),repeats in sorted(groups.items()):
        ia,ib=s[ga][i],s[gb][j];a,b=x[ia],x[ib];aa,bb=a[:,:2],b[:,:2]
        tiny=min(abs(cross(q[1]-q[0],q[2]-q[0])) for q in (aa,bb))<1e-18
        pairs=[]
        if not tiny and repeats<2:
            pts=list(aa)+list(bb)
            for k in range(3):
                e=aa[(k+1)%3]-aa[k]
                for l in range(3):
                    f=bb[(l+1)%3]-bb[l];det=cross(e,f)
                    if abs(det)<1e-20:continue
                    v=bb[l]-aa[k];u=cross(v,f)/det;w=cross(v,e)/det
                    if -1e-8<=u<=1+1e-8 and -1e-8<=w<=1+1e-8:pts.append(aa[k]+u*e)
            for point in pts:
                wa,wb=bary(point,aa),bary(point,bb)
                if min(wa.min(),wb.min())< -2e-8:continue
                wa=np.clip(wa,0,1);wa/=wa.sum();wb=np.clip(wb,0,1);wb/=wb.sum()
                pairs.append((wa,wb,float(wa@a[:,2]-wb@b[:,2])))
        if pairs and max(abs(d) for _,_,d in pairs)>1e-10:
            sign=np.sign(max(pairs,key=lambda e:abs(e[2]))[2])
            for wa,wb,d in pairs:
                add(np.r_[ia,ib],np.r_[-sign*wa,sign*wb],max(0.,sign*d*1000*.70-.00005))
        else:
            # A common translation preserves vertical or shared degenerate
            # projections without choosing an arbitrary side.
            nodes=np.unique(np.r_[ia,ib])
            for j in nodes[1:]:
                add([nodes[0],j],[-1,1],0.);add([nodes[0],j],[1,-1],0.)
    A=sparse.coo_matrix((vv,(rr,cc)),shape=(len(rhs),n)).tocsr();A.eliminate_zeros()
    return A,np.array(rhs),len(groups)
