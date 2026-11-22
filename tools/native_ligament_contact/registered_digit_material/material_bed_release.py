"""Existing stationary nodal bed barrier, included only in the offline loaded registration."""
import numpy as np

def stationary_bed(gaps,masses,dt,width):
    gaps=np.asarray(gaps,dtype=float)
    supported=gaps<width
    if np.any(gaps<=1e-4*width):raise ValueError('Registration touched the contact singularity')
    out=np.zeros((3,len(gaps)))
    d=gaps[supported];s=d*d;sh=width*width;off=s-sh
    k=10./np.maximum(dt*dt*sh/masses[supported],1.e-20)
    log=np.log(s/sh);u=2*off*log+off*off/s
    out[0,supported]=-k*off*off*log
    out[1,supported]=2*d*k*u
    out[2,supported]=-2*k*(u+2*s*(2*log+4*off/s-off*off/s**2))
    return out


def balance_emitted_mass(before,rounded,weights,limits,fixed):
    """Recover donor and remesh first moments together on the Float32 lattice."""
    from scipy.optimize import Bounds,LinearConstraint,milp
    out=rounded.copy()
    W=np.array(weights,dtype=float);tol=np.array(limits)
    for axis in range(3):
        drift=W@(out[:,axis]-before[:,axis])
        if np.all(abs(drift)<=tol):continue
        step=np.abs(np.spacing(out[:,axis].astype('f4')).astype(float))
        active=np.flatnonzero(~fixed[:,axis] & (np.max(abs(W)*step/tol[:,None],axis=0)>1e-10))
        n=len(active)
        if not n:raise ValueError('No eligible mass lattice correction')
        A=W[:,active]*step[active]/tol[:,None]
        bounds=Bounds(np.r_[-16*np.ones(n),np.zeros(n)],16*np.ones(2*n))
        C=np.block([[np.eye(n),-np.eye(n)],[-np.eye(n),-np.eye(n)]])
        r=milp(np.r_[np.zeros(n),step[active]/step[active].max()],
               integrality=np.r_[np.ones(n),np.zeros(n)],bounds=bounds,
               constraints=[LinearConstraint(np.c_[A,np.zeros_like(A)],-.9-drift/tol,.9-drift/tol),LinearConstraint(C,-np.inf,0)],
               options={'time_limit':12})
        if r.x is None:raise ValueError('No feasible joint donor and cavity rounding')
        out[active,axis]=(out[active,axis]+np.rint(r.x[:n])*step[active]).astype('f4')
    if np.any(abs(W@(out-before))>tol[:,None]):raise ValueError('Emitted first moments exceed limits')
    return out


def bind_contact_footprint(case,outer,points,minimum,width):
    """Pin the active physical node identities for the emitted pose in the native loader."""
    import re
    outer=np.asarray(outer,dtype=int)
    if len(outer)!=len(np.unique(outer)):raise ValueError('Duplicate outer skin ID')
    gaps=points[outer,2]
    nodes=np.sort(outer[gaps<width])
    if not len(nodes) or not np.isfinite(gaps).all() or min(gaps)!=minimum or min(points[:,2])!=minimum:
        raise ValueError('Changed first support gap')
    data=case/'NumiHumanLoadedSkinPoseData.inc'
    text=data.read_text()
    declaration='constexpr unsigned registeredBedContactNodes[]={' + ','.join(str(i)+'u' for i in nodes) + '};'
    pattern=r'constexpr unsigned registeredBedContactNodes\[\]=\{[0-9u,]+\};'
    if 'registeredBedContactNodes' in text:
        text,n=re.subn(pattern,declaration,text)
        if n!=1:raise ValueError('Unknown previous contact footprint table')
    else:
        text+='\nnamespace numiLoadedData { '+declaration+' }\n'
    data.write_text(text)
    path=case/'NumiHumanLoadedSkinPose.hpp';source=path.read_text()
    check='registered bed contact identities differ from emitted skin'
    if check not in source:
        guard=re.search(r'    size_t touch=0;.*?    require\(touch==[0-9]+,"no registered physical skin point touches the bed"\);',source,re.S)
        if not guard or guard.group(0).count('bundle.boundaryMap.outer.size()')!=1:
            raise ValueError('Unknown material support checker')
        replacement="""    require(bundle.boundaryMap.outer.size()==numiLoadedData::outerNodes,"registered outer boundary row count changed");
    std::vector<unsigned> occupied;
    for (const auto node : bundle.boundaryMap.outer) {
        const double gap=p.at(node)[2];
        require(std::isfinite(gap) && gap>0.0,"invalid registered outer bed gap");
        if (gap<WIDTH) occupied.push_back(node);
    }
    std::sort(occupied.begin(),occupied.end());
    const auto& authored=numiLoadedData::registeredBedContactNodes;
    const std::size_t touch=occupied.size();
    require(touch==std::size(authored) && std::equal(occupied.begin(),occupied.end(),std::begin(authored)),
            "registered bed contact identities differ from emitted skin");""".replace("WIDTH",float(width).hex())
        path.write_text(source[:guard.start()]+replacement+source[guard.end():])
    elif source.count(check)!=1:
        raise ValueError('Ambiguous physical support checker')
    return nodes.tolist()
