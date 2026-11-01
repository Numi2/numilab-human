#!/usr/bin/env python3
"""Read-only O->Skin short-edge census with Skin-only authenticated protection 008."""
from pathlib import Path
from collections import Counter,defaultdict
from itertools import combinations
import argparse,hashlib,importlib.util,json,sys,time
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
CORE=HERE.parent/'quality-o-endpoint-quotient-007-nativepreflight'/'outer_endpoint_quotient_007.py'
CORE_SHA='6d7cec119eb6e2e752f59c015ebce53a12575c878d5b20f511bc628ff351d27f'
PROTECT_ADAPTER=HERE/'o_skin_only_protection_010.py'
PROTECT_ADAPTER_SHA='5422ec971a193f317290f03114f72a11ee822fa8dbebceb872a1b6a3ef5676f3'
BATCH005=H/'quality-endpoint-collapse-001'/'batch_inner_endpoint_quotient_005.py'
BATCH005_SHA='eb57b4ad8a91feec2ed84ca7cf483093ec2b49f367bb1ff55977f927a4997619'
MAX_ORDINARY_HAUSDORFF_M=500e-6

def json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError('cannot serialize '+type(value).__name__)
def json_dumps(value,**kwargs):
    return json.dumps(value,default=json_default,**kwargs)
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def imp(name,path,expected=None):
    path=Path(path).resolve()
    if expected and sha(path)!=expected:raise ValueError(name+' SHA mismatch')
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader:raise RuntimeError('cannot load '+str(path))
    m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def _point_maps(source,skin):
    sp=np.asarray(source['points_world_m_f32'],np.float32)
    sf=np.asarray(source['faces'],np.int64)
    vp=np.asarray(skin.get('points_world_m_f32',skin['points_world_m']),np.float32)
    active=np.unique(sf.reshape(-1)).astype(np.int64)
    if sp.ndim!=2 or sp.shape[1]!=3 or sf.ndim!=2 or sf.shape[1]!=3 or vp.ndim!=2 or vp.shape[1]!=3:
        raise ValueError('O/Skin active point map arrays malformed')
    vlookup={};vamb=set()
    for i,p in enumerate(vp):
        k=COREMOD.CORE.f32key(p)
        if k in vlookup:vamb.add(k)
        else:vlookup[k]=int(i)
    maps=np.full(len(sp),-1,np.int64);reverse=np.full(len(vp),-1,np.int64);unmapped=[]
    source_keys={}
    for r in active:
        k=COREMOD.CORE.f32key(sp[int(r)])
        if k in source_keys:raise ValueError('active O source coordinates collide')
        source_keys[k]=int(r)
        if k in vamb or k not in vlookup:unmapped.append(int(r));continue
        vr=vlookup[k]
        if reverse[vr]>=0:raise ValueError('O active point map is not injective')
        maps[int(r)]=vr;reverse[vr]=int(r)
    if unmapped:raise ValueError('active O points lack unique exact Skin F32 mappings: '+str(unmapped[:8]))
    return maps,reverse,{'active_O_source_point_count':len(active),'active_O_unmapped_point_count':len(unmapped),
        'Skin_duplicate_coordinate_key_count':len(vamb)}

def _surface_edges(faces):
    edge_faces=defaultdict(list);neighbors=defaultdict(set)
    for fi,(a0,b0,c0) in enumerate(np.asarray(faces,np.int64)):
        a,b,c=map(int,(a0,b0,c0))
        for x,y in ((a,b),(b,c),(c,a)):
            e=tuple(sorted((x,y)));edge_faces[e].append(int(fi));neighbors[x].add(y);neighbors[y].add(x)
    if any(len(rows)>2 for rows in edge_faces.values()):raise ValueError('O source surface has nonmanifold edge')
    return edge_faces,neighbors

def build_roster(source,skin,*,protected_source_rows=(),protected_volume_rows=(),
                 max_ordinary_hausdorff_m=MAX_ORDINARY_HAUSDORFF_M):
    if max_ordinary_hausdorff_m<=0 or max_ordinary_hausdorff_m>MAX_ORDINARY_HAUSDORFF_M:
        raise ValueError('O ordinary Hausdorff cap exceeds fixed 0.5 mm policy')
    sp=np.asarray(source['points_world_m_f32'],np.float32);sf=np.asarray(source['faces'],np.int64)
    vp=np.asarray(skin.get('points_world_m_f32',skin['points_world_m']),np.float32)
    t=np.asarray(skin['tetrahedra'],np.int64)
    prot_src=set(map(int,np.asarray(protected_source_rows,np.int64).reshape(-1)))
    prot_vol=set(map(int,np.asarray(protected_volume_rows,np.int64).reshape(-1)))
    if prot_src and (min(prot_src)<0 or max(prot_src)>=len(sp)):raise ValueError('O protected source row outside domain')
    if prot_vol and (min(prot_vol)<0 or max(prot_vol)>=len(vp)):raise ValueError('O protected Skin row outside domain')
    maps,reverse,map_stats=_point_maps(source,skin)
    edge_faces,neighbors=_surface_edges(sf)
    interior={e:rows for e,rows in edge_faces.items() if len(rows)==2}
    det,q=COREMOD.CORE.M.measurements(np.asarray(skin['points_world_m'],np.float32),t)
    bad=np.flatnonzero(~COREMOD.CORE.M.passes(det,q))
    hits=Counter();bad_unmapped=0
    for cr in bad:
        sr=reverse[t[int(cr)]]
        mapped=sr[sr>=0]
        if len(mapped)<2 or len(np.unique(mapped))!=len(mapped):bad_unmapped+=1;continue
        for a,b in combinations(map(int,mapped),2):
            e=tuple(sorted((a,b)))
            if e not in interior:continue
            length=float(np.linalg.norm(sp[e[0]].astype(np.float64)-sp[e[1]].astype(np.float64)))
            if length<=max_ordinary_hausdorff_m:hits[e]+=1
    allrows=[];accepted=[]
    for e,fr in interior.items():
        length=float(np.linalg.norm(sp[e[0]].astype(np.float64)-sp[e[1]].astype(np.float64)))
        if length>max_ordinary_hausdorff_m:continue
        common=(neighbors[e[0]]&neighbors[e[1]])-{e[0],e[1]}
        opposite={int(x) for fi in fr for x in sf[int(fi)] if int(x) not in e}
        link=(common==opposite)
        star_rows=np.flatnonzero(np.any(np.isin(sf,np.asarray(e,np.int64)),axis=1))
        star_vertices=set(map(int,np.unique(sf[star_rows])))
        mapped_e=maps[np.asarray(e,np.int64)]
        if np.all(mapped_e>=0):
            volume_link_ok,volume_link=COREMOD.volume_edge_link_condition(t,int(mapped_e[0]),int(mapped_e[1]))
            vol_star_rows=np.flatnonzero(np.any(np.isin(t,mapped_e),axis=1))
        else:
            volume_link_ok=False;volume_link={'edge_incident_tets':0,'missing_edge_link_simplices':[]}
            vol_star_rows=np.empty(0,np.int64)
        vol_star_vertices=set(map(int,np.unique(t[vol_star_rows]))) if len(vol_star_rows) else set()
        source_conflicts=sorted(star_vertices&prot_src)
        volume_conflicts=sorted(vol_star_vertices&prot_vol)
        h=int(hits.get(e,0));reasons=[]
        if not link:reasons.append('surface_link_condition_fail')
        if not volume_link_ok:reasons.append('volume_3d_link_condition_fail')
        if h<=0:reasons.append('no_current_bad_tet_edge_hit')
        if source_conflicts:reasons.append('protected_O_back_rim_route_source_star')
        if volume_conflicts:reasons.append('protected_O_back_rim_route_Skin_star')
        if not np.all(mapped_e>=0):reasons.append('source_endpoint_missing_from_Skin')
        dirs=[{'drop':int(a),'keep':int(b),'eligible_endpoint':not(bool(set((int(a),int(b)))&prot_src))}
              for a,b in (e,e[::-1])]
        eligible=[d for d in dirs if d['eligible_endpoint']]
        if not eligible:reasons.append('protected_O_back_rim_route_endpoint')
        row={'role':'O','edge_rows':list(e),'length_m':length,
             'ordinary_surface_hausdorff_upper_bound_m':length,
             'source_face_rows':[int(x) for x in fr],'source_star_rows':star_rows.tolist(),
             'source_star_vertices':sorted(star_vertices),'link_condition_pass':bool(link),
             'volume_3d_link_condition_pass':bool(volume_link_ok),'volume_link_witness':volume_link,
             'bad_cell_edge_hits':h,'eligible_directions':eligible,'directions':dirs,
             'source_endpoint_map_to_Skin':mapped_e.tolist(),
             'source_star_protected_conflicts':source_conflicts,'Skin_star_protected_conflicts':volume_conflicts,
             'excluded_reasons':reasons}
        allrows.append(row)
        if link and volume_link_ok and h>0 and eligible and not source_conflicts and not volume_conflicts and np.all(mapped_e>=0):
            accepted.append(row)
    allrows.sort(key=lambda x:(x['length_m'],-x['bad_cell_edge_hits'],x['edge_rows']))
    accepted.sort(key=lambda x:(x['length_m'],-x['bad_cell_edge_hits'],x['edge_rows']))
    return {'status':'PASS_READ_ONLY_ACTIVE_ROW_BAD_CELL_CENSUS','schema':'h706.bad-short-O-endpoint-edge-roster.v1',
        'role':'O','target_volumes':{'Skin':{'native_bad_cells':int(len(bad)),
          'bad_cells_with_fewer_than_two_active_source_corners':int(bad_unmapped)}},
        'active_coordinate_mapping':map_stats,'maximum_edge_m':float(max_ordinary_hausdorff_m),
        'protected_scope':{'source_rows':len(prot_src),'Skin_volume_rows':len(prot_vol),
          'O_back_rim_route_endpoint_and_star_exclusion':True},
        'eligible_bad_cell_hit_edges':accepted,'eligible_count':len(accepted),
        'excluded_short_edges':allrows,'excluded_count':len(allrows)-len(accepted),
        'full_source_audit':'NOT_RUN','mass_mapping':'PENDING','physical_admission':False,'native_steps':0}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--state',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--max-edge-m',type=float,default=MAX_ORDINARY_HAUSDORFF_M)
    a=ap.parse_args();tic=time.monotonic()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    global COREMOD
    COREMOD=imp('outer_endpoint_core007_census',CORE,CORE_SHA)
    b=imp('frozen_batch005_for_O_census',BATCH005,BATCH005_SHA);b.core=COREMOD.CORE
    pm=imp('o_skin_only_protection_adapter010_census',PROTECT_ADAPTER,PROTECT_ADAPTER_SHA)
    stpath=a.state.resolve();st=json.loads(stpath.read_text())
    state,surfaces,srefs,volumes,vrefs=b.load_state_inputs(stpath)
    prot_dir=a.output.parent/'protection'
    fixed,back,protection,_=pm.protected_rows(stpath,state,'O',surfaces,volumes,prot_dir,COREMOD)
    oprot=protection['Skin']['report'].get('sources',{}).get('O')
    if not isinstance(oprot,dict):raise ValueError('855 lineage-aware protection owner did not provide O rows')
    psrc=np.unique(np.r_[oprot['back_rows'],oprot['rim_rows'],oprot['route_rows']]).astype(np.int64)
    pmap=COREMOD._exact_rows_to_volume(surfaces['O']['points_world_m_f32'],
       np.asarray(volumes['Skin'].get('points_world_m_f32',volumes['Skin']['points_world_m']),np.float32),psrc)
    report=build_roster(surfaces['O'],volumes['Skin'],protected_source_rows=psrc,
       protected_volume_rows=list(pmap.values()),max_ordinary_hausdorff_m=a.max_edge_m)
    report['input_state']={'path':str(stpath),'sha256':sha(stpath)}
    report['protected_scope'].update({'helper_path':str(b.PROTECTION),'helper_sha256':b.PROTECTION_SHA,
       'scope_adapter_path':str(PROTECT_ADAPTER),'scope_adapter_sha256':sha(PROTECT_ADAPTER),
       'scope_mode':'O_ONLY_SKIN_VOLUME',
       'protection_owner_sha256':b.PROTECTION_OWNER_SHA,'O_back_rows':len(oprot['back_rows']),
       'O_rim_rows':len(oprot['rim_rows']),'O_route_rows':len(oprot['route_rows'])})
    report['producer_sha256']=sha(Path(__file__))
    report['elapsed_s']=time.monotonic()-tic
    a.output.write_text(json_dumps(report,indent=2,sort_keys=True)+'\n')
    print(json_dumps({'status':report['status'],'role':'O','eligible_count':report['eligible_count'],
       'excluded_count':report['excluded_count'],'skin_bad_cells':report['target_volumes']['Skin']['native_bad_cells'],
       'elapsed_s':report['elapsed_s']},indent=2))

if __name__=='__main__':main()
