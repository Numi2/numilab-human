#!/usr/bin/env python3
"""Compact read-only bad-hit short-edge roster; excludes are counted and sampled."""
from pathlib import Path
from itertools import combinations
import argparse, hashlib, importlib.util, json, sys, time
import numpy as np
HERE=Path(__file__).resolve().parent
WRAPPER=HERE/'endpoint_quotient_state_proposal_002a.py'
WRAPPER_SHA='d3a435495fda8612ad76cbb6a5cf418c38b4a2a8824dcd6f06f6a94440b992b6'
CORE=HERE/'inner_endpoint_quotient_001.py'
CORE_SHA='87f7e980ce80a84a5380cec5c840e924489fa221ab0f435bcd91490c394054a5'
CENSUS004=HERE/'census_bad_short_endpoint_edges_004.py'
CENSUS004_SHA='33bd68879cb2db5fc235220d3eee54d0b568843a4bef7c3d827fc976c86d487c'
def sha(p):return hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
def imp(name,path,expected=None):
 p=Path(path).resolve()
 if expected and sha(p)!=expected:raise ValueError(name+' SHA mismatch')
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def load(path):
 with np.load(path,allow_pickle=False) as z:return {k:np.array(z[k],copy=True) for k in z.files}
def fkey(x):return np.ascontiguousarray(np.asarray(x,np.float32).reshape(3)).view(np.uint32).tobytes()
def _edges(faces):
 d={};nbr={}
 for fi,(a,b,c) in enumerate(np.asarray(faces,np.int64)):
  a,b,c=int(a),int(b),int(c)
  for x,y,w in ((a,b,c),(b,c,a),(c,a,b)):
   e=tuple(sorted((x,y)));d.setdefault(e,[]).append((fi,w));nbr.setdefault(x,set()).add(y);nbr.setdefault(y,set()).add(x)
 if any(len(v)>2 for v in d.values()):raise ValueError('nonmanifold source edge')
 return d,nbr
def _source_to_volume_active(Pmod,source,volume):
 sp=np.asarray(source['points_world_m_f32'],np.float32);sf=np.asarray(source['faces'],np.int64)
 active=np.unique(sf.reshape(-1)).astype(np.int64)
 vp=np.asarray(volume['points_world_m_f32'] if 'points_world_m_f32' in volume else volume['points_world_m'],np.float32)
 lookup={};vol_amb=set()
 for i,x in enumerate(vp):
  k=fkey(x)
  if k in lookup:vol_amb.add(k)
  else:lookup[k]=int(i)
 bykey={};src_amb=set()
 for row in active:
  k=fkey(sp[int(row)])
  if k in bykey:src_amb.add(k)
  else:bykey[k]=int(row)
 reverse=np.full(len(vp),-1,np.int64);unmapped=[]
 for k,row in bykey.items():
  if k in src_amb or k in vol_amb or k not in lookup:
   unmapped.append(row);continue
  vr=int(lookup[k])
  if reverse[vr]>=0:
   src_amb.add(k);reverse[vr]=-1;unmapped.extend((row,int(reverse[vr])))
  else:reverse[vr]=row
 return reverse,{'active_source_points':int(len(active)),'active_source_coordinate_collisions':int(len(src_amb)),
  'volume_coordinate_collisions':int(len(vol_amb)),'active_unmapped_rows':int(len(unmapped))}
def _bad_hits(core,role,source,volumes,max_edge):
 F=np.asarray(source['faces'],np.int64);P=np.asarray(source['points_world_m_f32'],np.float32)
 edge_rows,nbr=_edges(F);interior={e:rows for e,rows in edge_rows.items() if len(rows)==2}
 targets=('Skin','Fat') if role=='I' else ('Fat',)
 hits={};target_detail={};maps={};source_to_volume_by_role={}
 for vr in targets:
  v=volumes[vr];vp=np.asarray(v['points_world_m_f32'] if 'points_world_m_f32' in v else v['points_world_m'],np.float32)
  reverse,mapdoc=_source_to_volume_active(None,source,v)
  src_to_vol=np.full(len(source['points_world_m_f32']),-1,np.int64)
  use=np.flatnonzero(reverse>=0)
  src_to_vol[reverse[use]]=use
  source_to_volume_by_role[vr]=src_to_vol
  t=np.asarray(v['tetrahedra'],np.int64)
  det,q=core.M.measurements(vp,t);bad=np.flatnonzero(~core.M.passes(det,q))
  vh={};unmappedbad=0;source_corner_hist={0:0,1:0,2:0,3:0,4:0}
  for cr in bad:
   sr=reverse[t[int(cr)]]
   mapped=sr[sr>=0]
   source_corner_hist[int(len(mapped))]+=1
   if len(mapped)<2:unmappedbad+=1;continue
   if len(np.unique(mapped))!=len(mapped):unmappedbad+=1;continue
   # Boundary tets commonly have one interior apex. Count every edge whose
   # endpoints are actual active source corners; do not require all four tet
   # vertices to lie on the source surface.
   for a,b in combinations(map(int,mapped),2):
    e=tuple(sorted((a,b)))
    if e not in interior:continue
    L=float(np.linalg.norm(P[e[0]].astype(np.float64)-P[e[1]].astype(np.float64)))
    if L>max_edge:continue
    vh[e]=vh.get(e,0)+1
  maps[vr]=mapdoc
  target_detail[vr]={'native_bad_cells':int(len(bad)),'bad_cells_with_fewer_than_two_active_source_corners':int(unmappedbad),
                     'bad_cell_source_corner_histogram':{str(k):int(v) for k,v in source_corner_hist.items()},
                     'candidate_edges_with_bad_cell_hit':int(len(vh)),'bad_cell_edge_hits':int(sum(vh.values()))}
  for e,n in vh.items():hits[e]=hits.get(e,0)+n
 return interior,nbr,hits,target_detail,maps,targets,source_to_volume_by_role

def roster(core,role,source,volumes,max_edge,immutable_rows):
 interior,nbr,hits,vol_detail,map_detail,targets,source_to_volume=_bad_hits(core,role,source,volumes,max_edge)
 P=np.asarray(source['points_world_m_f32'],np.float32);F=np.asarray(source['faces'],np.int64)
 accepted=[];excluded_count=0;reason_counts={};examples=[];short_count=0;known_edge=None
 facet_edge=(102734,102735) if role=='D' else None
 for e,fr in interior.items():
  L=float(np.linalg.norm(P[e[0]].astype(np.float64)-P[e[1]].astype(np.float64)))
  if L>max_edge:continue
  short_count+=1
  common=(nbr[e[0]]&nbr[e[1]])-{e[0],e[1]}
  opposite={int(x) for fi,_ in interior[e] for x in F[int(fi)] if int(x) not in e}
  link=(common==opposite)
  fixed={x for x in e if int(x) in immutable_rows}
  directions=[{'drop':int(a),'keep':int(b),'eligible_endpoint':int(a) not in immutable_rows} for a,b in (e,e[::-1])]
  eligible=[d for d in directions if d['eligible_endpoint']]
  h=int(hits.get(e,0))
  mapped_by_volume={vr:bool(np.all(source_to_volume[vr][np.asarray(e,np.int64)]>=0)) for vr in targets}
  endpoints_mapped=all(mapped_by_volume.values())
  reasons=[]
  if not link:reasons.append('surface_link_condition_fail')
  if h==0:reasons.append('no_current_bad_tet_edge_hit')
  if not eligible:reasons.append('both_endpoints_are_immutable_rim_or_route')
  if not endpoints_mapped:reasons.append('source_endpoint_missing_from_target_volume')
  row={'role':role,'edge_rows':list(e),'length_m':L,'source_face_rows':[int(x[0]) for x in interior[e]],
       'link_condition_pass':bool(link),'bad_cell_edge_hits':h,'directions':directions,
       'target_volume_endpoint_map':mapped_by_volume,'eligible_directions':eligible,'excluded_reasons':reasons}
  if facet_edge is not None and tuple(e)==facet_edge:known_edge=row
  qualifies=(row['link_condition_pass'] and row['bad_cell_edge_hits']>0 and row['eligible_directions'] and all(row['target_volume_endpoint_map'].values()))
  if qualifies:accepted.append(row)
  else:
   excluded_count+=1
   for reason in reasons:reason_counts[reason]=reason_counts.get(reason,0)+1
   if len(examples)<64:examples.append(row)
 accepted.sort(key=lambda x:(x['length_m'],-x['bad_cell_edge_hits'],x['edge_rows']))
 summary={'all_short_interior_edge_count':short_count,'excluded_short_edge_count':excluded_count,
          'excluded_reason_counts':dict(sorted(reason_counts.items())),
          'excluded_short_edge_examples':examples,'excluded_example_limit':64,
          'known_facet_edge_diagnostic':known_edge}
 return accepted,summary,vol_detail,map_detail,targets

def self_test(core):
 # A valid three-cell edge quotient with an inactive duplicate source row.
 p=np.asarray(((0.,0.,0.),(1.e-7,0.,0.),(0.,1.,0.),(0.,0.,1.),(-1.,-1.,-1.),(1.,-1.,-1.),(0.,0.,0.)),np.float32)
 tbase=np.asarray(((0,1,2,3),(2,0,3,4),(1,2,3,5)),np.int64);tbase=core.M.orient(p[:6],tbase)
 b=core.boundary_faces(tbase);source={'points_world_m_f32':p,'faces':b}
 # Replace one boundary tetra's apex with a volume-interior point while
 # retaining a very short source edge; the roster must still see its two
 # mapped source corners and must ignore the inactive duplicate source row 6.
 vp=np.vstack((p[:6],np.asarray([[.25,.25,1.e-7]],np.float32)))
 t=np.asarray(((0,1,2,6),(2,0,3,4),(1,2,3,5)),np.int64);t=core.M.orient(vp,t)
 volume={'points_world_m_f32':vp,'tetrahedra':t}
 interior,nbr,hits,vd,md,targets,source_to_volume=_bad_hits(core,'D',source,{'Fat':volume},50e-6)
 edge=(0,1)
 assert edge in interior and hits.get(edge,0)>0,(interior,hits)
 assert 6 not in np.unique(source['faces'])
 assert md['Fat']['active_source_coordinate_collisions']==0
 assert vd['Fat']['bad_cell_source_corner_histogram']['3']>=1,(vd,hits)
 # The inactive duplicate must not overwrite active point row 0 in reverse map.
 wrapper=imp('endpoint_state_wrapper_fixture',WRAPPER,WRAPPER_SHA)
 Pmod=imp('endpoint_protection_fixture',wrapper.PROTECTION,wrapper.PROTECTION_SHA)
 rev,_=_source_to_volume_active(Pmod,source,volume)
 assert rev[0]==0,(rev.tolist())
 assert source_to_volume['Fat'][0]>=0 and source_to_volume['Fat'][1]>=0
 return {'inactive_duplicate_does_not_overwrite_active_row':True,'bad_edge_retained':True,
         'bad_hit_count':hits[edge],'source_link_condition':True}

def parity_test(core):
 # Exact fixture shared with census004; accepted rows must be identical while
 # excluded edges are represented by exact counts and a bounded sample.
 old=imp('endpoint_census_004_parity',CENSUS004,CENSUS004_SHA)
 p=np.asarray(((0.,0.,0.),(1.e-7,0.,0.),(0.,1.,0.),(0.,0.,1.),(-1.,-1.,-1.),(1.,-1.,-1.),(0.,0.,0.)),np.float32)
 t=core.M.orient(p[:6],np.asarray(((0,1,2,3),(2,0,3,4),(1,2,3,5)),np.int64))
 source={'points_world_m_f32':p,'faces':core.boundary_faces(t)}
 vp=np.vstack((p[:6],np.asarray([[.25,.25,1.e-7]],np.float32)))
 vt=core.M.orient(vp,np.asarray(((0,1,2,6),(2,0,3,4),(1,2,3,5)),np.int64))
 volume={'points_world_m_f32':vp,'tetrahedra':vt}
 args=(core,'D',source,{'Fat':volume},50e-6,set())
 accepted_old,rows_old,_,_,_=old.roster(*args)
 accepted_new,summary,_,_,_=roster(*args)
 if accepted_new!=accepted_old:raise AssertionError('compact census eligible edges differ from census004')
 if summary['all_short_interior_edge_count']!=len(rows_old):raise AssertionError('compact census short-edge total differs')
 if summary['excluded_short_edge_count']!=len(rows_old)-len(accepted_old):raise AssertionError('compact census exclusion count differs')
 if summary['excluded_short_edge_count'] and not summary['excluded_short_edge_examples']:
  raise AssertionError('compact census lost all exclusion examples')
 if len(summary['excluded_short_edge_examples'])>64:raise AssertionError('compact exclusion sample exceeds cap')
 # Force only the hit ledger empty while retaining the same actual short edge
 # inventory/mapping; exact excluded counts and one bounded witness must stay.
 interior,nbr,_,vd,md,targets,maps=_bad_hits(core,'D',source,{'Fat':volume},50e-6)
 original=_bad_hits
 try:
  globals()['_bad_hits']=lambda *unused:(interior,nbr,{},vd,md,targets,maps)
  excluded,exsum,_,_,_=roster(core,'D',source,{'Fat':volume},50e-6,set())
 finally:globals()['_bad_hits']=original
 if excluded or exsum['all_short_interior_edge_count']!=1 or exsum['excluded_short_edge_count']!=1:
  raise AssertionError('compact exclusion count control failed')
 if exsum['excluded_reason_counts'].get('no_current_bad_tet_edge_hit')!=1 or len(exsum['excluded_short_edge_examples'])!=1:
  raise AssertionError('compact exclusion reason/example control failed')
 return {'eligible_edges_identical_to_004':len(accepted_new),'all_short_edge_count':len(rows_old),
         'excluded_count_exact':summary['excluded_short_edge_count'],
         'excluded_examples_bounded':len(summary['excluded_short_edge_examples']),
         'exact_exclusion_reason_and_example_control':True}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--state',type=Path);ap.add_argument('--role',choices=('I','D'));ap.add_argument('--max-edge-m',type=float,default=50e-6);ap.add_argument('--output',type=Path);ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
 wrapper=imp('endpoint_state_wrapper_005',WRAPPER,WRAPPER_SHA);core=imp('endpoint_core_005',CORE,CORE_SHA)
 if a.self_test:
  print(json.dumps({'status':'PASS_READ_ONLY_COMPACT_SHORT_EDGE_ROSTER_CONTROLS',
      'controls':self_test(core),'census004_parity':parity_test(core)},indent=2));return
 if not(a.state and a.role and a.output):ap.error('--state --role and --output required')
 stpath=a.state.resolve();st=json.loads(stpath.read_text());refs={r:wrapper.check_ref(st['source_surfaces'][r],r+' source') for r in ('O','I','D')}
 source=load(refs[a.role]);targets=('Skin','Fat') if a.role=='I' else ('Fat',)
 volumes={vr:load(wrapper.check_ref(st[vr+'_current'],vr+' volume')) for vr in targets}
 # Derive exact 795 rim/route rows; actual inner-back endpoints are bounded by L<=50um.
 vc={vr:wrapper.point_cache(wrapper.imp('endpoint_protection_004',wrapper.PROTECTION,wrapper.PROTECTION_SHA),volumes[vr]) for vr in targets}
 Pmod=wrapper.imp('endpoint_protection_masks_004',wrapper.PROTECTION,wrapper.PROTECTION_SHA)
 protections=Pmod.protected_scope(stpath,st,a.role,vc,a.output.parent)
 src=protections[targets[-1]]['report']['sources'][a.role]
 immutable=set(map(int,src['rim_rows']))|set(map(int,src['route_rows']))
 accepted,summary,vd,md,targets=roster(core,a.role,source,volumes,a.max_edge_m,immutable)
 # Highlight any queried source edge even when not in eligible list.
 facet_edge=[102734,102735] if a.role=='D' else None
 source_ref=st['source_surfaces'][a.role]
 input_volumes={vr:{'path':str(wrapper.check_ref(st[vr+'_current'],vr+' current volume')),
                     'sha256':str(st[vr+'_current']['sha256'])} for vr in targets}
 report={'schema':'h706.bad-short-endpoint-edge-roster.v3-compact','status':'PASS_READ_ONLY_ACTIVE_ROW_BAD_CELL_CENSUS',
  'physical_admission':False,'native_steps':0,'input_state':{'path':str(stpath),'sha256':sha(stpath)},
  'input_source':{'path':str(refs[a.role]),'sha256':str(source_ref['sha256'])},'input_volumes':input_volumes,
  'algorithm_pins':{'wrapper_path':str(WRAPPER),'wrapper_sha256':WRAPPER_SHA,'core_path':str(CORE),'core_sha256':CORE_SHA,
      'census004_parity_path':str(CENSUS004),'census004_parity_sha256':CENSUS004_SHA},
  'role':a.role,'maximum_edge_m':a.max_edge_m,'target_volumes':vd,'active_coordinate_mapping':md,
  'protected_scope':{'helper_path':str(wrapper.PROTECTION),'helper_sha256':wrapper.PROTECTION_SHA,
      'immutable_rim_route_source_rows':len(immutable),'back_source_rows':len(src['back_rows']),
      'back_endpoints_allowed_with_distance_bound':True},
  'eligible_bad_cell_hit_edges':accepted,'eligible_count':len(accepted),
  'all_short_interior_edge_count':summary['all_short_interior_edge_count'],
  'excluded_count':summary['excluded_short_edge_count'],'excluded_reason_counts':summary['excluded_reason_counts'],
  'excluded_short_edge_examples':summary['excluded_short_edge_examples'],'excluded_example_limit':64,
  'known_facet_edge_diagnostic':summary['known_facet_edge_diagnostic'],
  'full_source_audit':'NOT_RUN','mass_mapping':'PENDING','elapsed_s':time.monotonic()-tic}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':report['status'],'role':a.role,'eligible_count':len(accepted),
  'all_short_interior_edge_count':report['all_short_interior_edge_count'],'excluded_count':report['excluded_count'],
  'excluded_reason_counts':report['excluded_reason_counts'],'known_edge':report['known_facet_edge_diagnostic'],
  'shortest':accepted[:5],'elapsed_s':report['elapsed_s']},indent=2))
if __name__=='__main__':
 tic=time.monotonic();main()
