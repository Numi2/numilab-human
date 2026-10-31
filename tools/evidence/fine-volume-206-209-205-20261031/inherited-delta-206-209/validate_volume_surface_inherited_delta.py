#!/usr/bin/env python3
"""Source-only inherited-delta validator for 206/209 quotient volume outputs."""
import argparse, hashlib, importlib, json, pathlib, sys, time
import numpy as np

H=pathlib.Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
SHELLS={
 'Skin':H/'complete-typed-shell-owner-074/skin-shell.npz',
 'Fat':H/'fat-inferred-boundary-shell-assembly-140-v2/fat-shell-refined-typed.npz',
}
DP_REPORT=H/'fixed-boundary-quality-topology-196/dp-batch-004/report.json'
COLLAPSE_REPORT=H/'fixed-boundary-quality-topology-196/collapse-batch-002/report.json'
SCOPE713=H/'publication-source-177-001/receipts/geometry-scope-713.json'
OWNER=pathlib.Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/deep-boundary-checkpoint-349/payload/exact-owner/numilab_human/cardiac_cavity_intersections.py')
ROLE_LETTERS={'Skin':('O','I'),'Fat':('I','D')}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def atomic_json(path,obj):
    path=pathlib.Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n'); tmp.replace(path)

def tri_keys(points,faces):
    f=np.asarray(faces,dtype=np.int64)
    p=np.asarray(points,dtype=np.float32)
    tri=np.ascontiguousarray(p[f])
    bits=tri.view(np.uint32).reshape((-1,3,3)).copy()
    bits[bits==0]=0
    order=np.lexsort((bits[:,:,2],bits[:,:,1],bits[:,:,0]),axis=1)
    canon=np.take_along_axis(bits,order[:,:,None],axis=1)
    return np.ascontiguousarray(canon.reshape((-1,9))).view('V36').reshape(-1),order

def point_keys(points):
    p=np.ascontiguousarray(np.asarray(points,dtype=np.float32)).copy(); p[p==0]=0
    return p.view('V12').reshape(-1)

def unique_index(keys,label):
    uniq,first=np.unique(keys,return_index=True)
    if len(uniq)!=len(keys): raise ValueError(f'{label} has duplicate exact triangle keys: {len(keys)-len(uniq)}')
    return uniq,first

def contains(query,reference,label):
    uq,_=unique_index(reference,label+' reference')
    ix=np.searchsorted(uq,query)
    ok=ix<len(uq)
    hit=np.zeros(len(query),dtype=bool)
    hit[ok]=uq[ix[ok]]==query[ok]
    return hit

def lookup_rows(query,reference,label):
    uq,first=unique_index(reference,label+' reference')
    ix=np.searchsorted(uq,query)
    ok=ix<len(uq)
    hit=np.zeros(len(query),dtype=bool)
    hit[ok]=uq[ix[ok]]==query[ok]
    rows=np.full(len(query),-1,dtype=np.int64)
    rows[hit]=first[ix[hit]]
    return hit,rows

def tet_boundary(tets):
    t=np.asarray(tets,dtype=np.int64)
    # Outward-oriented faces for consistently positive tetrahedra; orientation parity is
    # compared between the shared I shells below, while incidence defines the boundary.
    f=np.concatenate((t[:,[1,2,3]],t[:,[0,3,2]],t[:,[0,1,3]],t[:,[0,2,1]]),axis=0)
    sk=np.sort(f,axis=1)
    order=np.lexsort((sk[:,2],sk[:,1],sk[:,0]))
    ss=sk[order]
    start=np.r_[0,1+np.flatnonzero(np.any(ss[1:]!=ss[:-1],axis=1))]
    end=np.r_[start[1:],len(ss)]
    count=end-start
    b=order[start[count==1]]
    nonmanifold=int(np.count_nonzero(count>2))
    return f[b], {'boundary_faces':int(len(b)),'interior_face_keys':int(np.count_nonzero(count==2)),'nonmanifold_face_keys':nonmanifold,'unpaired_face_keys':int(np.count_nonzero(count==1))}

def role_shell_faces(z,letter):
    if letter=='O': return z['faces'][z['registered_O_shell_face_rows']]
    if letter=='I': return z['registered_I_shell_faces']
    if letter=='D': return z['faces'][z['registered_D_shell_face_rows']]
    raise ValueError(letter)

def owner_module():
    sys.path.insert(0,str(OWNER.parent.parent))
    return importlib.import_module('numilab_human.cardiac_cavity_intersections')

def run(volume_report_path,outdir):
    started=time.time(); volume_report_path=pathlib.Path(volume_report_path).resolve(); outdir=pathlib.Path(outdir).resolve()
    if outdir.exists(): raise FileExistsError(f'output directory exists: {outdir}')
    outdir.mkdir(parents=True)
    rep=json.loads(volume_report_path.read_text())
    if rep.get('physical_admission') is not False or int(rep.get('native_steps',-1))!=0:
        raise ValueError('volume report is not source-only/non-admitted')
    roles={r['role']:r for r in rep.get('roles',[]) if isinstance(r,dict) and 'role' in r}
    if set(roles)!=set(('Skin','Fat')): raise ValueError('volume report must contain Skin and Fat outputs')
    dp=json.loads(DP_REPORT.read_text()); collapse=json.loads(COLLAPSE_REPORT.read_text()); scope=json.loads(SCOPE713.read_text())
    if collapse.get('status')!='PASS_BOUNDED_EXACT_SHORT_EDGE_COLLAPSE_PROPOSAL': raise ValueError('collapse002 status/pin invalid')
    if int(collapse.get('joint_changed_vs_complete_OID_walls_all185',{}).get('pair_count',-1))!=0: raise ValueError('collapse002 exact joint delta was not zero')
    if int(collapse.get('joint_changed_vs_complete_OID_walls_all185',{}).get('new_face_self_pairs',-1))!=0: raise ValueError('collapse002 self-delta was not zero')
    if scope.get('status')!='PASS_GEOMETRY_SCOPE_INHERITED_PLUS_EXACT_DELTAS': raise ValueError('713 inherited scope not passing')
    if scope.get('physical_admission') is not False or int(scope.get('runtime',-1))!=0: raise ValueError('713 scope is not source-only')
    coverage=[]
    for item in scope.get('coverage_chain',[]):
        cp=pathlib.Path(item['path'])
        if sha(cp)!=item.get('sha256'): raise ValueError(f"713 coverage-chain pin mismatch: {item.get('name')}")
        cd=json.loads(cp.read_text()); name=item['name']
        entry={'name':name,'path':str(cp),'sha256':sha(cp),'status':cd.get('status')}
        if name=='010-self-and-interlayer-only':
            if cd.get('status')!='PASS_CHANGED_FACES_SELF_CROSS_ALL185': raise ValueError('713 010 self/interlayer source proof status failed')
            selfcounts=[int(v.get('count',-1)) for v in cd.get('full_current_self_results',{}).values()]
            crosscounts=[int(v.get('count',-1)) for v in cd.get('full_current_changed_cross_results',{}).values()]
            if not selfcounts or any(selfcounts) or any(crosscounts): raise ValueError('713 010 self/interlayer exact counts are not all zero')
            if item.get('all185_subsection')!='EXCLUDED_INVALID_MIXED_UNITS': raise ValueError('713 010 all185 limitation changed unexpectedly')
            entry.update({'self_pair_counts':selfcounts,'cross_pair_counts':crosscounts,'all185_subsection':'EXCLUDED_INVALID_MIXED_UNITS'})
        elif name=='014-all185-f32-f64':
            if cd.get('status')!='PASS_WHOLE_CURRENT_OID_VS_ALL185_F32_F64' or int(cd.get('all185_total_pairs',-1))!=0: raise ValueError('713 014 all185 exact proof failed')
            entry.update({'all185_pairs':int(cd.get('all185_total_pairs',-1)),'all185_records':item.get('records'),'domains':item.get('domains')})
        else:
            if int(item.get('surface_pairs',-1))!=0 or int(item.get('all185_pairs',-1))!=0: raise ValueError(f'713 {name} delta is not zero')
            entry.update({k:item[k] for k in ('surface_pairs','all185_pairs','changed_faces','complete_surface_faces') if k in item})
            if name=='complete-geometry-038':
                if cd.get('status')!='PASS_FIXED_COORDINATE_GEOMETRY_DELTA' or int(cd.get('changed_vs_complete_current_surfaces',{}).get('count',-1))!=0 or int(cd.get('changed_vs_all185_both_domains',{}).get('count',-1))!=0: raise ValueError('713 038 exact delta failed')
            elif name=='D-delta-128':
                if cd.get('status')!='PASS_EXACT_CURRENT_D_ONE_ULP_DELTA' or int(cd.get('changed_vs_complete_current_surfaces',{}).get('count',-1))!=0 or int(cd.get('changed_vs_all185_both_domains',{}).get('count',-1))!=0: raise ValueError('713 D128 exact delta failed')
            elif name=='D-delta-137':
                if cd.get('status')!='PASS_EXACT_CURRENT_D_BOUNDARY_REFINEMENT_DELTA' or int(cd.get('changed_vs_complete_current_surfaces',{}).get('count',-1))!=0 or int(cd.get('changed_vs_all185_both_domains',{}).get('count',-1))!=0: raise ValueError('713 D137 exact delta failed')
        coverage.append(entry)
    report={'schema':'numi.human.volume-surface-inherited-delta-audit.v1','status':'RUNNING','validator':{'path':str(pathlib.Path(__file__).resolve()),'sha256':sha(pathlib.Path(__file__).resolve())},'volume_report':{'path':str(volume_report_path),'sha256':sha(volume_report_path),'declared_status':rep.get('status'),'physical_admission':False,'native_steps':0},'references':{},'roles':{},'limitations':['Source-only surface identity and inherited exact-intersection scope; no mass, native execution, or physical admission.','Old-old pairs inherit geometry scope 713. New/current-vs-new pairs inherit collapse-batch-002 exact joint changed-vs-complete O/I/D/wall/all185 audit. Original faces restored by the volume quotient and absent from collapse-batch-002 are directly queried against actual new faces below.','713 exact-gates-010 all185 subsection is explicitly excluded due mixed units; all185 inheritance uses exact-gates-014 F32/F64 common-lattice proof.']}
    report['references']={
      'source_shells':{},
      'dp_report':{'path':str(DP_REPORT),'sha256':sha(DP_REPORT),'status':dp.get('status')},
      'collapse002_report':{'path':str(COLLAPSE_REPORT),'sha256':sha(COLLAPSE_REPORT),'status':collapse.get('status'),'changed_query_records':collapse['joint_changed_vs_complete_OID_walls_all185'].get('query_records'),'joint_changed_pair_count':collapse['joint_changed_vs_complete_OID_walls_all185'].get('pair_count'),'joint_changed_self_pairs':collapse['joint_changed_vs_complete_OID_walls_all185'].get('new_face_self_pairs'),'all185_records':collapse.get('input_all185_bone_records')},
      'inherited_713_scope':{'path':str(SCOPE713),'sha256':sha(SCOPE713),'status':scope.get('status'),'physical_admission':scope.get('physical_admission'),'runtime':scope.get('runtime'),'coverage_chain':coverage},
      'exact_owner':{'path':str(OWNER),'sha256':sha(OWNER)}
    }
    source_data={}; role_refs={}
    for role in ('Skin','Fat'):
        r=roles[role]
        for field in ('input_volume','output'):
            if field not in r: raise ValueError(f'{role} report lacks {field}')
        ip=pathlib.Path(r['input_volume']); op=pathlib.Path(r['output'])
        if sha(ip)!=r.get('input_sha256'): raise ValueError(f'{role} input volume hash mismatch')
        if sha(op)!=r.get('output_sha256'): raise ValueError(f'{role} output volume hash mismatch')
        shellp=SHELLS[role]; z=np.load(shellp,allow_pickle=False); points=np.asarray(z['points_world_m'],dtype=np.float32); allfaces=np.asarray(z['faces'],dtype=np.int64)
        markers=np.asarray(z['face_marker'],dtype=np.int64)
        if points.ndim!=2 or points.shape[1]!=3 or allfaces.ndim!=2 or allfaces.shape[1]!=3: raise ValueError(f'{role} shell arrays malformed')
        rolebase={}; rolecollapse={}; rolechanged={}; role_dp={}
        for letter in ROLE_LETTERS[role]:
            sf=role_shell_faces(z,letter)
            bkeys,_=tri_keys(points,sf); unique_index(bkeys,f'{role} {letter} shell role')
            dpfile=pathlib.Path(dp['input_candidates'][letter]['path']); dp_npz=np.load(dpfile,allow_pickle=False)
            dppts=dp_npz['points_world_m_f32'] if 'points_world_m_f32' in dp_npz.files else np.asarray(dp_npz['points_world_m'],dtype=np.float32)
            dkeys,_=tri_keys(dppts,dp_npz['faces']); unique_index(dkeys,f'{letter} DP input')
            if not (len(bkeys)==len(dkeys) and np.array_equal(np.sort(bkeys),np.sort(dkeys))): raise ValueError(f'{role} {letter} exact shell-to-DP baseline face identity mismatch')
            if sha(dpfile)!=dp['input_candidates'][letter]['sha256']: raise ValueError(f'{letter} DP candidate pin mismatch')
            dpout=dp['outputs'][letter]
            cinput=collapse['input_candidates'][letter]
            if dpout.get('path')!=cinput.get('path') or dpout.get('sha256')!=cinput.get('sha256') or sha(dpout['path'])!=dpout.get('sha256'):
                raise ValueError(f'{letter} collapse002 input is not the pinned DP output')
            cinfo=collapse['outputs'][letter]; cfile=pathlib.Path(cinfo['path']); cn=np.load(cfile,allow_pickle=False)
            cpoints=cn['points_world_m_f32'] if 'points_world_m_f32' in cn.files else np.asarray(cn['points_world_m'],dtype=np.float32)
            cfaces=np.asarray(cn['faces'],dtype=np.int64); ckeys,_=tri_keys(cpoints,cfaces); unique_index(ckeys,f'{letter} collapse output')
            if sha(cfile)!=cinfo['sha256']: raise ValueError(f'{letter} collapse output pin mismatch')
            # The frozen collapse report explicitly maps every changed output row by patch.
            # `face_output_to_input_rows` has additional sentinel semantics for typed
            # ancestry and is not the changed-row selector.
            changed_rows=np.asarray([int(row) for patch in cinfo.get('patches',[]) for row in patch.get('output_face_rows',[])],dtype=np.int64)
            if len(changed_rows)!=int(cinfo['new_faces']) or len(np.unique(changed_rows))!=len(changed_rows) or np.any(changed_rows<0) or np.any(changed_rows>=len(cfaces)):
                raise ValueError(f'{letter} collapse report changed output-row partition invalid')
            changed_keys,_=tri_keys(cpoints,cfaces[changed_rows]); unique_index(changed_keys,f'{letter} changed face subset')
            rolebase[letter]=bkeys; role_dp[letter]=dkeys; rolecollapse[letter]=ckeys; rolechanged[letter]=changed_keys
            role_refs[(role,letter)]={'shell_faces':sf,'shell_keys':bkeys,'collapse_keys':ckeys,'changed_keys':changed_keys,'collapse_points':cpoints,'collapse_faces':cfaces,'changed_output_rows':changed_rows}
            cn.close();dp_npz.close()
        shellkeys,_=tri_keys(points,allfaces); unique_index(shellkeys,f'{role} full source shell')
        outz=np.load(op,allow_pickle=False); invol=np.load(ip,allow_pickle=False)
        volpts=np.asarray(outz['points_world_m'],dtype=np.float32); tets=np.asarray(outz['tetrahedra'],dtype=np.int64)
        actualfaces,incidence=tet_boundary(tets)
        # Check any serialized boundary list has the same exact triangle set as incidence extraction.
        if 'boundary_faces' in outz.files:
            stored=np.asarray(outz['boundary_faces'],dtype=np.int64)
            sk,_=tri_keys(volpts,stored); ak,_=tri_keys(volpts,actualfaces)
            if len(sk)!=len(ak) or not np.array_equal(np.sort(sk),np.sort(ak)): raise ValueError(f'{role} serialized boundary_faces disagree with tetrahedral incidence')
        actualkeys,actual_order=tri_keys(volpts,actualfaces); unique_index(actualkeys,f'{role} actual volume boundary')
        # Check point row maps preserve exact coordinates and the immutable shell prefix.
        inputpts=np.asarray(invol['points_world_m'],dtype=np.float32)
        if len(inputpts)<len(points) or not np.array_equal(inputpts[:len(points)].view(np.uint32),points.view(np.uint32)):
            raise ValueError(f'{role} volume input does not preserve shell point prefix bitwise')
        node_map=np.asarray(outz['output_node_to_input_node'],dtype=np.int64)
        if node_map.shape!=(len(volpts),) or np.any(node_map<0) or np.any(node_map>=len(inputpts)): raise ValueError(f'{role} output node map invalid')
        if not np.array_equal(volpts.view(np.uint32),inputpts[node_map].view(np.uint32)): raise ValueError(f'{role} volume output points do not replay exact input node map')
        boundary_input_rows=node_map[actualfaces]
        surface_vertices=boundary_input_rows.reshape(-1)
        if np.any(surface_vertices>=len(points)): raise ValueError(f'{role} output boundary contains non-shell input vertices')
        # Exact baseline / full-collapse coverage for entire current boundary; role partitions use face keys.
        # Build vector union to keep the exact face partition reproducible and bounded.
        known_arr=np.concatenate([shellkeys,*rolecollapse.values()])
        known_unique=np.unique(known_arr)
        is_known=contains(actualkeys,known_unique,'known shell+collapse union')
        unknown_count=int(np.count_nonzero(~is_known))
        # Input boundary must also remain in the inherited baseline/collapse surface domain.
        in_faces,_=tet_boundary(np.asarray(invol['tetrahedra'],dtype=np.int64))
        in_keys,_=tri_keys(inputpts,in_faces)
        input_known=contains(in_keys,known_unique,'input known union')
        input_unknown=int(np.count_nonzero(~input_known))
        input_equals_original_shell=bool(len(in_keys)==len(shellkeys) and np.array_equal(np.sort(in_keys),np.sort(shellkeys)))
        input_boundary_vertices=in_faces.reshape(-1)
        input_boundary_all_shell_prefix=bool(not np.any(input_boundary_vertices>=len(points)))
        letter_counts={}; actual_role_masks={}
        for letter in ROLE_LETTERS[role]:
            B=rolebase[letter]; C=rolecollapse[letter]; CN=rolechanged[letter]
            inB=contains(actualkeys,B,f'{role}.{letter} baseline')
            inC=contains(actualkeys,C,f'{role}.{letter} collapse')
            rolemask=inB|inC
            newmask=rolemask & ~inB
            restoredmask=rolemask & inB & ~inC
            retainedmask=rolemask & inB & inC
            newkeys=actualkeys[newmask]
            newchanged=contains(newkeys,CN,f'{role}.{letter} changed collapse face')
            if not np.all(newchanged): raise ValueError(f'{role}.{letter} actual new faces are not all collapse002 changed-face rows')
            letter_counts[letter]={'shell_baseline_faces':int(len(B)),'dp_input_faces':int(len(role_dp[letter])),'collapse002_faces':int(len(C)),'collapse002_changed_faces':int(len(CN)),'actual_role_faces':int(np.count_nonzero(rolemask)),'actual_retained_original_and_collapse':int(np.count_nonzero(retainedmask)),'actual_restored_original_absent_from_collapse002':int(np.count_nonzero(restoredmask)),'actual_new_vs_original':int(np.count_nonzero(newmask)),'actual_new_subset_collapse002_changed_faces':True,'actual_boundary_baseline_missing':int(len(B)-np.count_nonzero(inB))}
            actual_role_masks[letter]={'mask':rolemask,'new':newmask,'restored':restoredmask}
        # Wall faces are immutable authored shell triangles and should remain present.
        wall_marker=3 if role=='Skin' else 6
        wallfaces=allfaces[markers==wall_marker]; wallkeys,_=tri_keys(points,wallfaces)
        wall_present=contains(wallkeys,actualkeys,f'{role} authored wall facets')
        role_report={'shell':{'path':str(shellp),'sha256':sha(shellp),'point_count':len(points),'face_count':len(allfaces),'wall_marker':wall_marker,'wall_face_count':len(wallfaces),'wall_faces_retained_exactly':bool(np.all(wall_present)),'wall_missing_count':int(np.count_nonzero(~wall_present))},'volume_input':{'path':str(ip),'sha256':sha(ip),'boundary_face_count':len(in_faces),'boundary_input_vertices_all_in_shell_prefix':bool(not np.any(surface_vertices>=len(points))),'outside_original_shell_plus_collapse_count':input_unknown},'volume_output':{'path':str(op),'sha256':sha(op),'node_count':len(volpts),'tetrahedron_count':len(tets),**incidence,'boundary_vertices_all_map_to_shell_prefix':bool(not np.any(surface_vertices>=len(points))),'boundary_unique_triangle_keys':len(actualkeys)},'role_face_partition':letter_counts,'actual_boundary_outside_original_shell_plus_collapse_count':unknown_count}
        if unknown_count or input_unknown or not input_boundary_all_shell_prefix or any(not x['actual_new_subset_collapse002_changed_faces'] for x in letter_counts.values()) or not np.all(wall_present):
            raise ValueError(f'{role} boundary scope failed: unknown={unknown_count}, input_unknown={input_unknown}, wall_missing={np.count_nonzero(~wall_present)}')
        source_data[role]={'shell_points':points,'shell_faces':allfaces,'shell_markers':markers,'role_faces':{l:role_shell_faces(z,l) for l in ROLE_LETTERS[role]},'actual_boundary_faces':actualfaces,'actual_boundary_keys':actualkeys,'actual_boundary_node_rows':boundary_input_rows,'actual_role_masks':actual_role_masks}
        report['roles'][role]=role_report
        report['references']['source_shells'][role]={'path':str(shellp),'sha256':sha(shellp)}
        outz.close();invol.close();z.close()
    # Shared I geometry is directly bound by exact keys and required opposite winding.
    i_actual={}
    i_orient={}
    for role in ('Skin','Fat'):
        d=source_data[role]; m=d['actual_role_masks']['I']['mask']; faces=d['actual_boundary_faces'][m]; points=d['shell_points']
        # Actual boundary rows map output volume->input row; the original shell point prefix is exact.
        # Point coordinates themselves are recovered from corresponding shell rows below.
        local_ids=d['actual_boundary_node_rows'][m]
        tri_shell=local_ids
        tri_points=points[tri_shell]
        k,order=tri_keys(points,tri_shell)
        if len(k)!=len(np.unique(k)): raise ValueError(f'{role} I actual surface has duplicate faces')
        i_actual[role]=k
        # Orientation parity of actual outward boundary triangles relative to canonical sorted vertex keys.
        inv=np.zeros(len(order),dtype=np.uint8)
        inv += (order[:,0]>order[:,1]); inv += (order[:,0]>order[:,2]); inv += (order[:,1]>order[:,2]); inv &= 1
        i_orient[role]={bytes(k[i]):int(inv[i]) for i in range(len(k))}
    if not (len(i_actual['Skin'])==len(i_actual['Fat']) and np.array_equal(np.sort(i_actual['Skin']),np.sort(i_actual['Fat']))):
        raise ValueError('Skin/Fat actual I boundary triangle key sets differ')
    ikeys=np.sort(i_actual['Skin']); imapF=np.searchsorted(np.sort(i_actual['Fat']),ikeys)
    common=len(ikeys); opposite=sum(i_orient['Skin'][k]!=i_orient['Fat'][k] for k in i_orient['Skin'])
    same=common-opposite
    if same: raise ValueError(f'Skin/Fat I boundary has {same} same-direction triangles')
    report['shared_I']={'triangle_count_per_role':common,'sets_identical_exact_F32':True,'oppositely_oriented_triangle_count':opposite,'same_direction_triangle_count':same}
    # Direct exact intersection query: actual new faces vs actual restored original faces absent from full collapse002.
    skinz=np.load(SHELLS['Skin'],allow_pickle=False); fatz=np.load(SHELLS['Fat'],allow_pickle=False)
    sp=np.asarray(skinz['points_world_m'],dtype=np.float32); fp=np.asarray(fatz['points_world_m'],dtype=np.float32)
    nsp=len(sp); joint=np.concatenate([sp,fp],axis=0)
    # Prove the two role shells carry the same I source rows and exact coordinates; these are the only cross-shell seam identities.
    si=np.asarray(skinz['registered_I_source_vertex_rows'],dtype=np.int64); sn=np.asarray(skinz['registered_I_shell_node_rows'],dtype=np.int64)
    fi=np.asarray(fatz['registered_I_source_vertex_rows'],dtype=np.int64); fn=np.asarray(fatz['registered_I_shell_node_rows'],dtype=np.int64)
    if len(np.unique(si))!=len(si) or len(np.unique(fi))!=len(fi) or not np.array_equal(np.sort(si),np.sort(fi)): raise ValueError('Skin/Fat I source-vertex crosswalk domain mismatch')
    fby={int(x):int(y) for x,y in zip(fi,fn)}
    sby={int(x):int(y) for x,y in zip(si,sn)}
    seam=[]
    for source_row in sorted(sby):
        sr=sby[source_row]; fr=fby[source_row]
        if not np.array_equal(sp[sr].view(np.uint32),fp[fr].view(np.uint32)): raise ValueError(f'I shared source vertex {source_row} differs in F32 bits')
        seam.append((sr,nsp+fr))
    if len(set(x for g in seam for x in g))!=2*len(seam): raise ValueError('shared I seam vertex rows overlap')
    owner=owner_module()
    # Build actual-new/restored faces on one authenticated source-vertex domain.
    # The shared I shell is one geometric surface (exact same F32 triangles on Skin/Fat),
    # so include its Skin copy once in the role audit; D comes from Fat.
    old_faces_by_role={}; new_faces_by_role={}; category_counts={}; old_domains={}; new_domains={}
    role_code={'O':0,'I':1,'D':2}
    for letter in ('O','I','D'):
        source_role='Skin' if letter in ('O','I') else 'Fat'
        d=source_data[source_role]; offset=0 if source_role=='Skin' else nsp
        masks=d['actual_role_masks'][letter]
        restore_idx=np.flatnonzero(masks['restored'])
        new_idx=np.flatnonzero(masks['new'])
        old_faces_by_role[letter]=d['actual_boundary_node_rows'][restore_idx]+offset
        new_faces_by_role[letter]=d['actual_boundary_node_rows'][new_idx]+offset
        shell_keys=role_refs[(source_role,letter)]['shell_keys']
        collapse_changed=role_refs[(source_role,letter)]['changed_keys']
        restore_keys=d['actual_boundary_keys'][restore_idx]
        new_keys=d['actual_boundary_keys'][new_idx]
        _,old_source_rows=lookup_rows(restore_keys,shell_keys,f'{letter} restored shell source rows')
        _,new_collapse_rows=lookup_rows(new_keys,collapse_changed,f'{letter} collapse changed output rows')
        if np.any(old_source_rows<0) or np.any(new_collapse_rows<0): raise ValueError(f'{letter} record crosswalk failed')
        old_domains[letter]=np.column_stack((np.full(len(restore_idx),role_code[letter],np.int64),restore_idx,old_source_rows))
        new_domains[letter]=np.column_stack((np.full(len(new_idx),role_code[letter],np.int64),new_idx,new_collapse_rows))
        category_counts[letter]={'restored_original_absent_collapse002':int(len(restore_idx)),
                                 'actual_new_vs_original':int(len(new_idx))}
    old_concat=np.concatenate([old_faces_by_role[x] for x in ('O','I','D')],axis=0)
    new_concat=np.concatenate([new_faces_by_role[x] for x in ('O','I','D')],axis=0)
    rec,_=owner.float32_surface_records(joint,np.vstack([old_concat,new_concat]),source_seam_groups=seam)
    old_count=len(old_concat); old_all=tuple(rec[:old_count]); new_all=tuple(rec[old_count:])
    old_by_role={}; new_by_role={}; cursor=0
    for letter in ('O','I','D'):
        n=len(old_faces_by_role[letter]); old_by_role[letter]=tuple(old_all[cursor:cursor+n]); cursor+=n
    cursor=0
    for letter in ('O','I','D'):
        n=len(new_faces_by_role[letter]); new_by_role[letter]=tuple(new_all[cursor:cursor+n]); cursor+=n
    direct_pairs=set(); direct_candidates=0; audit_pairs=[]
    for letter in ('O','I','D'):
        q=owner._audit_pair(old_by_role[letter],new_by_role[letter],same_surface=True) if old_by_role[letter] and new_by_role[letter] else {'triangle_pairs':[],'count':0,'aabb_candidate_pairs':0}
        direct_candidates += int(q.get('aabb_candidate_pairs',0))
        for a,b in q.get('triangle_pairs',[]): direct_pairs.add((int(a),int(b)))
        audit_pairs.append({'kind':'same-role','role':letter,'restored_faces':len(old_by_role[letter]),'new_faces':len(new_by_role[letter]),'pair_count':int(q.get('count',0)),'aabb_candidate_pairs':int(q.get('aabb_candidate_pairs',0)),'allowed_shared_feature_pairs':int(q.get('allowed_shared_vertex_or_edge_pairs',0))})
    for ai,a in enumerate(('O','I','D')):
        for b in ('O','I','D')[ai+1:]:
            q1=owner._audit_pair(old_by_role[a],new_by_role[b],same_surface=False) if old_by_role[a] and new_by_role[b] else {'triangle_pairs':[],'count':0,'aabb_candidate_pairs':0}
            q2=owner._audit_pair(old_by_role[b],new_by_role[a],same_surface=False) if old_by_role[b] and new_by_role[a] else {'triangle_pairs':[],'count':0,'aabb_candidate_pairs':0}
            direct_candidates += int(q1.get('aabb_candidate_pairs',0))+int(q2.get('aabb_candidate_pairs',0))
            for x,y in q1.get('triangle_pairs',[]): direct_pairs.add((int(x),old_count+int(y)))
            for x,y in q2.get('triangle_pairs',[]): direct_pairs.add((int(y),old_count+int(x)))
            audit_pairs.append({'kind':'cross-role','roles':[a,b],'restored_A_vs_new_B_pairs':int(q1.get('count',0)),'restored_B_vs_new_A_pairs':int(q2.get('count',0)),'aabb_candidate_pairs':int(q1.get('aabb_candidate_pairs',0))+int(q2.get('aabb_candidate_pairs',0))})
    witness=np.asarray(sorted(direct_pairs),dtype=np.int64).reshape((-1,2))
    np.save(outdir/'restored-original-vs-actual-new-exact-pairs.npy',witness,allow_pickle=False)
    old_map=np.concatenate([old_domains[x] for x in ('O','I','D')],axis=0)
    new_map=np.concatenate([new_domains[x] for x in ('O','I','D')],axis=0)
    np.savez_compressed(outdir/'restored-new-record-domain-map.npz',
                        old_record_role_actual_boundary_row_shell_face_row=old_map,
                        new_record_role_actual_boundary_row_collapse_changed_face_row=new_map,
                        role_codes=np.asarray(['O','I','D'],dtype='U1'))
    report['direct_restored_vs_new_audit']={'predicate_owner_path':str(OWNER),'predicate_owner_sha256':sha(OWNER),
        'restored_face_count':int(old_count),'actual_new_face_count':int(len(new_concat)),
        'restored_by_role':category_counts,'role_code_order':['O','I','D'],
        'shared_I_source_vertex_seams':len(seam),'shared_I_source_vertex_crosswalk_verified_exact':True,
        'audit_pairs':audit_pairs,'aabb_candidate_pairs':direct_candidates,
        'intersecting_pair_count':int(len(witness)),
        'record_domain_map_path':str(outdir/'restored-new-record-domain-map.npz'),
        'record_domain_map_sha256':sha(outdir/'restored-new-record-domain-map.npz'),
        'witness_path':str(outdir/'restored-original-vs-actual-new-exact-pairs.npy'),
        'witness_sha256':sha(outdir/'restored-original-vs-actual-new-exact-pairs.npy'),'witness_shape':list(witness.shape)}
    skinz.close(); fatz.close()
    report['status']='PASS_INHERITED_DELTA_PLUS_DIRECT_RESTORED_NEW_EXACT'
    if len(witness): report['status']='FAIL_RESTORED_ORIGINAL_VS_ACTUAL_NEW_INTERSECTION'
    report['physical_admission']=False; report['native_steps']=0; report['mass_transport']='NOT_RUN'
    report['elapsed_seconds']=time.time()-started
    atomic_json(outdir/'report.json',report)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--volume-report',required=True)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    try:
        res=run(args.volume_report,args.out)
        print(json.dumps({'status':res['status'],'report':str(pathlib.Path(args.out)/'report.json'),'report_sha256':sha(pathlib.Path(args.out)/'report.json'),'roles':{r:{'boundary_faces':v['volume_output']['boundary_faces'],'boundary_unknown':v['actual_boundary_outside_original_shell_plus_collapse_count'],'face_partition':v['role_face_partition']} for r,v in res['roles'].items()},'shared_I':res['shared_I'],'direct_restored_new':res['direct_restored_vs_new_audit']},indent=2))
        if res['status']!='PASS_INHERITED_DELTA_PLUS_DIRECT_RESTORED_NEW_EXACT': raise SystemExit(2)
    except Exception as e:
        fail={'schema':'numi.human.volume-surface-inherited-delta-audit.v1','status':'FAIL_CLOSED','error_type':type(e).__name__,'error':str(e),'physical_admission':False,'native_steps':0,'mass_transport':'NOT_RUN'}
        atomic_json(pathlib.Path(args.out)/'failure.json',fail)
        raise
