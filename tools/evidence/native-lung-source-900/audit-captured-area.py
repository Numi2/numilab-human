from __future__ import annotations
import hashlib, importlib.util, json, mmap, os, pathlib, struct, sys, math
from fractions import Fraction
import numpy as np

E = pathlib.Path('/Users/n/numi-human-resting-evidence-20261005')
RUN = E / 'native-final-lung-source-cycle-900'
OUT = E / 'native-final-lung-source-audit-901'
DECL = E / 'native-final-lung-source-cycle-900.declaration.json'
DRIVER = E / 'native-final-lung-source-cycle-900.py'
STEPS = (0, 6111, 9999)
SURFACES = {
    'lung_305': (51023, 305), 'lung_306': (51023, 306),
    'lung_307': (51023, 307), 'lung_308': (51023, 308),
    'lung_309': (51023, 309), 'pleura_310': (51024, 310),
    'diaphragm_311': (51010, 311),
}
HEADER = struct.Struct('<8sIIQQ32s24s')
DIRECTORY = struct.Struct('<IIQQQII32s')

def sha(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()

def exact_cross(points):
    p=[[Fraction.from_float(float(np.float32(x))) for x in row] for row in points]
    a=[p[1][i]-p[0][i] for i in range(3)]
    b=[p[2][i]-p[0][i] for i in range(3)]
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def pin_inputs():
    invocation=json.loads((RUN/'invocation.json').read_text())
    metadata=json.loads((RUN/'run-metadata.json').read_text())
    declaration=json.loads(DECL.read_text())
    driver_log=json.loads((E/'native-final-lung-source-cycle-900.driver.log').read_text())
    assert metadata.get('exit_code') == 0, metadata.get('exit_code')
    assert metadata.get('source_files_changed_during_run') == [], metadata.get('source_files_changed_during_run')
    assert metadata.get('loaded_metal_runtime',{}).get('verified') is True
    assert metadata.get('argv') == invocation.get('argv')
    assert metadata.get('asset_sha256') == invocation.get('asset_sha256')
    assert driver_log.get('exit_code') == 0 and driver_log.get('loaded_runtime_verified') is True
    assert driver_log.get('changed_sources') == []
    asset_checks=[]
    for path,want in sorted(invocation['asset_sha256'].items()):
        p=pathlib.Path(path)
        got=sha(p) if p.is_file() else None
        asset_checks.append({'path':path,'expected_sha256':want,'observed_sha256':got,'matches':got==want})
        assert got == want, path
    declaration_checks=[]
    for path,want in sorted(declaration['source_hashes'].items()):
        got=sha(path)
        declaration_checks.append({'path':path,'expected_sha256':want,'observed_sha256':got,'matches':got==want})
        assert got == want, path
    assert declaration['capture_steps'] == list(STEPS)
    assert float(metadata['wall_seconds']) > 0
    assert invocation['environment'].get('NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS') == '0,6111,9999'
    assert invocation['environment'].get('NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT') == '1'
    assert invocation['environment'].get('NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS') == '1'
    assert '--muscle-step-seconds' in invocation['argv'] and invocation['argv'][invocation['argv'].index('--muscle-step-seconds')+1] == '0.002'
    assert '--muscle-step-count' in invocation['argv'] and invocation['argv'][invocation['argv'].index('--muscle-step-count')+1] == '10000'
    log=(RUN/'native.log').read_text(errors='replace')
    assert 'resting_integrated_body=completed' in log and 'human_execution_stage=native_horizon_end' in log
    # Capture source code path is explicit in driver; source-tree commit is not in invocation.
    src_root='/Users/n/numi-human-lung-triangulation-candidate-003'
    assert f'NUMI_HUMAN_ROOT="{src_root}"' in DRIVER.read_text()
    return {
      'run_directory':str(RUN),'declaration_path':str(DECL),'declaration_sha256':sha(DECL),
      'invocation_sha256':sha(RUN/'invocation.json'),'run_metadata_sha256':sha(RUN/'run-metadata.json'),
      'driver_path':str(DRIVER),'driver_sha256':sha(DRIVER),'driver_log_path':str(E/'native-final-lung-source-cycle-900.driver.log'),
      'driver_log_sha256':sha(E/'native-final-lung-source-cycle-900.driver.log'),
      'exit_code':metadata['exit_code'],'wall_seconds':metadata['wall_seconds'],
      'source_files_changed_during_run':metadata['source_files_changed_during_run'],
      'loaded_metal_runtime':metadata['loaded_metal_runtime'],
      'asset_checks_count':len(asset_checks),'asset_checks_all_match':all(x['matches'] for x in asset_checks),
      'asset_checks':asset_checks,'declaration_source_checks':declaration_checks,
      'invocation_argv':invocation['argv'],'selected_environment':{
        k:invocation['environment'].get(k) for k in ('NUMI_HUMAN_EXECUTION_STAGES','NUMI_HUMAN_RESTING_EXPORT_MRV_STEPS','NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT','NUMI_HUMAN_ACCEPTED_COM_MOMENTUM_AUDIT_SEGMENT_STEPS','NUMI_HUMAN_ACCEPTED_Q_INTEGRATION_AUDIT')},
      'source_identity_limit':'The declaration pins the run driver, anatomy receipt, NHA and NHSKIN. The invocation/metadata pins all 25 runtime/input assets and reports no source changes during execution; it records the Human source root path but does not bind a source commit/tree hash.'
    }

def pack_open(pack_path):
    f=pathlib.Path(pack_path).open('rb')
    m=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
    h=HEADER.unpack_from(m,0)
    if h[0] != b'MRVPACK2' or h[1] != 2: raise ValueError('unsupported pack header')
    rows=[DIRECTORY.unpack_from(m,HEADER.size+i*DIRECTORY.size) for i in range(h[2])]
    by_type={r[0]:r for r in rows}
    vr,ir,pr=by_type[2],by_type[3],by_type[4]
    if (vr[5],ir[5],pr[5]) != (80,4,64): raise ValueError('unexpected section strides')
    v=np.ndarray((vr[4],20),dtype='<f4',buffer=m,offset=vr[2])
    ix=np.ndarray((ir[4],),dtype='<u4',buffer=m,offset=ir[2])
    prims={}
    for i in range(pr[4]):
        off=pr[2]+i*pr[5]
        geom=struct.unpack_from('<4I',m,off)
        ident=struct.unpack_from('<4I',m,off+16)
        key=(ident[0],ident[3])
        if key in prims: raise ValueError(f'duplicate surface key {key}')
        prims[key]={'identity':ident,'first_index':geom[0],'index_count':geom[1]}
    return f,m,v,ix,prims

def scan_surface(vertices,indices,meta):
    first=meta['first_index']; count=meta['index_count']
    if count%3: raise ValueError('nontriangle index count')
    ntri=count//3
    stats={'triangle_count':ntri,'nonfinite_position_triangles':0,'nonfinite_cross_triangles':0,
           'exact_zero_area_triangles':0,'near_exact_predicate_triangles':0,'min_area_m2':None,
           'max_area_m2':0.0,'min_altitude_m':None,'max_edge_m':0.0,'min_edge_m':None,
           'min_area_face_row':None,'min_altitude_face_row':None,'min_altitude_vertex_indices':None,
           'min_altitude_triangle_world_xyz_f32':None,'zero_area_examples':[]}
    min_area=float('inf'); min_alt=float('inf'); min_edge=float('inf')
    chunk_faces=16384
    for begin in range(0,ntri,chunk_faces):
        end=min(ntri,begin+chunk_faces)
        tri_ids=indices[first+3*begin:first+3*end].reshape(-1,3)
        if tri_ids.size and int(tri_ids.max())>=len(vertices): raise ValueError('surface index out of vertex bounds')
        p32=vertices[tri_ids,:3]
        finite=np.isfinite(p32).all(axis=(1,2))
        stats['nonfinite_position_triangles']+=int((~finite).sum())
        if not finite.all():
            p32=p32[finite]
            tri_ids=tri_ids[finite]
            if len(p32)==0: continue
        p=p32.astype(np.float64)
        e01=p[:,1]-p[:,0]; e12=p[:,2]-p[:,1]; e20=p[:,0]-p[:,2]
        c=np.cross(e01,p[:,2]-p[:,0])
        cn=np.linalg.norm(c,axis=1)
        finite_cross=np.isfinite(c).all(axis=1)&np.isfinite(cn)
        stats['nonfinite_cross_triangles']+=int((~finite_cross).sum())
        valid=finite_cross
        if not valid.all(): p,p32,tri_ids,e01,e12,e20,c,cn=(a[valid] for a in (p,p32,tri_ids,e01,e12,e20,c,cn))
        zero=np.all(c==0.0,axis=1)
        # Recompute double-zero or cancellation-scale faces over exact rationals of stored F32 coordinates.
        edge2=np.maximum.reduce((np.einsum('ij,ij->i',e01,e01),np.einsum('ij,ij->i',e12,e12),np.einsum('ij,ij->i',e20,e20)))
        near=(cn <= (64.0*np.finfo(np.float64).eps)*edge2)
        stats['near_exact_predicate_triangles']+=int(near.sum())
        for local in np.flatnonzero(near):
            exact=exact_cross(p32[local])
            exact_zero=all(x==0 for x in exact)
            if exact_zero:
                zero[local]=True
            elif zero[local]:
                raise AssertionError('binary64 zero cross disagrees with exact rational determinant')
        stats['exact_zero_area_triangles']+=int(zero.sum())
        for z in np.flatnonzero(zero):
            if len(stats['zero_area_examples'])<16:
                stats['zero_area_examples'].append({'face_row':begin+int(z),'vertex_indices':[int(x) for x in tri_ids[z]],'world_xyz_f32':p32[z].tolist()})
        areas=0.5*cn
        edge01=np.linalg.norm(e01,axis=1); edge12=np.linalg.norm(e12,axis=1); edge20=np.linalg.norm(e20,axis=1)
        longest=np.maximum.reduce((edge01,edge12,edge20)); shortest=np.minimum.reduce((edge01,edge12,edge20))
        alt=np.divide(cn,longest,out=np.zeros_like(cn),where=longest>0.0)
        if len(areas):
            local_min=int(np.argmin(areas)); local_alt=int(np.argmin(alt)); local_edge=int(np.argmin(shortest))
            if float(areas[local_min])<min_area:
                min_area=float(areas[local_min]); stats['min_area_face_row']=begin+local_min
            if float(alt[local_alt])<min_alt:
                min_alt=float(alt[local_alt]); stats['min_altitude_face_row']=begin+local_alt
                stats['min_altitude_vertex_indices']=[int(x) for x in tri_ids[local_alt]]
                stats['min_altitude_triangle_world_xyz_f32']=p32[local_alt].tolist()
            min_edge=min(min_edge,float(shortest[local_edge]))
            stats['max_area_m2']=max(stats['max_area_m2'],float(areas.max()))
            stats['max_edge_m']=max(stats['max_edge_m'],float(np.maximum.reduce((edge01,edge12,edge20)).max()))
    stats['min_area_m2']=None if min_area==float('inf') else min_area
    stats['min_altitude_m']=None if min_alt==float('inf') else min_alt
    stats['min_edge_m']=None if min_edge==float('inf') else min_edge
    stats['min_altitude_nm']=None if min_alt==float('inf') else min_alt*1e9
    stats['min_area_mm2']=None if min_area==float('inf') else min_area*1e6
    return stats

def validate_capture(step, path, receipt_path, anatomy_sha, helper):
    f,m,v,ix,prims=pack_open(path)
    rows=[DIRECTORY.unpack_from(m,HEADER.size+i*DIRECTORY.size) for i in range(HEADER.unpack_from(m,0)[2])]
    vertex_offset=next(r[2] for r in rows if r[0]==2)
    validation=helper.validate_accepted_receipt(path,receipt_path,step,m,vertex_offset,prims)
    receipt=json.loads(pathlib.Path(receipt_path).read_text())
    assert receipt.get('physical_endpoint')=='accepted' and receipt.get('surface_audit_endpoint')=='passed'
    assert receipt.get('surface_audit',{}).get('mesh_zero_area_triangles')==0
    assert receipt.get('surface_audit',{}).get('mesh_nonfinite_area_triangles')==0
    assert receipt.get('surface_audit',{}).get('mesh_triangles_checked')==4910273
    assert receipt.get('common_field_source_anatomy_payload_sha256')==anatomy_sha
    surface_reports={}
    for label,key in SURFACES.items():
        if key not in prims: raise ValueError(f'missing target surface {key}')
        row=prims[key]
        surface_reports[label]={'identity':row['identity'],'first_index':row['first_index'],'index_count':row['index_count'],**scan_surface(v,ix,row)}
        assert surface_reports[label]['exact_zero_area_triangles']==0, (step,label,surface_reports[label]['zero_area_examples'])
        assert surface_reports[label]['nonfinite_position_triangles']==0
        assert surface_reports[label]['nonfinite_cross_triangles']==0
    result={'step':step,'pack_path':str(path),'receipt_path':str(receipt_path),'receipt_validation':validation,
            'receipt_surface_audit':receipt['surface_audit'],'surfaces':surface_reports}
    del v,ix
    m.close();f.close()
    return result

def main():
    if (OUT/'report.json').exists(): raise RuntimeError('refuse to overwrite existing report')
    pins=pin_inputs()
    helper_path=E/'cardiac-wall-native-self-audit-001/accepted_mrvpack_surface_audit.py'
    spec=importlib.util.spec_from_file_location('mrv_audit_helper',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    anatomy_path=pathlib.Path('/Users/n/numi-human-resting-evidence-20261005/lung-current-precision-candidate-897/final/resting-thorax.nhanatomy')
    anatomy_sha=sha(anatomy_path)
    captures=[]
    for step in STEPS:
        p=RUN/'accepted-geometry'/f'step-{step}.mrvpack'
        r=RUN/'accepted-geometry'/f'step-{step}.receipt.json'
        if not p.is_file() or not r.is_file(): raise FileNotFoundError((p,r))
        captures.append(validate_capture(step,p,r,anatomy_sha,helper))
    # The older 896 step-5375 accepted capture is only a prior-source sensitivity comparison.
    old_run=E/'native-lung-conditioned-common-skin-cycle-896'
    old_pack=old_run/'accepted-geometry/step-5375.mrvpack'
    old_receipt=old_run/'accepted-geometry/step-5375.receipt.json'
    if old_pack.is_file() and old_receipt.is_file():
        old_r=json.loads(old_receipt.read_text())
        f,m,v,ix,prims=pack_open(old_pack)
        rows=[DIRECTORY.unpack_from(m,HEADER.size+i*DIRECTORY.size) for i in range(HEADER.unpack_from(m,0)[2])]
        vertex_offset=next(r[2] for r in rows if r[0]==2)
        old_validation=helper.validate_accepted_receipt(old_pack,old_receipt,5375,m,vertex_offset,prims)
        old_surface=scan_surface(v,ix,prims[(51023,305)])
        del v,ix
        m.close();f.close()
        old_comparison={'step':5375,'source_anatomy_sha256':old_r.get('common_field_source_anatomy_payload_sha256'),
          'pack_path':str(old_pack),'pack_sha256':sha(old_pack),
          'receipt_validation':old_validation,'receipt_surface_audit':old_r.get('surface_audit'),
          'lung_305':old_surface,'source_scope':'accepted pose from prior 894 source, not a native 897 capture'}
    else: old_comparison={'status':'prior 896 step-5375 capture absent'}
    report={'schema':'numi.human.native-final-lung-selected-surface-f32-area-audit.v1',
      'status':'passed_selected_surface_f32_area_audit',
      'scope':{'native_run':'900','native_steps':[0,6111,9999],'selected_rows':['51023:305','51023:306','51023:307','51023:308','51023:309','51024:310','51010:311'],
        'checks':'Streaming binary32-position finite checks and binary64 exact-real determinant/area for selected triangles; exact rational fallback for cancellation-scale cross products. No whole-mesh intersection scan.'},
      'run_pins':pins,'anatomy_payload':{'path':str(anatomy_path),'sha256':anatomy_sha},
      'captures':captures,'prior_896_step5375_comparison':old_comparison}
    out=OUT/'report.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'report_path':str(out),'report_sha256':sha(out),'status':report['status'],
      'capture_summaries':[{ 'step':c['step'],'surfaces':{k:{'triangles':v['triangle_count'],'zero':v['exact_zero_area_triangles'],'nonfinite':v['nonfinite_position_triangles'],'min_altitude_nm':v['min_altitude_nm'],'min_face':v['min_altitude_face_row']} for k,v in c['surfaces'].items()}} for c in captures],
      'prior_896_step5375_lung305':{k:old_comparison['lung_305'].get(k) for k in ('triangle_count','exact_zero_area_triangles','min_altitude_nm','min_altitude_face_row')} if old_comparison.get('lung_305') else old_comparison},indent=2))

if __name__=='__main__': main()
