#!/usr/bin/env python3
"""Pinned O-to-Skin-only adapter for 855 lineage-aware protected rows, using the owner-level point index API.

The source owner still authenticates O/I/D lineages and all O back/rim/route
rows. Only the volume-domain request is narrowed to Skin because this O-only
quotient does not edit Fat.
"""
from pathlib import Path
import hashlib, importlib.util, sys
import numpy as np

HERE=Path(__file__).resolve().parent
H=HERE.parent
PROTECTION=H/'quality-partial-cavity-684-root'/'paired_connected_boundary_patch_855_active_rows_lineage.py'
PROTECTION_SHA='5645dbfda4312b6df81b31bca03ba75fdf16af4f4956540c3b92f66b67eb101d'
PROTECTION_OWNER_SHA='68124ffcffcc2bf5147de081e50f87739089c3be5c2370be8ae3b27bef4e922a'
EXPECTED_O_BACK_ROWS=177

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def imp(name,path,expected):
    path=Path(path).resolve()
    if sha(path)!=expected:
        raise ValueError(name+' SHA mismatch')
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader:
        raise RuntimeError('cannot load '+str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def _integer_rows(record,key,limit,label,required=False):
    raw=record.get(key,[])
    arr=np.asarray(raw)
    if arr.ndim!=1:
        raise ValueError(label+' must be a one-dimensional integer row list')
    if arr.size==0:
        return np.empty(0,np.int64)
    if arr.dtype.kind not in 'iu':
        raise ValueError(label+' must be a one-dimensional integer row list')
    rows=arr.astype(np.int64,copy=False)
    if required and rows.size==0:
        raise ValueError(label+' is empty')
    if rows.size and (int(rows.min())<0 or int(rows.max())>=int(limit)):
        raise ValueError(label+' is outside its pinned source domain')
    if np.unique(rows).size!=rows.size:
        raise ValueError(label+' contains duplicate source rows')
    return rows

def o_rows_from_protection(protection,source_point_count):
    """Read only Skin's O-local record; Fat payload/cardinality is irrelevant."""
    if not isinstance(protection,dict) or 'Skin' not in protection:
        raise ValueError('protection lacks requested Skin volume scope')
    skin=protection['Skin']
    report=skin.get('report',{}) if isinstance(skin,dict) else {}
    sources=report.get('sources',{}) if isinstance(report,dict) else {}
    rec=sources.get('O') if isinstance(sources,dict) else None
    if not isinstance(rec,dict):
        raise ValueError('Skin protected-scope report lacks authenticated O source rows')
    back=_integer_rows(rec,'back_rows',source_point_count,'O back rows',required=True)
    if back.size!=EXPECTED_O_BACK_ROWS:
        raise ValueError('O back-row count is not the pinned 177-row domain')
    rim=_integer_rows(rec,'rim_rows',source_point_count,'O rim rows')
    route=_integer_rows(rec,'route_rows',source_point_count,'O route rows')
    all_rows=np.unique(np.concatenate((back,rim,route))).astype(np.int64,copy=False)
    fixed=np.unique(np.concatenate((rim,route))).astype(np.int64,copy=False)
    return fixed,back,all_rows,{'back_rows':int(back.size),
        'rim_rows':int(rim.size),'route_rows':int(route.size),'source_rows':int(all_rows.size)}

def protected_rows_with_owner(state_path,state,role,surfaces,volumes,out,coremod,owner):
    if role!='O':
        raise ValueError('Skin-only protection adapter is restricted to O')
    if 'Skin' not in volumes:
        raise ValueError('O protection requires the current Skin volume')
    if not hasattr(owner,'P') or sha(owner.P.__file__)!=PROTECTION_OWNER_SHA:
        raise ValueError('pinned 795 protection owner mismatch')
    surface=surfaces.get('O')
    if not isinstance(surface,dict) or 'points_world_m_f32' not in surface:
        raise ValueError('O source lacks pinned Float32 point rows')
    source_points=np.asarray(surface['points_world_m_f32'],np.float32)
    skin=volumes['Skin']
    volume_raw=skin['points_world_m_f32'] if 'points_world_m_f32' in skin else skin['points_world_m']
    volume_points=np.asarray(volume_raw,np.float32)
    if source_points.ndim!=2 or source_points.shape[1]!=3 or volume_points.ndim!=2 or volume_points.shape[1]!=3:
        raise ValueError('O/Skin point domains are malformed')
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    # 855 validates O/I/D source lineage, route and rim rows. Its volume
    # scope is intentionally only Skin for an O-only edit.
    skin_cache={'points':volume_points,'coordinate_index':owner.point_key_index(volume_points)}
    protected=owner.protected_scope(state_path,state,'O',{'Skin':skin_cache},out)
    fixed,back,source_rows,counts=o_rows_from_protection(protected,len(source_points))
    mapped=coremod._exact_rows_to_volume(source_points,volume_points,source_rows)
    if set(map(int,mapped.keys()))!=set(map(int,source_rows)):
        raise ValueError('O protection exact Skin mapping is incomplete')
    volume_rows=np.asarray(list(mapped.values()),np.int64)
    if np.unique(volume_rows).size!=volume_rows.size:
        raise ValueError('O protected source-to-Skin point map is not injective')
    if volume_rows.size and (int(volume_rows.min())<0 or int(volume_rows.max())>=len(volume_points)):
        raise ValueError('O protected row maps outside Skin point domain')
    if not set(map(int,back)).issubset(set(map(int,source_rows))):
        raise ValueError('authenticated O back rows missing from O protection')
    return fixed,back,protected,owner,{'source_rows':source_rows,
        'volume_rows':volume_rows,'counts':counts}

def protected_rows(state_path,state,role,surfaces,volumes,out,coremod):
    owner=imp('o_protection_scope_855_pinned_010',PROTECTION,PROTECTION_SHA)
    result=protected_rows_with_owner(state_path,state,role,surfaces,volumes,out,coremod,owner)
    fixed,back,protected,owner,detail=result
    return fixed,back,protected,owner
