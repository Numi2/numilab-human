from pathlib import Path
import importlib.util,json,sys
import numpy as np
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
HELP=H/'quality-partial-cavity-684-root/paired_connected_boundary_patch_854_active_rows_lineage.py'
STATE=H/'all-quality-897-qualified-root/verified-state.json'
spec=importlib.util.spec_from_file_location('protection854_actual',HELP);P=importlib.util.module_from_spec(spec);sys.modules[spec.name]=P;spec.loader.exec_module(P)
st=json.loads(STATE.read_text())
surfaces={r:P.load_npz(P.check_ref(st['source_surfaces'][r],'current '+r)) for r in ('O','I','D')}
volumes={}
for r in ('Skin','Fat'):
    v=P.load_npz(P.check_ref(st[r+'_current'],'current '+r))
    points=np.asarray(v['points_world_m_f32'] if 'points_world_m_f32' in v else v['points_world_m'],np.float32)
    volumes[r]={'points':points,'coordinate_index':P.point_key_index(points)}
protection=P.protected_scope(STATE,st,'I',volumes,H/'all-quality-897-protection-854-readonly',fixture=False)
irec=protection['Fat']['report']['sources']['actual_fat_back_union']
assert irec['historical_record_count']==790
assert len(irec['historical_source_rows'])==790 and len(irec['historical_to_current_source_rows'])==790
assert len(irec['historical_to_current_volume_rows'])==790 and len(irec['historical_direct_volume_rows_or_minus_one'])==790
assert irec['protected_union_volume_rows_count']==len(irec['current_historical_protected_volume_row_union'])
assert protection['Skin']['report']['sources']['actual_skin_i_back_lineage']['historical_record_count']==790
assert protection['Skin']['report']['sources']['O']['back_lineage']['maximum_historical_cumulative_l1_displacement_m']==0.0
print(json.dumps({'status':'PASS_854_AUTHENTICATED_897_BACK_LINEAGE','I_historical_records':790,
 'I_unique_current_source_rows':irec['unique_current_source_representatives'],
 'I_collapsed_historical_records':irec['collapsed_historical_records'],
 'I_unique_current_Fat_volume_rows':irec['unique_lineage_mapped_volume_rows'],
 'I_historical_direct_coordinate_matches':irec['historical_coordinates_directly_present_count'],
 'I_historical_direct_coordinate_missing':irec['historical_coordinates_directly_missing_count'],
 'I_Fat_protected_union_rows':irec['protected_union_volume_rows_count'],
 'I_max_historical_cumulative_L1_m':irec['maximum_historical_cumulative_l1_displacement_m'],
 'I_records_over_50um_historical_final':irec['historical_records_over_50um_final'],
 'I_max_current_transition_L1_m':irec['maximum_current_transition_l1_displacement_m'],
 'O_back_bitwise_fixed':True,'Skin_and_Fat_I_protection':True},sort_keys=True))
