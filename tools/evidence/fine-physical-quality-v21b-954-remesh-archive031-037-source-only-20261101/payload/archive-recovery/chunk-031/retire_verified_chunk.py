from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess
OUT=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706/quality-volume-archive-chunk-031-plan')
plan=json.loads((OUT/'selection-plan.json').read_text());pre=json.loads((OUT/'fresh-retirement-preflight.json').read_text());build=json.loads((OUT/'archive-build-receipt.json').read_text());verify=json.loads((OUT/'independent-verification.json').read_text());meta=json.loads((OUT/'github-release-metadata.json').read_text());index=json.loads((OUT/'archive-index.json').read_text())
def sha(b):return hashlib.sha256(b).hexdigest()
assert pre['status']=='PASS_FRESH_SOURCES_GH_RESTORE_CURRENT_OWNER_AND_LIVE_HANDLE_CHECKS' and pre['retirement_performed'] is False
assert pre['plan_sha256']==sha((OUT/'selection-plan.json').read_bytes())
assert verify['status']=='PASS_GH_DOWNLOAD_ARCHIVE_SHA_ALL_NPZ_CRC_AND_ALL_NPY_ARRAYS_LOADED_EXACT_RAM_RESTORE' and verify['blob_count']==len(plan['files'])
assert build['archive_sha256']==verify['asset_sha256']==meta['assets'][0]['digest'].removeprefix('sha256:')
assert sha(json.dumps(index,sort_keys=True,separators=(',',':')).encode())==verify['archive_index_canonical_sha256']
for row in pre['current_quality_state_scan']['json_pins']:
 p=Path(row['path']);assert p.is_file() and sha(p.read_bytes())==row['sha256']
paths=[row['path'] for row in plan['files']];lsof=[]
for p in paths:
 r=subprocess.run(['/usr/sbin/lsof','-t',p],capture_output=True,text=True)
 if r.returncode not in (0,1):raise RuntimeError('lsof failed '+r.stderr[-300:])
 if r.stdout.strip():lsof.append({'path':p,'pids':r.stdout.strip().splitlines()})
ps=subprocess.run(['/bin/ps','-Ao','pid,command'],capture_output=True,text=True,check=True).stdout;process=[line for line in ps.splitlines() if any(p in line for p in paths)]
assert not lsof and not process
# Final fresh target/source-receipt/hash checks; delete only the exact listed volume NPZs.
removed=[];free0=shutil.disk_usage(OUT).free
for row in plan['files']:
 p=Path(row['path']);s=json.loads(Path(row['source_receipt_path']).read_text());q=s[row['role']+'_current'];st=p.stat()
 assert s['physical_admission'] is False and s['native_steps']==0 and q['path']==str(p) and q['sha256']==row['expected_sha256_from_source_receipt']
 assert p.is_file() and not p.is_symlink() and st.st_nlink==1 and st.st_size==row['size_bytes_from_stat'] and sha(p.read_bytes())==q['sha256']
 digest=q['sha256'];size=st.st_size;allocated=st.st_blocks*512;p.unlink();removed.append({'path':str(p),'sha256':digest,'logical_bytes_retired':size,'allocated_bytes_before':allocated,'source_receipt_path':str(Path(row['source_receipt_path']))})
# Keep a remote restore source and all proof; remove only the verified local archive staging blob.
archive=Path(plan['release_target']['local_archive_path']);ast=archive.stat();assert not archive.is_symlink() and ast.st_nlink==1 and sha(archive.read_bytes())==build['archive_sha256'];archive_sha=build['archive_sha256'];archive_size=ast.st_size;archive_alloc=ast.st_blocks*512;archive.unlink()
free1=shutil.disk_usage(OUT).free
receipt={'schema':'numi-human.pre824-volume-proposal-retirement.v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00','Z'),'status':'PASS_GH_RESTORED_PRE824_VOLUME_PROPOSALS_RETIRED','chunk':'031','plan_sha256':sha((OUT/'selection-plan.json').read_bytes()),'roster_sha256':sha((OUT.parent/'quality-volume-pre824-archive-roster-001.json').read_bytes()),'fresh_preflight_sha256':sha((OUT/'fresh-retirement-preflight.json').read_bytes()),'independent_verification_sha256':sha((OUT/'independent-verification.json').read_bytes()),'archive_build_receipt_sha256':sha((OUT/'archive-build-receipt.json').read_bytes()),'archive_index_sha256':sha((OUT/'archive-index.json').read_bytes()),'github_release_metadata_sha256':sha((OUT/'github-release-metadata.json').read_bytes()),'archive_sha256':archive_sha,'archive_local_staging_file_removed':True,'archive_local_size_bytes':archive_size,'archive_local_allocated_bytes':archive_alloc,'archive_remote_tag':meta['tagName'],'archive_remote_asset_id':meta['assets'][0]['id'],'archive_remote_asset_digest':meta['assets'][0]['digest'],'independent_download_full_sha_pass':True,'all_selected_npz_zip_crc_pass':True,'all_npy_arrays_loaded':True,'npy_arrays_loaded':verify['npy_arrays_loaded'],'exact_ram_restore_per_file':True,'filesystem_restore_performed':False,'files_removed':removed,'file_count_removed':len(removed),'logical_bytes_retired':sum(x['logical_bytes_retired'] for x in removed),'allocated_bytes_retired':sum(x['allocated_bytes_before'] for x in removed),'df_free_bytes_before':free0,'df_free_bytes_after':free1,'measured_df_free_increase_bytes':free1-free0,'physical_admission':False,'native_steps':0,'source_arrays_states_reports_witnesses_preserved':True,'selected_only_volume_proposal_npz':True}
rec=OUT/'retirement-receipt.json';rec.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':receipt['status'],'count':len(removed),'logical_bytes_retired':receipt['logical_bytes_retired'],'allocated_bytes_retired':receipt['allocated_bytes_retired'],'local_archive_removed':archive_size,'df_before':free0,'df_after':free1,'df_gain':free1-free0,'receipt':str(rec),'receipt_sha256':sha(rec.read_bytes())},sort_keys=True))
