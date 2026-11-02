from pathlib import Path
import json,hashlib,gzip,shutil,subprocess
S=Path(__file__).resolve().parent.parent
U=Path('/Users/numi/Work/current-physical-transfer-publication-20261101')
L=Path('/Users/numi/Work/current-fractional-020-publication-20261101')
E=U/'tools/evidence/current-native-memory-recovery-20261102'
terminal=json.loads((S/'native-import-115/install-motion.execution.json').read_text())
post=json.loads((S/'older-storage-recovery-108/post-verification.json').read_text())
assert post['status']=='PASS'
assert terminal['resource_stop_reason'], 'Inspect physical terminal result and geometry before publishing'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def copy(p,rel):
 p=Path(p);target=E/'evidence'/rel
 target.parent.mkdir(parents=True,exist_ok=True)
 if (p.stat().st_size>1024**2 and p.suffix in ('.jsonl','.json')) or p.name=='gpu-entry-vmmap.stdout.log':
  target=target.with_name(target.name+'.gz')
  with p.open('rb') as src,target.open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as dst:shutil.copyfileobj(src,dst)
  with gzip.open(target,'rb') as restored:assert hashlib.file_digest(restored,'sha256').hexdigest()==sha(p)
 else:
  shutil.copy2(p,target);assert sha(target)==sha(p)
for label in ['native-import-106','native-import-115']:
 for p in (S/label).iterdir():
  if p.is_file() and p.suffix in ('.json','.jsonl','.py','.log','.txt') and p.stat().st_size<16*1024**2:copy(p,Path(label)/p.name)
for label in ['older-storage-recovery-108','older-small-file-storage-109','older-small-file-storage-109/remaining-002','older-large-transparent-storage-110','older-compressed-frozen-storage-111','older-near-copy-storage-112','older-workspace-copy-storage-113','completed-native-copy-storage-116']:
 d=S/label
 names=['run.py','verify.py','verify_current_integrity.py','prepare_publication.py','execution.json','inventory.json','selection.json','post-verification.json','replacements.jsonl','clone-control.json','block-control.json','page-similarity-survey.json','protected-current-inputs.json']
 for name in names:
  if (d/name).is_file():copy(d/name,Path(label)/name)
for name in ['run.py','archive-index.json','SHA256SUMS','release-notes.md','execution.json','restore-verification.json','retirement-intent.json','release-create.stdout.log']:
 copy(S/'older-witness-github-retention-114'/name,Path('older-witness-github-retention-114')/name)
for name in ['native-source-pins-104.json','retry-001/build-source-device-manifest.json']:
 copy(S/'preparation-autorelease-lifetime-101'/name,Path('execution-source-101')/name)
copy(S.parent/'current-native-1024-root/integration-checkpoint.json','integration-checkpoint.json')
recovery=json.loads((S/'older-witness-github-retention-114/execution.json').read_text())
summary={'storage_status':'PASS_CONTENT_PRESERVED_AND_ARCHIVE_RESTORE_VERIFIED','post_restart_baseline_free_bytes':11995049984,'after_archive_free_bytes':32321232896,'net_available_increase_at_cleanup_checkpoint_bytes':20326182912,'current_available_bytes':shutil.disk_usage(S).free,'independently_verified_remote_retired_files':5,'retired_allocated_bytes':recovery['retired_original_allocated_bytes'],'independent_cow_paths_reverified':post['independent_cow_paths_checked'],'source_hashes_reverified':post['execution_source_hashes_checked'],'native_input_pins_reverified':post['current_asset_and_role_pins_checked'],'content_failures':len(post['storage_content_or_metadata_failures']),'native_trial_115':terminal,'physical_motion_verified':False,'accepted_full_human_steps':0,'source_skin_fat_quality_failures':0,'source_crossings':0,'source_revision_execution':'f75955f08d6edf0f85027eac7614017ac9fbc996','tested_lab_commit':'ace651f073e32916435d1eb7385ff43bc7a44340','large_trial_comparison':'one GPU-entry pair, not sustained performance; 29 preparation outputs byte-identical'}
(E/'qualification-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(E/'README.md').write_text(f"""# Current Skin/Fat native memory recovery and retained evidence

Cleanup on SSH Numi increased available storage from 11,995,049,984 B after restart to 32,321,232,896 B after retention: **20,326,182,912 B recovered (20.3 GB decimal)** at that checkpoint. New native outputs and OS paging subsequently consume space; current availability is separately recorded, not substituted for the cleanup checkpoint.

Existing files were preserved through byte-verified transparent compression and independent APFS copy-on-write storage sharing. Whole-file hashes and metadata were checked before replacement; independent-write controls passed. The final recheck covers {post['independent_cow_paths_checked']:,} changed paths, {post['compression_paths_checked']} compressed paths, all {post['execution_source_hashes_checked']} current execution-source hashes and all {post['current_asset_and_role_pins_checked']} native input pins, with no failures. This changes filesystem storage, not model coordinates, geometry, materials, mass, contact or acceptance policy.

Five older pre-compensation witness outputs were uploaded to a **draft retention archive** in Numi2/numilab-human. All four downloaded archive hashes, five member hashes and an actual 933,678,749-byte restored file passed independent verification before local retirement. Per-file ancestry and hashes remain in archive-index.json. Historical producer witness paths may require restoration for rerunning those historical producer/audit stages; none of the 88 current native asset/role pins was retired.

## Native result

Trial106 retained 29 byte-identical preparation/initial-state outputs versus099. At the same GPU-entry observation, vmmap reported rounded physical footprint 9.8G to9.0G, peak13.3G to13.1G, and DefaultMalloc allocations 10,338,485 to74,307. This is one diagnostic pair with differing existing OS swap, not a matched multi-run performance claim. The source/import checks passed; the process stopped on its 768MiB extra-swap guard before a terminal solver result.

After restart and cleanup, trial115 used the same tested binary, 760 execution-source hashes, 88 native pins, all15,453,677 contact rows, strict contact-capacity overflow rejection,a64-iteration FGMRES budget and one-Newton diagnostic budget. The operational resource envelope permitted3GiB additional swap, retained12GiB minimum disk and a900s wall limit; physical tolerances and mandatory true-residual certification were unchanged. Its terminal status is **{terminal['resource_stop_reason']}**, exit {terminal['exit_code']}, {terminal['elapsed_s']:.3f}s. It did not produce a certified accepted full-Human physical endpoint.

**Source Skin/Fat quality failures and source crossings remain0; full-Human motion remains unverified, accepted physical steps0.** Neither the initial-state render pack nor source import is motion qualification. The wrapper's source_import_pass flag tests the source-only exit marker; false in an interrupted motion run does not overturn the retained source-preflight PASS.

The execution owner is a preserved dirty checkout at f75955f08d6edf0f85027eac7614017ac9fbc996. Tested memory fixes are published in numi-lab main through ace651f073e32916435d1eb7385ff43bc7a44340. Exact commands, environments, device/binary/library hashes, source pins, failed trials and storage journals are retained here. No visual inspection was performed in this round.

## Restore older witness evidence

Authenticated GitHub CLI can download the draft by its stable tag:

```sh
gh release download human-pre-compensation-witness-retention-20261102-114 --repo Numi2/numilab-human --dir /path/to/empty-recovery-directory
```

Verify SHA256SUMS and the per-member hashes in archive-index.json before restoring standard tar.gz members relative to its source_root. Do not run old producers against absent archived witnesses. The draft URL is https://github.com/Numi2/numilab-human/releases/tag/untagged-96f69469c9d6f8c10fcd . This archive is historical evidence, not a qualified simulation release.

## Next executable work

The resource fixes now permit a longer GPU attempt, but one microstep still lacks a terminal solver certificate. Inspect the queued whole-stage timings and solver/contact dispatch cost with bounded, owner-level instrumentation before another identical full run. Preserve the full physical residual, strict overflow/rollback and all required coupling. Once a certified endpoint exists, use native-import-115/geometry-verification-command.json with newly pinned run paths, then extend to16 and60 simulated seconds. Do not claim0.1x or motion from this diagnostic.
""")
# Update the existing tested Lab change's qualification rather than leaving a stale RUNNING claim.
Q=L/'tools/evidence/current-native-preparation-lifetime-20261102'
q=json.loads((Q/'qualification-summary.json').read_text())
q['native_memory_benefit']='ONE_GPU_ENTRY_PAIR_MEASURED_NOT_STEADY_STATE'
q['native_trial_106']='RESOURCE_STOP_SWAP_768MiB_NO_TERMINAL_SOLVER_RESULT'
q['native_trial_115']=terminal['resource_stop_reason']
q['accepted_full_human_steps']=0
q['large_scene_preparation_parity_files']=29
(Q/'qualification-summary.json').write_text(json.dumps(q,indent=2)+'\n')
for name in ['memory-comparison.json','preparation-output-parity.json']:
 dst=Q/'evidence'/name;shutil.copy2(S/'native-import-106'/name,dst)
text=(Q/'README.md').read_text()
text=text.replace('The large full-Human trial is pending. This commit does not claim reduced peak memory, accepted full-body motion, convergence, or a performance gain. Earlier native trials stopped under resource guards without a terminal solver result.','A single large-scene GPU-entry pair now reports rounded vmmap footprint9.8G to9.0G, peak13.3G to13.1G and DefaultMalloc allocations10,338,485 to74,307, with29 preparation/initial-state outputs byte-identical. This does not establish sustained throughput, matched multi-run peak memory, convergence or accepted full-body motion. Trial106 stopped on its extra-swap guard. Trial115, after restart and storage recovery, stopped at '+terminal['resource_stop_reason']+' before a terminal solver result.')
text=text.replace('The actual large-scene benefit still requires measurement.','The retained one-pair result is limited to the sampled GPU-entry event; sustained benefit remains unqualified.')
(Q/'README.md').write_text(text)
manifest={str(p.relative_to(E)):sha(p) for p in sorted(E.rglob('*')) if p.is_file() and p.name!='sha256-manifest.json'}
(E/'sha256-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'PREPARED','files':len(manifest),'total_bytes':sum(p.stat().st_size for p in E.rglob('*') if p.is_file()),'summary':summary},indent=2))
