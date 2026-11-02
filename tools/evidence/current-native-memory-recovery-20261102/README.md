# Current Skin/Fat native memory recovery and retained evidence

Cleanup on SSH Numi increased available storage from 11,995,049,984 B after restart to 32,321,232,896 B after retention: **20,326,182,912 B recovered (20.3 GB decimal)** at that checkpoint. New native outputs and OS paging subsequently consume space; current availability is separately recorded, not substituted for the cleanup checkpoint.

Existing files were preserved through byte-verified transparent compression and independent APFS copy-on-write storage sharing. Whole-file hashes and metadata were checked before replacement; independent-write controls passed. The final recheck covers 52,964 changed paths, 112 compressed paths, all 760 current execution-source hashes and all 88 native input pins, with no failures. This changes filesystem storage, not model coordinates, geometry, materials, mass, contact or acceptance policy.

Five older pre-compensation witness outputs were uploaded to a **draft retention archive** in Numi2/numilab-human. All four downloaded archive hashes, five member hashes and an actual 933,678,749-byte restored file passed independent verification before local retirement. Per-file ancestry and hashes remain in archive-index.json. Historical producer witness paths may require restoration for rerunning those historical producer/audit stages; none of the 88 current native asset/role pins was retired.

## Native result

Trial106 retained 29 byte-identical preparation/initial-state outputs versus099. At the same GPU-entry observation, vmmap reported rounded physical footprint 9.8G to9.0G, peak13.3G to13.1G, and DefaultMalloc allocations 10,338,485 to74,307. This is one diagnostic pair with differing existing OS swap, not a matched multi-run performance claim. The source/import checks passed; the process stopped on its 768MiB extra-swap guard before a terminal solver result.

After restart and cleanup, trial115 used the same tested binary, 760 execution-source hashes, 88 native pins, all15,453,677 contact rows, strict contact-capacity overflow rejection,a64-iteration FGMRES budget and one-Newton diagnostic budget. The operational resource envelope permitted3GiB additional swap, retained12GiB minimum disk and a900s wall limit; physical tolerances and mandatory true-residual certification were unchanged. Its terminal status is **15_minute_diagnostic_budget**, exit -15, 904.893s. It did not produce a certified accepted full-Human physical endpoint.

**Source Skin/Fat quality failures and source crossings remain0; full-Human motion remains unverified, accepted physical steps0.** Neither the initial-state render pack nor source import is motion qualification. The wrapper's source_import_pass flag tests the source-only exit marker; false in an interrupted motion run does not overturn the retained source-preflight PASS.

The execution owner is a preserved dirty checkout at f75955f08d6edf0f85027eac7614017ac9fbc996. Tested memory fixes are published in numi-lab main through ace651f073e32916435d1eb7385ff43bc7a44340. Measured memory evidence is published at numi-lab commit8e9e4e69356c34304ff1f6f890d29807a20a8475. Exact commands, environments, device/binary/library hashes, source pins, failed trials and storage journals are retained here. No visual inspection was performed in this round.

## Restore older witness evidence

Authenticated GitHub CLI can download the draft by its stable tag:

```sh
gh release download human-pre-compensation-witness-retention-20261102-114 --repo Numi2/numilab-human --dir /path/to/empty-recovery-directory
```

Verify SHA256SUMS and the per-member hashes in archive-index.json before restoring standard tar.gz members relative to its source_root. Do not run old producers against absent archived witnesses. The draft URL is https://github.com/Numi2/numilab-human/releases/tag/untagged-96f69469c9d6f8c10fcd . This archive is historical evidence, not a qualified simulation release.

## Next executable work

The resource fixes now permit a longer GPU attempt, but one microstep still lacks a terminal solver certificate. Inspect the queued whole-stage timings and solver/contact dispatch cost with bounded, owner-level instrumentation before another identical full run. Preserve the full physical residual, strict overflow/rollback and all required coupling. Once a certified endpoint exists, use native-import-115/geometry-verification-command.json with newly pinned run paths, then extend to16 and60 simulated seconds. Do not claim0.1x or motion from this diagnostic.
