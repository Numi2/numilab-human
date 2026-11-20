# Native mass transfer after the constitutive refit

The four exact Float32 node changes preserve the original element
indices. They change the runtime reference volume of 49 cells. Exact
reprojection of the moved nodes preserves their typed P1 ancestry in
both materials. Eight joined-node records, four inner-interface
registration rows, and 27 carrier distance witnesses were regenerated.
Only the affected surface normals were updated. The baseline source
identities, donor quotas, coordinate frames and layer material bounds
remain in force.

The mass correction uses the existing compensated flow solver and
the already-positive carrier graph. It conserves 13.939678963460478 kg
with zero density outliers over 1,448,804 cells. The exact native
checker replayed 5,910,093 expanded carrier rows and its production
role writer and both role importers passed. The limiting fat cell
now receives about 850 kg/m³ at its new volume. The P1 and role
receipts pin the derived files.

The first native attempt, 019, installed the refitted layers and
admitted all 34 ligament source rows. It exited before step 1 on the
native 2 GiB reserve plus pack-size check. Already superseded derived
sidecars and capture payloads were losslessly compacted by hashing
their independently decompressed bytes before retiring the plain
copies. The compaction receipts preserve their restore paths.
Trial 020 uses the same physical inputs and retains the storage check.

## Replay

Apply the supplied patches to the already published native parent
mass checker launch and guarded solver launch. Keep the exact prior
reference, projection, role and source assets in the SSH Numi
workspace. The refit node patch and source geometry receipt are in
the parent directory. The pinned C++ reference-volume reader uses
the native owner. Run the volume reader, local reflow, P1 and
registration witness scripts in that order. The native mass replay
requires an existing empty role output directory and uses
`NUMI_HUMAN_CONSTITUTIVE_REFIT=1` with the existing reference-floor
full replay. Run the guarded scene launch with `--constitutive-refit`
and the work settings pinned in the launch receipt. No tolerances or
force owners were changed.

## Native result

Trial 020 accepted 12 full-scene steps with the refit and mass replay.
It advanced 0.187500009 ms in 326.137 s end to end (5.7491e-7×,
with viewer presentation disabled). Accepted GPU command time was
142.346 s (1.3172e-6×). The matched predictor control 018 took
179.281 s of GPU command time for 12 accepts. Used Krylov columns
declined from 3,248 to 2,224 over this prefix. This is one matched
short comparison at the same timestep and solver limits.

The thirteenth attempt still failed equilibrium. The original
fascia object 13 accepted a full local trial fraction with a
minimum determinant of 0.999998 on the last Newton pass. The
final failure reports object 19. The certified mechanical residual
was 5.62767e-4, and the vascular domain residual was 5.03781e-8.
The candidate exhausted 16 Newton passes and 1,024 Krylov columns.
The published clocks stayed at step 12. This bounded fascia capture
does not establish whether the prior fat witness still limits the
coupled correction.

Variable pleural-pressure coupling was enabled. The accepted
samples span only 0.1875 ms and all show −490.332489 Pa at output
precision. The terminal body remained in initial release, with
zero measured bed support force and no root assistance. This
trial contains no independent accepted mesh crossing audit.
The 16-second settled motion, zero-crossing, and 0.1× targets
remain open.

The bounded follow-up 021 produced a byte-identical accepted
physiology/body trace for the 12-step prefix. On the rejected root,
the refitted skin/fat shell (object 15) accepted a full material
trial on the last Newton pass with minimum J = 0.813881 and zero
invalid tiles. Contact slot 2222094 limited the coupled fraction
to 0.00202934. Vascular line search did not shorten that fraction.
The contact bound fell from 0.160126 on the initial pass to 0.00202934
by the end. The mechanical and vascular domain norms reproduced 020.
The first reported failing object was 16 in 021 and 19 in 020 despite
the same terminal residual and correction. The reproducible blocking
witness is the skin/fat contact row, rather than the reported organ
index. The raw region counts and exact object layout are retained in
the limiter summary.
