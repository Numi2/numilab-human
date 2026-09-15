# AddBiomechanics subject acquisition

This increment binds one real subject-level AddBiomechanics B3D artifact to the
Numi Human calibration input boundary. The dataset's [official download page](https://www.addbiomechanics.org/download_data.html)
describes the core release as CC BY 4.0 and provides individually scaled
skeletons with measured ground-contact forces and moments and estimated joint
kinematics and torques. The acquisition command records that provenance before
any fitting or Numi prediction is admitted.

The selected archive member is
`train/With_Arm/Falisse2017_Formatted_With_Arm/subject_1/subject_1.b3d` from
the aggregate archive. Its compressed member is 7,041,122 bytes with SHA-256
`bd31e6d125cb354cef761fcb2ff8153a67eab6ffbd2d41ce4829463892905c40`. The
decompressed subject artifact is 13,445,904 bytes with SHA-256
`2d1f9eb4c9173989dac5dd0c2cfd1110693a76a5bfc4e85c1aafab713d3ce0af`; the
ignored local copy is
`Sources/addbiomechanics/Falisse2017_subject_1.b3d`. The pinned metadata is a
43-year-old male, 1.78 m, 65.5 kg, with the source href retained in the
receipt.

Run the source-bound extraction from the repository root:

```sh
.numi/commands/human-addbiomechanics-acquire \
  --source Sources/addbiomechanics/Falisse2017_subject_1.b3d \
  --references Docs/media/addbiomechanics-falisse-20260915/references \
  --output Docs/media/addbiomechanics-falisse-20260915/acquisition-receipt.json
```

The immutable receipt and four extracted CSV tables are under
`Docs/media/addbiomechanics-falisse-20260915/`. They contain the four source
trials `Gait_5_segment_0`, `Gait_7_segment_0`, `StairUp_4_segment_0`, and
`StairUp_5_segment_0`, with pelvis tilt, right hip/knee/ankle angles, total
vertical ground reaction, and right knee moment. The source has walking and
stair trials, but no standing or recovery trial.

This closes source acquisition and measured-reference extraction for one adult
male. It does not close the Numi comparison: no Numi prediction tables,
calibration/held-out residuals, uncertainty estimate, activation recruitment,
anatomical support loading, material identification, organ/blood/tissue/fat
mechanical ownership, or standing/walking qualification is asserted. The
existing validation compiler remains fail-closed until a Numi runtime produces
source-revision-bound predictions on disjoint trials.
