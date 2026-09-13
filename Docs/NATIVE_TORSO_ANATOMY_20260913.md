# Native torso anatomy visual admission

The physical Apple M4 Pro run now admits the exact BodyParts3D torso source
surfaces through the native MyoSim visual owner. The payload is generated from
the pinned `partof_BP3D_4.0_obj_99.zip` archive and the existing source-to-world
registration (`6a48e223`), with 12 hash-bound surfaces: five organs, six
vessels, and one spinal-cord surface.

The native loader now validates sparse authored surface IDs with a sized lookup
and the camera gate aggregates source visibility across the selected camera
family. This preserves strict payload admission while allowing a valid nerve
surface to be occluded in individual views. The physical M4 Pro probe rendered
all 12 source surfaces across front, oblique, side, and rear views at the
canonical `12.5 us` clock with one deterministic step, activation `1.0`, the
authored support-contact payload, and no root assistance. The focused vascular
selection passed `4/4` CTest cases.

The hash-bound run, payload, frames, native identity, and CTest log are retained
in [`Docs/media/native-torso-anatomy-20260913/`](media/native-torso-anatomy-20260913/)
and summarized by [`receipt.json`](media/native-torso-anatomy-20260913/receipt.json).
The native source revision is `de83ba9bbde08fee921c1a2ff1e567f981714326`.

This closes source membership, source-to-world visual registration, and
kinematic visual binding only. It does not establish organ FEM/MPM, vessel tube
or lumen mechanics, neural mechanics, materials, density, blood mass,
pressure-driven momentum transfer, tissue exchange, subject calibration,
anatomical loading, standing, recovery, or walking.
