# Registered passive viscera

The SSH Mac mini compiled 66 additional unique BodyParts3D surfaces into the
existing NHANAT ABI5 scene: complete source families for small and large bowel,
gallbladder, urinary bladder, and prostate. FJ2599 (ileocecal junction) belongs
to both bowel families and is serialized once with both source incidences.
The initial authoring attempt rejected that shared membership before writing
an output. The corrected author preserves it explicitly without duplicating
geometry.

The compiler reuses `organ_family_geometry.compiled_surface`, the pinned atlas
archives, existing whole-body reference frames, and current registration. Bowel
and gallbladder follow the abdominal frame; rectum, bladder, and prostate follow
the pelvic frame. They remain passive inspection anatomy, add no physical mass,
and have no digestive, urinary, endocrine, or reproductive physiology.

The candidate retains every prior record, vertex, normal, and index byte exactly.
All added source faces match their source OBJs. Reconstructed native Float32
world coordinates differ from the declared source transform by at most
8.28 nanometres. This establishes compilation/registration consistency, not
loaded interface or biological validity.

Payload SHA-256: `10c847decea932fd4e1fb486d5dbc40abbe33141604046573f834edc1dd54902`.
463 surfaces, 1,113,755 vertices, 5,157,570 indices. The immutable candidate is
retained on the Mini at
`/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/output/passive-viscera-candidate-002`.
It is based on the repaired lung candidate and still needs the final cardiac
derivation merge and complete native-scene inspection.

The receipt also maps the existing 21 CVSim compartment identities to registered
major vessel/chamber surfaces or explicitly reduced regional vascular beds.
Displayed geometry is not an additional blood-volume owner. `append-audit.json`
binds the source files, byte-preservation checks, and every added surface.
BodyParts3D attribution and CC-BY-4.0 rights are retained in the receipt.
