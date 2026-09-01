# MyoSim Extensor Hood v2

`numilab-human myosim-build` emits
`myosim-fullbody-extensor-hood.nhhood` alongside the rigid, muscle, contact,
and joint-equality payloads. The `NHHOOD2` artifact compiles eight independent
extensor mechanisms: digits 2-5 on both hands. This replaces the fifth-ray-only
artifact without adding opaque joint springs or copying the little finger's
extensor digiti minimi branch into digits where it does not exist.

## Evidence boundary

Each ray record explicitly names its side and digit. Each node preserves an
exact compiled MyoSim site ID, site name, source body,
Core body index, and COM-frame position. Each muscle input also preserves the
exact MyoSim muscle ID, route-cut ordinal, and preceding route
site used to recover its proximal force direction. The compiler fails closed if a required
site, muscle, route predecessor, body binding, ray identity, or bilateral
topology is absent. Digits 2-4 each bind EDC, radial interosseous, ulnar
interosseous, and lumbrical inputs. Digit 5 additionally binds EDM.

MyoSim does not segment the extensor expansion. The medial and lateral bands,
sagittal-band anchor, terminal junction, intercrossing fibres, free-node roles,
areas, moduli, and rest-length scales are therefore explicitly labelled
`literature_topology_inference`. Their ranges follow the open extensor-hood
fibre-network reference linked in the generated manifest; they are not claimed
as subject-specific measurements.

## Binary layout

The little-endian payload contains one header, eight side/digit ray records, 84
node records, 100 tension-element records, and 34 muscle-input records. The header
embeds the pinned MyoSim archive SHA-256 and fixed record sizes. Runtime code
must reject a mismatched magic, ABI, record size, payload length, or source
hash before constructing a tension network.

This artifact is a source-posed mechanics input. It does not by itself prove a
loaded hand rollout, subject-specific hood calibration, deformable fascia, or
whole-body equilibrium; those require native runtime qualification.
