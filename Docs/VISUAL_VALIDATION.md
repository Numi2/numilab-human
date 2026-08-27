# Source-static visual validation

For the active MyoSim full-body muscle source, see the newer
[visual-progress record](VISUAL_PROGRESS.md). Its three Apache-2.0-derived
frames are source-rendered and are intentionally kept distinct from the
BodyParts3D source-static evidence below.

## BodyParts3D full-skin preview — 2026-08-26

The visual preview begins with the exact `FJ2810` skin OBJ from
`isa_BP3D_4.0_obj_99.zip`. It is converted only from source millimetres to
preview metres; it is not registered to a Rajagopal body frame or the Core
FunctionBased state. This deliberately keeps visual-source validation separate
from the device-qualified skeleton and Millard-actuation path.

| Item | Value |
| --- | --- |
| BodyParts3D archive SHA-256 | `40665852c49f218326590e204db91064a1ecfc3c6f8cbd7bbbcaac62c7cd409e` |
| Source OBJ member / SHA-256 | `isa_BP3D_4.0_obj_99/FJ2810.obj` / `682f402206f15592acdeaae8ffb6b34c3e5c3267fa4685e63d2e4920ef2a80e0` |
| Source surface | 102,467 vertices; 203,382 triangles |
| Core revision | `14c64f303adb713f3a011546908688adb5848c61` (`origin/coupled`) |
| Cooked `.mrvpack` SHA-256 | `5b78d852357ea3cbfe44a9d0d55fb9a68251b8971d3bdfa4a0806750267063b9` |
| Visual probe SHA-256 | `7cb286f927a4d31d323d90a87d748fb4e3678945e0fd9838c647bb25b04f2b0d` |
| Core library SHA-256 | `08d548091af84d460f1c326cf3b7cc6b67fa89b2fbd266ed9b585d1a69a0d59d` |
| Metal library SHA-256 | `8351863cbbf5ce523956d9b49484ae39c315f9b8bdec54285544bcffaff71922` |
| GPU | Apple M4 Pro on `macmini` |
| Render profile | `sensor_reference`, 512 × 512, one static environment |

The checked final views are an anterior-looking `axis_negative_y` view, an
upright side-oblique view, and a posterior-looking `axis_positive_y` view.
Their BodyParts3D source-pixel counts were respectively 13,045, 8,455, and
13,345. The final PPM SHA-256 values were, in that order,
`c95035994447b26cf18dc03a81e9a9ed519e7d63f787effffd7acf4c0351337a`,
`5e0120ee16039f09e5581f8d65a37235febbc75e495705665e70850b3ac38011`, and
`6e0c0fb5fe9ed8d7161c0e47cdaa80cc8baf556fcfdda2e74600b6e32a659bcd`.
The device log SHA-256 is
`21a5d9f344d5384c9ca1c4aba80eba0b6c8cce4a583683ad7f6beee4f2d37c1e`.

The front and rear contours, limbs, hands, head, and feet were visible and
upright in the inspected frames. The first camera-basis implementation exposed
roll instability in oblique and opposite-axis views; the final render uses a
world-up-preserving basis and was rechecked at all three angles.

These frames are retained outside Git because they are derived from a
third-party BodyParts3D geometry source. They validate source-surface cooking,
camera framing, and renderer visibility only. They do **not** validate anatomy
registration, skinned deformation, joint motion, muscles, collision, contact,
organ/vessel mechanics, tissue material parameters, or a full Human RobotPack.

## Source-static anatomy-layer previews — 2026-08-26

`numi human visual-layers` selected the largest exact source mesh for each
requested layer, then each GLB was cooked and rendered by
`metalrobo_bodyparts3d_visual_probe` on the Apple M4 Pro. All three stable
inspection cameras contained source pixels for every layer.

| Layer | Source mesh | Vertices / triangles | Front / oblique / rear pixels |
| --- | --- | --- | --- |
| Skin | `FJ2810` | 102,467 / 203,382 | 13,045 / 8,455 / 13,345 |
| Bone | `FJ1368` | 20,582 / 39,524 | 14 / 9 / 14 |
| Muscle | `FJ1451` | 85,775 / 98,928 | 507 / 488 / 538 |
| Vessel | `FJ2145` | 15,163 / 30,036 | 58 / 51 / 67 |
| Nerve | `FJ1806` | 15,757 / 26,512 | 110 / 155 / 120 |

This used the isolated Core `14c64f3` visual worktree and its M4 Pro probe.
The packs and PPM captures are retained outside Git at
`/Users/n/numilab-human-layer-renders-2d570d5-v2`; they are third-party
geometry-derived evidence. This validation establishes only source-static
layer visibility from three angles. It does not establish an anatomical
attachment transform, skinned motion, collision/contact, tissue mechanics, or
a walking Human.
