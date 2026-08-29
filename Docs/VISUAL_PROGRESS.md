# NumiLab Human visual progress

## Left toe enthesis identity and force transfer — 2026-08-29

<p align="center">
  <img src="media/numi-human-toe-enthesis-v5-2048/clean/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-153-front.png" width="24%" alt="Corrected left toe surfaces, front" />
  <img src="media/numi-human-toe-enthesis-v5-2048/clean/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-153-oblique.png" width="24%" alt="Corrected left toe surfaces, oblique" />
  <img src="media/numi-human-toe-enthesis-v5-2048/clean/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-153-side.png" width="24%" alt="Corrected left toe surfaces, side" />
  <img src="media/numi-human-toe-enthesis-v5-2048/clean/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-153-rear.png" width="24%" alt="Corrected left toe surfaces, rear" />
</p>

The five BodyParts3D toe chains were intact; the defect came from interpreting
one source EDL/FDL route as one toe even though each muscle surface has four
lesser-toe slips. The v7 visual payload locks complete distal bands to the
correct toe frame, while the v5 `NHTENDON2` payload distributes the unchanged
lumped endpoint wrench across the exact distal phalanges of toes 2--5. Hallux
EHL/FHL routes remain hallux-only.

<p align="center">
  <img src="media/numi-human-toe-enthesis-v5-2048/edl/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-source-route-centrelines-tendon-attachment-envelopes-focus-body-153-front.png" width="24%" alt="Left EDL four-toe envelope, front" />
  <img src="media/numi-human-toe-enthesis-v5-2048/edl/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-source-route-centrelines-tendon-attachment-envelopes-focus-body-153-oblique.png" width="24%" alt="Left EDL four-toe envelope, oblique" />
  <img src="media/numi-human-toe-enthesis-v5-2048/edl/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-source-route-centrelines-tendon-attachment-envelopes-focus-body-153-side.png" width="24%" alt="Left EDL four-toe envelope, side" />
  <img src="media/numi-human-toe-enthesis-v5-2048/edl/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-source-route-centrelines-tendon-attachment-envelopes-focus-body-153-rear.png" width="24%" alt="Left EDL four-toe envelope, rear" />
</p>

The cyan line is the unchanged source route; the warm four-node footprint is
the live force-transfer envelope. The isolated
[FDL views](media/numi-human-toe-enthesis-v5-2048/fdl/) check the plantar path
from the same four angles. The Apple M4 Pro 64+64-step replay executes 304
distributed envelopes and 528 source-point fallbacks per step, with bitwise
replay and zero endpoint migration. See the concise
[toe enthesis record](TOE_ENTHESIS_V5.md) for identities, counters, and the
remaining one-rigid-toes-body boundary.

## Distal-chain continuity and final M4 Pro replay — 2026-08-29

<p align="center">
  <img src="media/numi-human-distal-continuity-v4/fullbody/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-front.png" width="24%" alt="Final muscle-driven Human front" />
  <img src="media/numi-human-distal-continuity-v4/fullbody/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-oblique.png" width="24%" alt="Final muscle-driven Human oblique" />
  <img src="media/numi-human-distal-continuity-v4/fullbody/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-side.png" width="24%" alt="Final muscle-driven Human side" />
  <img src="media/numi-human-distal-continuity-v4/fullbody/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-rear.png" width="24%" alt="Final muscle-driven Human rear" />
</p>

Human `37fab88` and runtime
`45fede450ba889b8feb1df0a8330db3c31706497` close the reported floating-hand
defect without moving an authored muscle site. Attachment-refined parent bones
remain authoritative; unsupported thumb and distal-finger meshes inherit the
exact displacement to their parent from the coherent BodyParts3D common frame.
The resulting transformed mesh gaps are 0.5/0.3 mm at the right thumb,
0.4/0.1 mm at the left thumb, and 0.3--0.7 mm at the corrected distal finger
joints. Compact unsupported carpals retain the narrower same-side body-local
fallback. All exact source triangles, scale, and orientation are preserved.

<p align="center">
  <img src="media/numi-human-distal-continuity-v4/right-hand/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-57-side.png" width="24%" alt="Corrected right hand side" />
  <img src="media/numi-human-distal-continuity-v4/left-hand/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-107-oblique.png" width="24%" alt="Corrected left hand oblique" />
  <img src="media/numi-human-distal-continuity-v4/left-foot/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-153-oblique.png" width="24%" alt="Final left foot oblique" />
  <img src="media/numi-human-distal-continuity-v4/left-foot/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-153-side.png" width="24%" alt="Final left foot side" />
</p>

The final Apple M4 Pro run uses the v4 bone, v6 `NHTISS4`, and v4
`NHTENDON2` payloads. Sixty-four assisted and 64 assistance-removed steps at
100 µs reevaluate all 416 routes and apply 106,496 endpoint transfers. The
maximum force/moment residuals are `1.72633488546e-4 N` and
`2.44306352215e-6 N m`; one-step q/v parity errors are
`3.90537220115e-8` and `3.90584484736e-4`; replay is bitwise. The complete
[qualification transcript](media/numi-human-distal-continuity-v4/qualification.transcript.txt)
owns those counters.

The left-foot report was checked from front, oblique, side, and rear at the
same final state. Its bone chain is continuous, and the largest left/right
ankle/subtalar/MTP difference is `8.65e-4 rad`; there is no pathological
left-only motion in this bounded replay. The visible distal red strands are
exact BodyParts3D muscle surfaces with `NHTISS4` kinematic bindings. They are
not a deformable tendon continuum. MyoSim provides one articulated `toes`
segment per side, so the individual source toe bones cannot yet move
independently. Likewise, 296 endpoint records have admitted four-node surface
envelopes while 536 retain exact body-owned source-point transfer. That is a
real tendon force-transfer law, but only the admitted envelopes are literal
surface-distributed enthesis candidates.

The compiler reports `balanced=false`; this 12.8 ms device transaction is not
static balance, gait, photorealistic skin, independently driven fingers/toes,
or clinical anatomy. The purpose of these clean views is to show the corrected
source geometry without the cyan diagnostic overlay while keeping that
boundary explicit.

## Per-step tendon-load transaction — 2026-08-28

<p align="center">
  <img src="media/numi-human-tendon-step-transaction-v3-2048/anconeus/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-42-front.png" width="24%" alt="Persistent anconeus tendon load transaction, front" />
  <img src="media/numi-human-tendon-step-transaction-v3-2048/anconeus/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-42-oblique.png" width="24%" alt="Persistent anconeus tendon load transaction, oblique" />
  <img src="media/numi-human-tendon-step-transaction-v3-2048/anconeus/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-42-side.png" width="24%" alt="Persistent anconeus tendon load transaction, side" />
  <img src="media/numi-human-tendon-step-transaction-v3-2048/anconeus/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-source-route-centrelines-tendon-attachment-envelopes-focus-body-42-rear.png" width="24%" alt="Persistent anconeus tendon load transaction, rear" />
</p>

These are the first retained attachment views produced by the persistent Human
horizon rather than a one-step reference pose. Eight assisted and eight
zero-root-wrench steps generate 13,312 accepted terminal-load records while all
416 routes are reevaluated on Apple M4 Pro. The same-command-buffer consumer,
rollback, no-direct-torque identity, and bitwise replay gates pass. All four
views retain nonzero four-node envelope coverage and manual inspection found no
floating endpoint. See the [transaction record](HUMAN_TENDON_STEP_TRANSACTION.md).

## Tendon attachment v2 — 2026-08-28

<p align="center">
  <img src="media/numi-human-tendon-attachment-v2-2048/anconeus/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-tendon-attachment-envelopes-focus-body-42-front.png" width="24%" alt="Muscle-driven anconeus attachment v2, front" />
  <img src="media/numi-human-tendon-attachment-v2-2048/anconeus/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-tendon-attachment-envelopes-focus-body-42-oblique.png" width="24%" alt="Muscle-driven anconeus attachment v2, oblique" />
  <img src="media/numi-human-tendon-attachment-v2-2048/anconeus/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-tendon-attachment-envelopes-focus-body-42-side.png" width="24%" alt="Muscle-driven anconeus attachment v2, side" />
  <img src="media/numi-human-tendon-attachment-v2-2048/anconeus/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-tendon-attachment-envelopes-focus-body-42-rear.png" width="24%" alt="Muscle-driven anconeus attachment v2, rear" />
</p>

This current mechanical-anatomy lead shows exact BodyParts3D right anconeus
surface stable ID 99, humerus/ulna owners 41/42, and source actuator 228. Both
source endpoints pass the v2 envelope gates. Warm strands connect the unchanged
source terminal to its four actual transfer nodes, and the warm footprint joins
those nodes on the bone surface. Cyan remains a deliberately separate route
diagnostic. No source-proximity cap, triangle migration, or visual collar is
used. The front/oblique/side/rear frames retain 182/446/940/1,160 attachment
pixels.

<p align="center">
  <img src="media/numi-human-tendon-attachment-v2-2048/subscapularis/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-tendon-attachment-envelopes-focus-body-34-front.png" width="24%" alt="Muscle-driven subscapularis attachment v2, front" />
  <img src="media/numi-human-tendon-attachment-v2-2048/subscapularis/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-tendon-attachment-envelopes-focus-body-34-oblique.png" width="24%" alt="Muscle-driven subscapularis attachment v2, oblique" />
  <img src="media/numi-human-tendon-attachment-v2-2048/subscapularis/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-tendon-attachment-envelopes-focus-body-34-side.png" width="24%" alt="Muscle-driven subscapularis attachment v2, side" />
  <img src="media/numi-human-tendon-attachment-v2-2048/subscapularis/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-source-route-centrelines-tendon-attachment-envelopes-focus-body-34-rear.png" width="24%" alt="Muscle-driven subscapularis attachment v2, rear" />
</p>

The second check uses exact BodyParts3D right subscapularis stable ID 85,
owners 41/34, and source actuator 215. It exercises a six-segment route with
one source wrap and admitted envelopes at both endpoints. Coverage is
443/287/106/1,463 pixels. Manual review of all eight images confirms that the
surface footprint stays on the named bone and that occlusion changes naturally
across the cameras rather than making a floating connection appear valid.

Both Mac-mini captures excite only their selected source at `0.2` for one
100 µs update while Apple Metal evaluates all 416 paths. The images show the
loaded transfer program; the separate [all-endpoint reference transcripts](media/numi-human-tendon-attachment-v2-2048/reference/)
execute its forces. See the [capture records and hashes](media/numi-human-tendon-attachment-v2-2048/).
This is exposed mechanics evidence, not a finished skin render, deformable
tendon continuum, clinical enthesis map, contact result, or gait validation.

## Retired tendon visual lead — 2026-08-27

The prior Z-Anatomy close-up is retained as a diagnostic record, not a visual
lead. Review found that its closed tendon cap still read as a detached surface
against the calcaneus, and its composition did not meet the presentation bar.
Its visual-only mesh repair is no longer used as evidence for mechanics; the
reviewed `NHTENDON2` source-point and four-node program above is the current
attachment diagnostic. Neither the archived mesh nor v2 establishes a
deformable tendon continuum, contact, gait, or clinical attachment.

## Native source-bound torso anatomy — 2026-08-27

<p align="center">
  <img src="media/myosim-native-torso-anatomy-2048/myosim-fullbody-articulated-bodyparts-bones-source-torso-anatomy-muscle-driven-focus-body-20-front.png" width="24%" alt="Native source-bound torso anatomy, front" />
  <img src="media/myosim-native-torso-anatomy-2048/myosim-fullbody-articulated-bodyparts-bones-source-torso-anatomy-muscle-driven-focus-body-20-oblique.png" width="24%" alt="Native source-bound torso anatomy, oblique" />
  <img src="media/myosim-native-torso-anatomy-2048/myosim-fullbody-articulated-bodyparts-bones-source-torso-anatomy-muscle-driven-focus-body-20-side.png" width="24%" alt="Native source-bound torso anatomy, side" />
  <img src="media/myosim-native-torso-anatomy-2048/myosim-fullbody-articulated-bodyparts-bones-source-torso-anatomy-muscle-driven-focus-body-20-rear.png" width="24%" alt="Native source-bound torso anatomy, rear" />
</p>

`NHANAT1` is the native C++/Metal input for a deliberately bounded, exposed
torso layer. It retains 12 exact BodyParts3D 4.0 OBJ components (21,648
vertices / 40,410 triangles): a selected heart-group component, stomach,
pancreas, both kidneys, ascending/arch/descending/abdominal aorta, superior
and inferior vena cava, and spinal cord. The offline importer places each
source triangle set in the registered default inertial frame of MyoSim `torso`
or `Abdomen`; the native renderer binds it to that articulated link.

The local Apple M4 rendered the four 2048 px frames after one 100 µs,
all-416-path MyoSim update at activation 0.05. Organ/vessel/nerve coverage was
110,361/41,719/115 (front), 79,523/35,101/21 (oblique),
54,673/21,719/191 (side), and 28,838/2,476/138 (rear). Visual review confirms
readable placement from every angle; the [capture record](media/myosim-native-torso-anatomy-2048/capture.transcript.txt)
has the full native transcript.

This is source-surface kinematic presentation only. It does not infer missing
heart/lung/liver components, create deformable organs, compliant vessels or
neural mechanics, introduce material parameters, or establish collision,
contact, force transfer, clinical registration, or photorealism.

## Retained BodyParts3D calcaneal correspondence evidence — 2026-08-27

This replaces the generated tendon-to-bone collar in the source correspondence
check, but it is not gallery imagery.
The exact BodyParts3D `FJ1405` calcaneal-tendon topology now uses the same
per-anchor calcaneus registration as the rendered `FJ3360` bone. Its 944
distal lock vertices and 26-vertex feather band are projected directly onto
the named calcaneus triangles with a 0.35 mm exterior offset. In particular,
the oblique, side, and rear views show the visible tendon ending on the actual
source bone instead of ending at a synthetic quad collar or drifting from the
independently registered bone frame.

The 2K Apple M4 capture applies a 100 µs, 0.5 activation pulse to MyoSim
`gaslat_r` / `gasmed_r` / `soleus_r` (indices 348 / 349 / 369), while Metal
evaluates all 416 authored source routes. It has nonzero muscle, tendon, and
calcaneus coverage in all four views and zero generated collar pixels. The
[capture record](media/myosim-native-calcaneal-attachment-2048/capture.transcript.txt)
keeps the native execution and the intentional boundary. Its low-detail source
framing and visible source discontinuity are not accepted as a tendon-quality
showcase; use the smooth-insertion visual lead above for attachment inspection.
This remains a source-surface ownership check—not a claim of photorealistic
skin, deformable tendon, force transfer, collision/contact, gait, or clinical
validation.

## Retired smooth-insertion right-calf diagnostic — 2026-08-27

The archived Z-Anatomy capture contains a deterministic CC-BY-SA derivative
of the source tendon: one Catmull-Clark evaluation level (13,049 vertices /
26,090 triangles) and an 8 mm source-frame inset. It is retained for source
provenance, not presented as a successful attachment visual. The current
importer adds a separate 1.5 mm depth-tested interior enthesis inset at the
named calcaneus triangles; that corrected result is awaiting a clean remote
high-resolution review before it is published.

Only in this five-surface inspection, `Calcaneus.r` replaces the visible
BodyParts3D `FJ3360` mesh and remains rigidly bound to MyoSim/Core `calcn_r`
body `138`. The tendon retains copied femur `131`, tibia `136`, and calcaneus
`138` visual weights. This changes no MyoSim path, tendon, body, or force
parameter and creates neither an enthesis bridge nor a tendon continuum.

The local Apple M4 capture excites MyoSim `348`, `349`, and `369` at `0.5` for
one 100 µs step. Metal evaluates all 416 source paths in two transactions (90
wraps); the bounded configuration displacement is `0.000123820755509`. Every
2048 px view has nonzero bone, muscle, and tendon coverage. The [capture
record](media/myosim-native-zanatomy-smooth-insertion-2048/capture.transcript.txt)
contains the frames, hashes, source attribution, and exact execution boundary.

Both the archived diagnostic and the current source-derived correction remain
outside claims of photorealistic skin, deformable muscle/tendon, force
transfer, collision/contact, gait, or clinical attachment validation.

## Native passive-FEM soleus specimen — 2026-08-27

<p align="center">
  <img src="media/myosim-native-passive-fem-calf-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-passive-fem-tissue-muscle-driven-selected-actuators-focus-body-136-front.png" width="24%" alt="Source-bound passive soleus FEM, front" />
  <img src="media/myosim-native-passive-fem-calf-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-passive-fem-tissue-muscle-driven-selected-actuators-focus-body-136-oblique.png" width="24%" alt="Source-bound passive soleus FEM, oblique" />
  <img src="media/myosim-native-passive-fem-calf-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-passive-fem-tissue-muscle-driven-selected-actuators-focus-body-136-side.png" width="24%" alt="Source-bound passive soleus FEM, side" />
  <img src="media/myosim-native-passive-fem-calf-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-passive-fem-tissue-muscle-driven-selected-actuators-focus-body-136-rear.png" width="24%" alt="Source-bound passive soleus FEM, rear" />
</p>

This focused lower-leg specimen begins from exact BodyParts3D `FJ1437` soleus
surface geometry, then executes the same all-416-path MyoSim force update as
the anatomy view. Only source actuator `369` (right soleus) receives 0.5
activation for 64 × 100 µs; the final tibia (`136`) and calcaneus (`138`)
poses prescribe the two ends of a 12-node, nine-tetrahedron native Matter FEM
cage. Eight further 100 µs Matter steps completed on the local Apple M4 with
two peak FGMRES iterations, a minimum `J` of `0.900724887848`, 13.58 mm
prescribed-anchor travel, and 0.269 mm maximum free-node movement. The exact
source soleus surface is deformed by that native cage for all four inspected
views; the adjacent gastrocnemius and Achilles remain their named
source-surface bindings.

The [capture record](media/myosim-native-passive-fem-calf-2048/capture.transcript.txt)
has complete hashes, devices, coverage, and the non-qualification boundary.
This is a source-bound passive FEM specimen, not a realistic skin render,
active muscle stress law, calibrated muscle volume, tendon-to-bone
force-transfer law, whole-body deformable tissue, collision/contact, gait, or
clinical validation. The matte, separate BodyParts3D surfaces remain visible
by design so this evidence is not mistaken for a finished human presentation.

## Retired inferred exterior binding — 2026-08-27

The exact 102,467-vertex, 203,382-triangle BodyParts3D `FJ2810` shell is a
valuable static source mesh, but it has no authored per-vertex anatomical skin
weights. The `NHSKIN1` proximity-derived binding therefore remains diagnostic
only. A 2048-pixel all-416-muscle recheck found split/overlapping shell patches
in the oblique and rear views; reducing the proximity band or selecting only
the nearest bone merely changed where the discontinuities appeared.

The older exterior frames and their transcript are retained as rejected
diagnostic evidence, not linked as current gallery imagery or motion proof.
They do not establish articulated skin, a deformable tissue law,
collision shell, gait result, or clinical registration. The exposed source
muscle/bone/tendon captures below are the current muscle-driven presentation.

## Current full-body source-muscle and tendon review — isolated cameras — 2026-08-27

<p align="center">
  <img src="media/myosim-native-fullbody-isolated-cameras-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-front.png" width="24%" alt="All-muscle Human, front" />
  <img src="media/myosim-native-fullbody-isolated-cameras-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-oblique.png" width="24%" alt="All-muscle Human, oblique" />
  <img src="media/myosim-native-fullbody-isolated-cameras-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-side.png" width="24%" alt="All-muscle Human, side" />
  <img src="media/myosim-native-fullbody-isolated-cameras-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-rear.png" width="24%" alt="All-muscle Human, rear" />
</p>

This is the current whole-body mechanical anatomy reference: 184 exact
BodyParts3D bone meshes and 150 named muscle/tendon surfaces at 2048 px.
All 416 authored MyoSim routes ran on Apple M4 before one 100 µs bounded state
update at 0.01 activation (two Metal force transactions, 416 active records,
90 wraps, and `1.44013083483e-05` maximum configuration delta). It uses the
`NHTISS3` three-body Achilles binding. Each angle creates a fresh native
renderer and Metal world sample, eliminating the former prior-camera leakage.
The normal source tendon boundary is
triangle-projected onto its named calcaneus; the old short visual collar is
available only as an explicit diagnostic, so the renderer no longer adds a
synthetic muscle-to-tendon bridge. The
[capture record](media/myosim-native-fullbody-isolated-cameras-2048/capture.transcript.txt)
has the four image hashes, coverage, and devices.

This is an active muscle-force and articulated-anatomy inspection. The collar
does not create a weld, a tendon material law, contact, gait, or clinical
attachment validation.

## Shared three-body Achilles review — 2026-08-27

<p align="center">
  <img src="media/myosim-native-three-body-achilles-2048/rest/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-front.png" width="24%" alt="Three-body Achilles, front" />
  <img src="media/myosim-native-three-body-achilles-2048/rest/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-oblique.png" width="24%" alt="Three-body Achilles, oblique" />
  <img src="media/myosim-native-three-body-achilles-2048/rest/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-side.png" width="24%" alt="Three-body Achilles, side" />
  <img src="media/myosim-native-three-body-achilles-2048/rest/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-rear.png" width="24%" alt="Three-body Achilles, rear" />
</p>

The old Achilles proxy incorrectly used tibia/calcaneus ownership only. The
new native `NHTISS3` payload carries all source-route body owners: femur,
tibia, and calcaneus. It inherits proximal weights from the exact named
gastrocnemius and soleus source surfaces, then locks the distal source
triangles to the exact calcaneal surface (944 right and 943 left vertices,
with 26/25 feathered). The 2048 px Apple-M4 review has nonzero tendon coverage
in every angle: 1,715 / 4,329 / 9,407 / 21,648 pixels. The [capture record](media/myosim-native-three-body-achilles-2048/capture.transcript.txt)
contains provenance, hashes, and the separate selective contraction result.

The matching small contraction excites only `348`/`349`/`369` at 0.5 for one
100 µs step. Metal still evaluates all 416 source paths (two transactions;
416 records) before the bounded Core FP64 update. The cyan paths in that
separate diagnostic are source routes, not replacement tendon geometry.

This improves visual kinematic ownership of the exact BodyParts3D tendon mesh;
it does not add tendon material, a surface force-transfer law, a weld, contact,
gait, or clinical attachment validation.

## Selective posterior-calf source-actuator route review — 2026-08-27

<p align="center">
  <img src="media/myosim-native-selective-calf-route-attachment-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-surface-projected-sites-focus-body-136-front.png" width="24%" alt="Selective calf source route, front" />
  <img src="media/myosim-native-selective-calf-route-attachment-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-surface-projected-sites-focus-body-136-oblique.png" width="24%" alt="Selective calf source route, oblique" />
  <img src="media/myosim-native-selective-calf-route-attachment-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-surface-projected-sites-focus-body-136-side.png" width="24%" alt="Selective calf source route, side" />
  <img src="media/myosim-native-selective-calf-route-attachment-2048/myosim-fullbody-articulated-bodyparts-bones-muscle-driven-selected-actuators-source-support-contact-source-route-centrelines-surface-projected-sites-focus-body-136-rear.png" width="24%" alt="Selective calf source route, rear" />
</p>

This native 2048 px review is deliberately skeletal: it makes the true
source-force path legible instead of asking a separate BodyParts3D collagen
surface to prove force attachment. Actuators `348` (right lateral
gastrocnemius), `349` (right medial gastrocnemius), and `369` (right soleus)
alone received `0.5` excitation for 64 × 100 µs steps. Every one of the 416
authored MyoSim paths was still evaluated on the Apple M4, yielding 128 Metal
force transactions, 26,624 active records, 11 rendered route segments, and
six source-projected endpoint cues. The final bounded source-foot contact had
five active witnesses (six peak); the 157-body contact island remains Core FP64
because the current Metal contact bucket rejects it. The [capture record](media/myosim-native-selective-calf-route-attachment-2048/capture.transcript.txt)
retains exact counters and hashes.

The cyan geometry is the CPU FP64 resolved spatial route at that same final
pose, plus visual-only nearest-surface endpoint cues. It is not a collagen
mesh, tendon material, tendon-to-bone force-transfer law, deformation result,
stable gait, or clinical attachment validation.

## Retired multi-step articulated exterior shell — 2026-08-27

<p align="center">
  <img src="media/myosim-native-skinned-fullbody-metal-force-2048/myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell-muscle-driven-source-support-contact-front.png" width="24%" alt="Muscle-driven articulated Human exterior, front" />
  <img src="media/myosim-native-skinned-fullbody-metal-force-2048/myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell-muscle-driven-source-support-contact-oblique.png" width="24%" alt="Muscle-driven articulated Human exterior, oblique" />
  <img src="media/myosim-native-skinned-fullbody-metal-force-2048/myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell-muscle-driven-source-support-contact-side.png" width="24%" alt="Muscle-driven articulated Human exterior, side" />
  <img src="media/myosim-native-skinned-fullbody-metal-force-2048/myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell-muscle-driven-source-support-contact-rear.png" width="24%" alt="Muscle-driven articulated Human exterior, rear" />
</p>

This is a retained engineering artifact, not the current whole-human
presentation: the exact 102,467-vertex,
203,382-triangle BodyParts3D exterior mesh is posed natively after the
all-416-muscle force update. Each vertex carries four registered Core body
influences selected from 86 bone envelopes. The offline source import
reconstructs the registered rest pose with a maximum error of
`1.0111756560930368e-15 m`; native C++/Metal owns the final pose and render.

The Apple M4 2K run executed 32 × 100 µs updates, 64 Metal force
transactions, 13,312 active-muscle records, and 2,866 wrapped route contacts.
Its four frames have 485,845 / 426,364 / 308,186 / 516,832 nonzero skin pixels
(front / oblique / side / rear). The dynamic source-foot support probe retained
two contacts at the final step (six at peak); as with the exposed anatomy,
full-tree contact was correctly not admitted to the installed Metal bucket and
Core FP64 owns that bounded exact-cone fallback. The exact [capture record](media/myosim-native-skinned-fullbody-metal-force-2048/capture.transcript.txt)
retains parameters and output identities.

Its source-proximity weights visibly fold under this long unconstrained update,
so this gallery is retired from presentation. It remains an articulated visual
shell, not FEM/MPM skin, a skin material law, collision geometry, general
contact, gait, or clinical-registration evidence. The separate exposed-anatomy
and tendon views remain the source surface evidence; an opaque shell is never
used to imply tendon continuity.

## Retained 32-step full-body all-muscle Metal force inspection — 2026-08-27

<p align="center">
  <img src="media/myosim-native-fullbody-supported-metal-force-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-front.png" width="24%" alt="Ground-supported muscle-driven BodyParts3D Human, front" />
  <img src="media/myosim-native-fullbody-supported-metal-force-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-oblique.png" width="24%" alt="Ground-supported muscle-driven BodyParts3D Human, oblique" />
  <img src="media/myosim-native-fullbody-supported-metal-force-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-side.png" width="24%" alt="Ground-supported muscle-driven BodyParts3D Human, side" />
  <img src="media/myosim-native-fullbody-supported-metal-force-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-rear.png" width="24%" alt="Ground-supported muscle-driven BodyParts3D Human, rear" />
</p>

This is a retained longer-step exposed-anatomy presentation: 184 BodyParts3D bone meshes and 150
named muscle/tendon surfaces at 2048 × 2048 from front, oblique, side, and
rear. The posterior tendons are rendered with their associated muscle chain
in the full-body view rather than as an anatomically misleading free segment.

The capture executes 32 100 µs bounded updates. Per update, Metal evaluates
all 416 authored MyoSim routes, active and zero-activation sidecars, and the
summed 128-DoF force vector; Core FP64 then performs the currently supported
state step and its exact-cone support contact over ten source foot witnesses;
Metal then renders the resulting pose. The run reported 64 Metal force
transactions, 13,312 active-muscle records, 2,866 wrapped path contacts, and
a maximum active/passive configuration difference of 0.0278143121505. It
observed two active contacts in the final step (six at peak) and a minimum
plane gap of -9.81e-8 m. The four images have nonzero bone, muscle, and tendon
coverage. The concise [capture record](media/myosim-native-fullbody-supported-metal-force-2048/capture.transcript.txt)
contains the exact runtime boundary and output identities.

This establishes a bounded, muscle-driven articulated presentation. It does
not establish skinning, tendon force transfer, a tendon continuum, deformable
soft tissue, general collision, gait, or clinical registration. The current
157-body contact island is correctly reported as not admitted to the installed
Metal contact bucket; its bounded contact step is Core FP64 rather than a
claimed GPU contact result.

## Withdrawn calcaneal-tendon detail — 2026-08-27

The archived 2048 calcaneal detail images are not anatomy-quality evidence.
Review found a detached-looking source terminal cap and disconnected tendon
fragments; the current high-resolution exact-reference renderer also does not
produce valid whole-frame coverage. The files remain as reproducibility
artifacts, not gallery media.

The corrective source path retains only the tendon member's dominant connected
sheet, removes fully interior terminal-cap faces, and moves the terminal lock
band 5 mm inside the matching calcaneal triangles. A narrow, explicitly
inferred visual enthesis strip joins that opened source boundary to the named
calcaneal display surface; it is not represented as source geometry or a
mechanical weld. A
four-angle Apple-M4 640 px diagnostic check has nonzero bone/muscle/tendon
coverage. It remains outside the gallery because the exact-reference path is
not currently qualified above 640 px; historical 2K media remains withheld
from presentation. This is visual kinematic evidence only—not tendon
continuity, force transfer, deformable tissue, collision, gait, or clinical
registration.

## Native source-skin context — 2026-08-27

<p align="center">
  <img src="media/bodyparts3d-skin-source-reference-2048/axis_negative_y.png" width="32%" alt="BodyParts3D full-skin source, front" />
  <img src="media/bodyparts3d-skin-source-reference-2048/oblique_positive_x_negative_y.png" width="32%" alt="BodyParts3D full-skin source, oblique" />
  <img src="media/bodyparts3d-skin-source-reference-2048/axis_positive_y.png" width="32%" alt="BodyParts3D full-skin source, rear" />
</p>

The source skin is a materially better whole-human visual context than the
small bone/tendon inspection meshes: 102,467 vertices and 203,382 triangles
from exact BodyParts3D member `FJ2810`. The native C++/Metal preview now builds
a camera-relative three-point studio rig per view, avoiding the old fixed key
that overexposed the front and flattened the rear. The Apple M4 capture is
2048 × 2048 and retains 425,068, 277,323, and 440,136 covered pixels in its
front, oblique, and rear frames respectively; its transcript and output hashes
are retained beside the images.

This retained source-static reference establishes mesh provenance only. The
proximity-derived animated exterior binding is retired after its high-resolution
visual failure. Neither view provides contact geometry, constitutive parameters,
volume, or physical skin deformation.

## Device-resident full-body muscle-force projection — 2026-08-27

The visual captures below now have a stronger mechanical companion check. The
Apple M4 native probe evaluates the 416 authored MyoSim MuJoCo muscle routes,
their static actuator forces, each route's `J^T` generalized-force contribution,
and the deterministic full-body force reduction in one Metal command-buffer
sequence. It compares the results with the FP64 Core reference at the same
source-default pose: maximum per-muscle generalized-force error is
`0.00471758869298`, and the summed 128-DoF vector differs by
`0.00642352090836`.

This means the device no longer stops at a visual muscle route or scalar force:
its force projection follows the current source attachments and wrapped path
segments. The same command-buffer family now advances a non-equilibrium
416-muscle activation sidecar for one 100 µs explicit Metal step and returns
it with zero observed update error; this is a mechanical state transition, not
a visual attachment adjustment. The next boundary is deliberately explicit:
the 157-body/128-DoF Human exceeds the current Metal **dynamics** bucket, so
CPU FP64 still owns the bounded forward-dynamics state step and support
contact. This validation does not claim a persistent device-only full-body
integrator, tendon continuum, deformable soft tissue, gait, or clinical
anatomy.

### Current bounded visual-runtime path — 2026-08-27

The current native visual executable no longer projects 416 muscle paths in a
host loop. For each bounded muscle-driven step it submits both the requested
and zero-activation MyoSim sidecars to retained Apple-Metal contexts, consumes
their returned 128-DoF force vectors, and passes only the baseline-subtracted
force to the current Core FP64 state step. A local M4 four-angle 512 px,
two-step smoke completed four Metal force transactions and 832 active-muscle
records before producing every renderer frame.

This changes the live capture path, not the provenance of media already
published above. Those 2048 images retain their original transcripts and must
be recaptured before they can evidence the new Metal-force visual path. The
remaining state integration and support contact are still Core FP64; no tendon
continuum, deformable anatomy, or device-only Human rollout is claimed.

## Focused right upper-limb actuation — 2026-08-27

<p align="center">
  <img src="media/myosim-native-right-upper-limb-flexion-drive-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-focus-body-41-front.png" width="24%" alt="Right upper-limb source drive, front" />
  <img src="media/myosim-native-right-upper-limb-flexion-drive-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-focus-body-41-oblique.png" width="24%" alt="Right upper-limb source drive, oblique" />
  <img src="media/myosim-native-right-upper-limb-flexion-drive-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-focus-body-41-side.png" width="24%" alt="Right upper-limb source drive, side" />
  <img src="media/myosim-native-right-upper-limb-flexion-drive-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-selected-actuators-focus-body-41-rear.png" width="24%" alt="Right upper-limb source drive, rear" />
</p>

This native 2048 px inspection focuses on the torso, right clavicle, scapula,
humerus, ulna, and radius—not a separate animated arm. It shows 42 exact
BodyParts3D bone meshes and 20 selected source surfaces: pectoralis major,
deltoid, rotator cuff, teres, coracobrachialis, triceps, anconeus, biceps,
brachialis, and brachioradialis. The oblique view exposes the
shoulder/scapular relationship that a full-body frame hides.

The drive does not simply activate the whole model. It applies 0.2 excitation
to source indices `210`, `211`, `218`, `219`, `220`, `224`, `230`, `231`,
`232`, and `233`: pectoralis major, anterior/acromial deltoid,
coracobrachialis, both biceps heads, brachialis, and brachioradialis. All 416
authored paths remain evaluated on the Apple M4 in every step. The 64 × 100 µs
run recorded 128 Metal force transactions, 26,624 active records, 5,765 source
wrap applications, and an active/passive configuration difference of
`0.0446275454086`. The [capture transcript](media/myosim-native-right-upper-limb-flexion-drive-2048/capture.transcript.txt)
pins the source package and all output hashes.

This is source geometry through a bounded selective-muscle free-dynamics update. It
does not establish upper-limb contact, a deformable tendon or skin model,
surface-force transfer, stable movement, or clinical registration.

## Focused posterior-chain attachment and bounded muscle update — 2026-08-27

<p align="center">
  <img src="media/myosim-native-posterior-tendon-inspection-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-side.png" width="32%" alt="Right posterior-chain source geometry, side" />
  <img src="media/myosim-native-posterior-tendon-inspection-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-oblique.png" width="32%" alt="Right posterior-chain source geometry, oblique" />
  <img src="media/myosim-native-posterior-tendon-inspection-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-136-rear.png" width="32%" alt="Right posterior-chain after bounded muscle update, rear" />
</p>

The broad images below are retained as source coverage, not the current Human
presentation. This 2048 px inspection filters the native scene to 30 meshes
on six exact source bone parents (right femur, tibia/fibula, talus, calcaneus,
toes, and patella) plus four exact surfaces: right lateral gastrocnemius,
right medial gastrocnemius, right soleus, and right calcaneal tendon. The
side, oblique, and rear views expose the calcaneal insertion instead of
hiding it behind the full body.

The bounded comparison takes eight 100 microsecond steps. Each step evaluates
all 416 source muscle-tendon paths, reprojects the ten authored source-foot
witnesses against their plane, and activates the six witnesses that reach the
plane. The result reports a maximum configuration difference of
0.00106110731348 rad/m from the zero-activation baseline, six active support
contacts, and a minimum plane gap of -0.000000190044077062 m. Pose and
renderer execute on Apple M4; the current 157-body articulation is explicitly
not admitted to the installed Metal contact bucket, so contact is Core FP64
for this capture.

The [capture transcript](media/myosim-native-posterior-tendon-inspection-2048/capture.transcript.txt)
pins the exact inputs, options, output hashes, and limitations. This is visual
continuity of source geometry through a bounded update. It does not establish
a tendon continuum, tendon-to-bone force transfer, deformable tissue, stable
standing or gait, general collision, skin, or clinical registration.

## Muscle-driven tendon junction continuity — 2026-08-27

<p align="center">
  <img src="media/myosim-native-supported-tendon-junction-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-136-front.png" width="24%" alt="Driven tendon junction, front" />
  <img src="media/myosim-native-supported-tendon-junction-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-136-oblique.png" width="24%" alt="Driven tendon junction, oblique" />
  <img src="media/myosim-native-supported-tendon-junction-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-136-side.png" width="24%" alt="Driven tendon junction, side" />
  <img src="media/myosim-native-supported-tendon-junction-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-136-rear.png" width="24%" alt="Driven tendon junction, rear" />
</p>

This four-angle Apple-M4 2048 px check uses the 150-surface exact
BodyParts3D `NHTISS2` import in the same 32 × 100 µs all-416-muscle,
ground-supported pose as the broad gallery. A visual-only collar is emitted
only at an open source tendon boundary and only to the nearest visible muscle
sharing an authored endpoint or to the named secondary bone surface, within
30 mm. It closes the source-mesh raster gap through the final dynamic pose;
it does not alter the MyoSim route, activation, force, or Core dynamics.

The 2K frames and input/output identities are retained in the [capture record](media/myosim-native-supported-tendon-junction-2048/capture.transcript.txt).
This is a rendering continuity improvement, not a tendon weld, constitutive
model, force-transfer result, collision result, gait result, or clinical
attachment validation.

## Presentation correction — 2026-08-27

The earlier native route galleries below are retained only for renderer
regression and source-coverage evidence; they are **retired as anatomy
presentation**. They drew every MyoSim route as a straight line between sites
and wrap centres, which can cut through a sphere/cylinder wrap and visibly miss
an anatomical surface. They are not acceptable tendon imagery.

The current native renderer keeps routes hidden by default. Its opt-in focused
inspection starts and ends at exact source sites, uses the source solver's
tangent contacts, and samples the wrapped sphere/cylinder arc. Surface-anchor
caps make the projected origin/insertion points readable at the bone without
changing the force path. That remains an alignment diagnostic, not tendon
surface geometry or a medical registration. The reviewed showcase below uses
the separate exact BodyParts3D muscle/tendon surfaces instead of that route
diagnostic.

## Geometry-framed full-body source anatomy — 2026-08-27

<p align="center">
  <img src="media/myosim-native-fullbody-geometry-framed-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-front.png" width="32%" alt="Geometry-framed full BodyParts3D anatomy, front" />
  <img src="media/myosim-native-fullbody-geometry-framed-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-oblique.png" width="32%" alt="Geometry-framed full BodyParts3D anatomy, oblique" />
  <img src="media/myosim-native-fullbody-geometry-framed-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-rear.png" width="32%" alt="Geometry-framed full BodyParts3D anatomy, rear" />
</p>

The prior broad muscle-surface capture focused camera body 128, which made
the source anatomy read as an unhelpful torso close-up. The new native
`myosim-native-fullbody-soft-tissue-visuals` entry point builds the visual
pack first, evaluates its exact current world-space source geometry bounds,
and targets each camera at the rendered-vertex centroid. It is not a
camera-distance guess from rigid-body centres or an AABB midpoint in empty
oblique-view space.

The reviewed 2048 × 2048 M4 capture contains 184 BodyParts3D bone meshes and
150 source tissue surfaces (148 muscle, two calcaneal tendon). The source
geometry extent was 1.72125351429 m and the native camera distance is
1.85895383358 m. Every view has nonzero bone, muscle, and tendon coverage:

| View | Bone / muscle / tendon pixels |
| --- | ---: |
| Front | 56,105 / 240,748 / 101 |
| Oblique | 52,332 / 198,083 / 660 |
| Side | 41,616 / 118,125 / 619 |
| Rear | 79,396 / 215,071 / 5,502 |

The full [capture transcript](media/myosim-native-fullbody-geometry-framed-2048/capture.transcript.txt)
pins the rendered input hashes and output frames. This repairs presentation
framing only: it is a Metal articulated-pose snapshot with visual two-body
muscle/tendon surfaces, not a deformable-tissue solve, contact qualification,
controller, rollout, or medical registration.

## Triangle-locked calcaneal tendon inspection — 2026-08-27

<p align="center">
  <img src="media/myosim-native-calcaneal-tendon-triangle-lock-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-side.png" width="32%" alt="Triangle-locked right Achilles insertion, side" />
  <img src="media/myosim-native-calcaneal-tendon-triangle-lock-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-oblique.png" width="32%" alt="Triangle-locked right Achilles insertion, oblique" />
  <img src="media/myosim-native-calcaneal-tendon-triangle-lock-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-rear.png" width="32%" alt="Triangle-locked right Achilles insertion, rear" />
</p>

The old nearest-vertex test left sparsely tessellated areas of a tendon eligible
to blend with the wrong body even when they sat on the calcaneal surface. The
current package measures exact closest points to the named BodyParts3D
calcaneus triangles. Vertices within 3 mm are fully calcaneus-bound and the
3–15 mm band is feathered: 944 locked + 26 feathered on the right, and 943 +
25 on the left. The four native 2048 px views all contain bone, muscle, and
tendon pixels; the rear tendon coverage is 46,393 pixels. See the
[capture transcript](media/myosim-native-calcaneal-tendon-triangle-lock-2048/capture.transcript.txt).

This is a source-default Metal-pose inspection that prevents a visible
two-body presentation seam at the named insertion. It does not constitute a
tendon weld, force transfer, a constitutive model, deformable tissue, or a
medical attachment claim.

## Supported tendon attachment review — 2026-08-27

<p align="center">
  <img src="media/myosim-native-supported-posterior-chain-attachment-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-136-front.png" width="32%" alt="Supported posterior-chain attachment view, front" />
  <img src="media/myosim-native-supported-posterior-chain-attachment-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-136-side.png" width="32%" alt="Supported posterior-chain attachment view, side" />
  <img src="media/myosim-native-supported-posterior-chain-attachment-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-focus-body-136-rear.png" width="32%" alt="Supported posterior-chain attachment view, rear" />
</p>

These four inspected 2048 × 2048 Apple M4 frames replace the earlier
free-body posterior-chain stress picture. The active pose applies the
zero-baseline-subtracted 5% activation to all 416 MyoSim muscles for 1 ms,
then resolves two bilateral support witnesses from MyoSim's own authored foot
collision primitives in Core FP64 exact-cone contact before Metal produces the
final pose and native renderer frame.

The BodyParts3D calcaneal-tendon surfaces preserve their exact triangles, but
the generic two-body blend has been corrected where it matters: source vertices
within 3 mm of the named calcaneus mesh are fully calcaneus-bound and the next
12 mm is feathered. That locks 341 right and 492 left tendon vertices, with
629 and 476 feathered vertices. It prevents an active pose from visibly
pulling an insertion off the bone while retaining a smooth tendon surface.

| View | Bone / muscle / tendon pixels | Review |
| --- | ---: | --- |
| Front | 343,512 / 42,216 / 1,654 | insertion and anterior ankle remain visible |
| Oblique | 380,943 / 64,287 / 5,480 | lateral tendon contour remains continuous |
| Side | 304,884 / 45,081 / 6,373 | posterior insertion remains attached to calcaneus |
| Rear | 247,926 / 121,337 / 22,281 | complete calf → Achilles → heel chain is legible |

The companion opt-in route image makes the force-path ownership readable:

<p align="center">
  <img src="media/myosim-native-supported-posterior-chain-route-attachment-2048/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-source-support-contact-source-route-centrelines-surface-projected-sites-focus-body-136-rear.png" width="48%" alt="Cyan exact MyoSim gastrocnemius and soleus route attachment diagnostic" />
</p>

Its cyan curves are the posed MyoSim gastrocnemius lateralis, gastrocnemius
medialis, and soleus routes, including the source wrap arcs; cyan caps are
nearest registered-bone endpoint cues. They demonstrate the route-to-bone
relationship without pretending that a line is tendon geometry. The route is
naturally occluded in the lateral camera by the anatomy it passes beneath, so
the diagnostic now requires visibility in at least one reviewed angle rather
than silently failing valid anatomy views.

This is a bounded supported snapshot, not stable standing or gait. The
full-body Metal contact program explicitly rejected the 157-body connected
articulation as larger than its current dynamics bucket, so the capture uses
the native FP64 contact result and labels GPU contact as not admitted. It does
not claim general collision, a controller, repeated integration, deformable
tendon mechanics, force transfer from triangles, or clinical registration.

## Reviewed native full-body muscle-surface inspection — 2026-08-27

<p align="center">
  <img src="media/myosim-native-fullbody-muscle-surfaces-2048/default/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-128-front.png" width="32%" alt="Native full-body muscle surfaces, front" />
  <img src="media/myosim-native-fullbody-muscle-surfaces-2048/default/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-128-oblique.png" width="32%" alt="Native full-body muscle surfaces, oblique" />
  <img src="media/myosim-native-fullbody-muscle-surfaces-2048/default/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-128-rear.png" width="32%" alt="Native full-body muscle surfaces, rear" />
</p>

This earlier torso-focused capture is retained as a close surface inspection;
the geometry-framed capture above is now the main Human anatomical view. The
native Apple M4 capture poses 150 exact BodyParts3D surfaces over the 184-mesh
articulated skeleton: 148 muscle surfaces across both limbs,
shoulders, arms, forearms, hands, abdomen, pelvis, and legs; plus two
calcaneal tendon surfaces. The 19.9 MB `NHTISS2` package has 438,491 vertices
and 1,894,392 indices.

The offline package checks each source FJ member and FMA name against the
versioned map. For each ordinary muscle, it reads the first and final site of
the named MyoSim actuator route from the compiled `NHMYO1` source payload and
uses those exact Core bodies as the two skinning parents. Partitioned source
muscles such as gluteus and adductor magnus require all named MyoSim routes to
agree on the endpoint pair. Only the bilateral calcaneal tendon is an
explicitly labelled anatomical shared-tendon pair, because gastrocnemius and
soleus routes have distinct proximal links.

| View | PNG SHA-256 | Bone / muscle / tendon pixels |
| --- | --- | ---: |
| Front | `c3f2eaeddefb8308cb300ecdbe71278982356a68a20b619e36036faf4ea9c153` | 132,739 / 1,586,546 / 0 |
| Oblique | `f050fcf464c93818273a1b2d72a0cb4c1259e4180e759461a7a326af53f617ad` | 118,157 / 1,315,689 / 0 |
| Side | `138b4422a645713ff4fa0b1b56537736fe6ddc3a2210fad2f4fad792fd65bc30` | 74,180 / 775,656 / 0 |
| Rear | `1582eb21926cdcaeeeac42ddd22e2166f093dca7bba1591fc57c74d187e60677` | 353,793 / 1,201,421 / 0 |

The close camera is deliberately torso-centred, so the Achilles surfaces are
occluded in those four views. The geometry-framed replacement above retains
nonzero tendon coverage from every direction; a separate lower-leg capture
retains high-detail tendon coverage, including 11,613 pixels in the rear view.
The complete older close-capture command, payload/map hashes, device, coverage,
and bounded 416-muscle incremental-activation coupling smoke are in the
[capture transcript](media/myosim-native-fullbody-muscle-surfaces-2048/default/capture.transcript.txt).

This validates source surface → named MyoSim endpoint bodies → Metal pose →
native renderer. It does not claim force transfer from surface triangles,
continuum muscle/tendon deformation, collision/contact, a controller, gait,
or medical registration. The active bounded force step is retained as a
coupling smoke only; unsupported free-body co-activation is not presentation
motion.

The active check now applies 5% activation/excitation to all 416 source
muscles for 1 ms after subtracting the force at the source-default
zero-activation state. That reports the activation-induced force rather than
mistaking passive pre-stress for a control signal. It produces a finite maximum
configuration difference of `0.00720064735327` and retains tendon coverage in
all four lower-leg cameras (383 / 1,288 / 1,497 / 11,635 pixels):

<p align="center">
  <img src="media/myosim-native-fullbody-muscle-surfaces-2048/lower-leg/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-rear.png" width="48%" alt="Rest posterior lower-leg muscle and Achilles surface binding" />
  <img src="media/myosim-native-fullbody-muscle-surfaces-2048/lower-leg-muscle-driven/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-muscle-driven-focus-body-136-rear.png" width="48%" alt="Incremental 416-muscle driven posterior lower-leg binding" />
</p>

This is a one-step incremental-activation coupling check, not stable standing
or motion: support/contact, a posture controller, and time-series stability
are still required before a rollout can be claimed.

## Reviewed native posterior-calf source-surface inspection — 2026-08-27

<p align="center">
  <img src="media/myosim-native-posterior-chain-2048/default/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-front.png" width="32%" alt="Right posterior-calf source surfaces, front" />
  <img src="media/myosim-native-posterior-chain-2048/default/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-oblique.png" width="32%" alt="Right posterior-calf source surfaces, oblique" />
  <img src="media/myosim-native-posterior-chain-2048/default/myosim-fullbody-articulated-bodyparts-bones-source-soft-tissues-focus-body-136-rear.png" width="32%" alt="Right posterior-calf source surfaces, rear" />
</p>

These are reviewed native 2048 × 2048 reference frames of the exact
BodyParts3D 4.0 right lateral/medial gastrocnemius, soleus, and calcaneal
tendon surfaces over the 184-mesh BodyParts3D skeleton. The posterior view
shows the gold tendon surface continuing from the calf surfaces to the
calcaneus; it no longer uses a straight route line as a visual stand-in. The
four source surfaces contain 10,348 vertices and 45,582 indices. Muscle is
red, tendon gold, and bone ivory solely as presentation labels.

The four surfaces use the same fitted BodyParts3D → MyoSim source-default
frame as the skeleton. The gastrocnemius meshes use a femur-to-calcaneus
per-vertex blend; soleus and the calcaneal tendon use tibia-to-calcaneus.
At rest, both body-frame evaluations reproduce the same source vertex; after a
posed skeleton update, the native renderer blends the two evaluations before
rendering the world-surface snapshot. This prevents the old one-rigid-parent
shear at a crossing structure's two ends. It does not prove a watertight
biological attachment, deformable tissue, force-path transfer, or a mechanical
tendon constraint.

The local Apple M4 native probe uses one fresh reference renderer per camera;
the occupied Mac mini was not touched. All four cameras report nonzero bone,
muscle-surface, and tendon-surface coverage. Exact payload, runtime, and image
hashes are in the [transcript](media/myosim-native-posterior-chain-2048/default/capture.transcript.txt).

| Camera | PNG SHA-256 | Bone / muscle / tendon pixels |
| --- | --- | ---: |
| Front | `b919bb34c872f92d1adc9bca89407c73497a7f5b3858bb25041cbd5d0b076e0f` | 343,221 / 42,225 / 1,656 |
| Oblique | `cc8a60d00eb1fcc0ae99412e78a4eb06287f5ea04bcaa1bf66d49c3e83c1df2c` | 379,512 / 64,441 / 5,467 |
| Side | `f06ab2d6e9ef05b39c0c8c7643093e502289681963c77c3fbc44da465caf0cd1` | 302,924 / 45,325 / 6,384 |
| Rear | `623745bf6c83d712fbe083d5fdc7b341fb791134f1101404ab873fee465382eb` | 247,853 / 121,509 / 22,255 |

This is the current focused source-anatomy presentation. It is not a full Human
beauty render, skin, organ/vessel/nerve view, articulated continuum
deformation, muscle-driven rollout, or medically validated attachment model.
The active 1 ms all-muscle free-body stress frame was also checked from all
four angles but is intentionally not presented: without contact or a posture
controller, unsupported skeletal parts separate. Those claims remain separate
from this source-default visual binding.

## Reviewed native 184-mesh full skeleton — 2026-08-27

<p align="center">
  <img src="media/myosim-native-full-skeleton-184-2048/default/myosim-fullbody-articulated-bodyparts-bones-front.png" width="32%" alt="Native full skeleton, front" />
  <img src="media/myosim-native-full-skeleton-184-2048/default/myosim-fullbody-articulated-bodyparts-bones-oblique.png" width="32%" alt="Native full skeleton, oblique" />
  <img src="media/myosim-native-full-skeleton-184-2048/default/myosim-fullbody-articulated-bodyparts-bones-rear.png" width="32%" alt="Native full skeleton, rear" />
</p>

Core `1e247dd` rendered four inspected 2048 × 2048 reference frames on the
local Apple M4 from a `NHBONES1` package with 184 exact BodyParts3D 4.0 source
meshes (250,721 vertices, 1,370,928 indices). The package binds 17 conservative
fit landmarks plus 9 major extensions, 8 cranial/mandibular bones, 24 ribs, 10
mid-foot tarsals, atlas, axis, both triquetra, 52 wrists/hands/digits, 38
feet/toes, and 22 axial vertebrae to 86 named MyoSim parents in the active
157-body pose. `FJ1282`, the retired
"skull" selection, was an ocular component from a broad `part_of` listing; it
was removed and replaced by explicitly named cranial and mandibular source
bones before this capture.

The visual probe creates one native reference renderer per fixed camera, so a
2048 px camera cannot reuse an earlier camera's in-flight workspace. The
default view deliberately hides route lines. All source mesh instances are
attached to their current Core articulated parent pose; where the source model
has only one torso or toes body, ribs or foot meshes share that real parent
instead of receiving fabricated joint mechanics.

| View | Default-pose SHA-256 | Bone pixels | 1 ms complete-muscle SHA-256 | Bone pixels |
| --- | --- | ---: | --- | ---: |
| Front | `133a26781ad42893f69849073900a2827c30e85e7e49a06de55c917e5014d1d6` | 85,215 | `ddbd1d97118ffe6001be63f61721fe5c89d8378d78560c9712edb78977865095` | 85,171 |
| Oblique | `db8e1aa49a897f66882937be27456c6c0bef9c60ae5e06a7614a3033c6b37c8c` | 76,658 | `19c07753e142ef6064cf904eb0d50949c665688e79f5d627e95a11afa87db575` | 77,029 |
| Side | `44630a447c9865bef5cd22eb80113dd376a51419737e983b03af1b1b11de40f4` | 53,566 | `6ba8e9f0b87de0a64b886bfdfd6de5ca86a1ce1bab5e1c66fe6369c2249bc152` | 53,742 |
| Rear | `6d39289f874ae87bb13f09751aa2c764943ad5401334fe0b3aa165f537627b17` | 88,448 | `efea614d06ffbe41e02ec7d288fac3c34eb50ff5fc728c0fbb17341b21e0a912` | 88,256 |

The paired muscle-driven frames use the complete 416 source muscle-tendon
force set, 90 applied wraps, and one 1 ms CPU-FP64 free-body sensitivity step
before Metal poses the final 184 visual meshes. Exact commands, device,
payload/registration hashes, and every image hash are in the
[default-pose transcript](media/myosim-native-full-skeleton-184-2048/default/capture.transcript.txt)
and [muscle-driven transcript](media/myosim-native-full-skeleton-184-2048/muscle-driven/capture.transcript.txt).
The Mac mini was running an unrelated BirdFlow workload, so this is a bounded
local Apple M4 fallback; it is not an M4 Pro qualification.

This validates `full source visual skeleton → active MyoSim rigid parent →
Metal pose → native renderer`, and separately a bounded `416 source muscles →
one free-body step → Metal pose → the same skeleton` chain. It does not
validate skinning, tendons as surface geometry, organs, vessels, nerves,
collision/contact, a controller, a replay, gait, deformable tissue, or medical
registration.

## Retired full-body source validation snapshot — 2026-08-27

These images are retained source-provenance artifacts from the pinned MyoHub
`myo_sim` `33c89c2bde282553dde3f526768eb3bdcfaa7649` source. They are 640 ×
480 default-pose renders of the composed `myofullbody` model, and are retired
from presentation because their framing and generic source geometry do not
make a credible anatomical tendon view. The source is Apache-2.0; its
attribution and exact pin are in [third-party notices](../THIRD_PARTY_NOTICES.md).

| View | SHA-256 | Inspection result |
| --- | --- | --- |
| Anterior | `136d27ebbbae009997abfdd761bbb0ae2375a3f76210f0b5d8e2ff4509a09d39` | head, bilateral shoulders/arms/hands, thorax, pelvis, legs, and feet visible |
| Lateral | `a5b3a27de38d143384d623d082643a32c43ab31d37fc471b79a1229d1cb0a52a` | continuous skull–spine–pelvis–leg posture with intact foot profile |
| Posterior | `59b98823de79643b1b7f2688d2f261bb74904fbed8a2cfd7bacef3d7f16bd97a` | posterior spine, scapular region, gluteal/hamstring chains, calves, and feet visible |

The images matter because the active mechanics source now covers the whole
body rather than treating lower-body, upper-body, and neck imports as separate
animated shells. They are visual evidence of the source model only. They do
not show native Core rendering, mesh registration, skinning, contact,
locomotion, deformable anatomy, or biological validation.

## Retired native articulated route snapshot — 2026-08-27

Core `79cc34a` captured these 640 × 640 default-pose views on an Apple M4 using
`numi human myosim-native-visuals`. The command directly reads `NHRIGID2` and
`NHMYO1`, runs the Metal articulated operator, then renders the published pose
through Core's native visual renderer. The pale shapes are intentionally simple
inertial-body proxies; red geometry is 1,815 source attachment sites plus 1,432
straight route-centreline segments. It remains a historical coverage artifact,
not a tendon rendering or a path-to-bone attachment assessment.

| View | SHA-256 | Rendered coverage | Inspection result |
| --- | --- | --- | --- |
| Front | `b11acf05f0f6a46d1fabd5474d4db7266d431d6f8060e30ed7a74c939f6eba47` | `17,415` body / `641` site / `1,147` route pixels | bilateral torso, pelvis, legs, feet, and routed lower-body chains visible |
| Side | `ce2b2fc5d86437d0646bc6de0644cb0a9597b7b9faaed5544d3add0c3551d484` | `9,844` body / `521` site / `850` route pixels | continuous profile with shoulder, torso, pelvis, lower-leg, and foot route evidence |
| Rear | `c5067eccb567b71319f7ee49084005f114af71476d478359ccfcaf970380c91b` | `16,873` body / `1,070` site / `2,200` route pixels | posterior trunk, pelvic, calf, and bilateral route coverage visible |

This is native pose-bound visual evidence, not a human anatomy beauty render.
It proves neither BodyParts3D registration nor skin/organ deformation, live
device-buffer presentation, contact, motion, muscle-force feedback in a
rollout, or clinical validation. The tracked visual-pack manifest records the
scene provenance alongside the three frames. Its pack and manifest SHA-256
values are `633ddb213167c1cc47b733ae80d8f25a7af36d86bf830fbf67a625f16e2a8b59`
and `8d21b2f3a265285655dde72f3611891c69a187248627419dc1be2788b101734f`.

## Retired native BodyParts3D 27-major-bone route-overlay snapshot — 2026-08-27

Core `2aab522f92f44644c35bbde1a8ea3fd85356b027` captured this exact
`NHBONES1` package on the Apple M4 Pro on `macmini`. The native C++ program
read the compiled `NHRIGID2`/`NHMYO1` payloads plus 27 source-derived
BodyParts3D major-bone meshes (56,995 vertices; 322,074 indices), dispatched
the Metal articulated operator, and bound each mesh to its named Core
inertial-body pose. It also rendered all 1,815 compiled muscle sites and
1,432 straight route-centreline segments in a now-retired diagnostic overlay.

The offline rest-frame fit is deliberately unchanged: its original 18
unambiguous segment anchors enumerate 24 proper signed axis maps and select
the identity axis map with positive scale `1.007736155369` after mm→m
conversion. Its equal-weight mesh-vertex-centroid to source-inertial-COM score
is `0.059372888 m` RMS (`0.123618266 m` maximum). The nine additions are
bilateral hip bones, fibulae, tali, patellae, and sternum body. They inherit
that fitted common frame instead of re-fitting to less meaningful centroids;
the fibulae attach visually to their ipsilateral tibial link because MyoSim has
no separate fibular segment. A mesh centroid and an inertial COM are not
homologous anatomical landmarks, so these remain common-frame plausibility
diagnostics—not surface-registration accuracy or a medical registration claim.

| View | PNG SHA-256 | Bone / site / route pixels | Inspection result |
| --- | --- | --- | --- |
| Front | `a611906f2f92ba02cc271e53cd095d14535d340a5d6a17bc4c942aed418dbc66` | `4,529 / 1,035 / 3,195` | bilateral hip bones, patellae, fibulae, and both leg/foot chains are visible with the complete path overlay |
| Oblique | `9eec3e986be00463e6f6482a4faa8427dba8952c01155606e9a34694d23ac04d` | `3,654 / 936 / 2,911` | sternum, shoulder/scapular depth, pelvis, and distal-leg additions are visible without a mirrored frame |
| Side | `4cd9519bcb1520cd368b50ebedf4540dcd19cd8cff5b645835d52fd78508139b` | `2,017 / 714 / 1,855` | sagittal skull–sternum–pelvis–patella–leg–foot sequence is continuous |
| Rear | `d8f1ae245b74b1f119289e1b3aace0eba5bc6645310473e5288f4ee929818eaf` | `4,797 / 1,003 / 3,558` | posterior bilateral limbs, fibulae, and sacral/pelvic connection are visible |

The native [transcript](media/myosim-native-bodyparts-major-bones-27/native-articulated-major-bones-27.transcript.txt),
visual-pack manifest, and `.mrvpack` accompany the four frames. This validates
the historical `Metal pose → articulated BodyParts3D bone instance → native
renderer` chain at the source default pose. The red overlay does **not**
validate tendon attachment and is not current showcase material.

## Retired bounded muscle-driven 27-bone route-overlay snapshot — 2026-08-27

Core `2aab522f92f44644c35bbde1a8ea3fd85356b027` captured these 640 × 640
frames on the Apple M4 Pro on `macmini`. The new native visual mode reads the
same verified `NHRIGID2`, `NHMYO1`, and `NHBONES1` inputs as the default-pose
binding, reconstructs all 416 source MuJoCo muscle definitions, projects their
source-default `0.5` excitation / `0.5` activation forces in Core FP64, and
advances one free-body step. Metal then computes the final 157-body pose; Core
bound the 27 BodyParts3D bone instances and complete site/route overlay to
that pose for rendering. There is no Python process in this capture, but its
straight-line overlay is retired for the same reason as the default-pose view.

The selected 1 ms step is deliberately a bounded visual sensitivity probe. It
applies all 90 source-default wraps and changes the active state relative to
the identically integrated passive state by maximum velocity
`71.4839058782` and configuration `0.0714839058782`. This large
co-activation response is exactly why the capture is **not** called a posture,
trajectory, gait, or physiological prediction: there is no controller,
contact, repeated integration, or stability qualification.

| View | PNG SHA-256 | Bone / site / route pixels | Inspection result |
| --- | --- | --- | --- |
| Front | `1c4f91fa09c4b1bba7482040a00e8f7d2a03d86941c079fde95222c454449c7a` | `4,525 / 1,044 / 3,237` | bilateral hip bones, patellae, fibulae, and limb chains remain coherent with the complete path overlay |
| Oblique | `359d06e9fc1f6107f0335e91e2d1c9360f42dc710b7813ec2cc77304ad29a405` | `3,697 / 931 / 2,943` | sternum, shoulder girdles, pelvic depth, and distal-leg additions remain visible after the force-driven state change |
| Side | `1d26bc3ef918e74c865a71b256d3581ae6565427ec28723ba32eba5211efd2eb` | `2,047 / 723 / 1,873` | sagittal skull–sternum–pelvis–patella–leg–foot sequence remains continuous |
| Rear | `d27152ce837f7eaec25cfd34f4422b6531d19fb2abea9423b16e9d1fe04c3c15` | `4,802 / 1,007 / 3,542` | posterior bilateral limbs, fibulae, and sacral/pelvic connection remain present |

The native [transcript](media/myosim-native-muscle-driven-major-bones-27/native-articulated-muscle-driven-major-bones-27.transcript.txt),
visual-pack manifest, and `.mrvpack` accompany the four frames. This closes the
bounded `complete 416-muscle force → articulated state step → Metal pose →
BodyParts3D bone renderer` evidence chain. It is not valid tendon-attachment
imagery and does not make the provisional bone transforms colliders; prove
deformable muscle bellies, skinning, organs, vessels, nerves, contact, a
sustained muscle-force rollout, or clinical anatomy separately.

## Native mechanics progress

```text
MyoSim source composition (offline)
              |
              v
 NHRIGID2 articulated tree + NHMYO1 muscle-route payloads
              |
              v
 C++ Core: kinematics -> spatial tendon routes -> muscle force -> J^T scatter
              |
              v
             forward dynamics (FP64 reference)
              |
              +--> Metal: poses + analytic point Jacobians -> 416 routes + static force
```

The native probe at Core `86790f3` passed with:

| Native property | Measured result |
| --- | --- |
| Source bodies / Core bodies | 103 / 157 (54 exact zero-inertia serial transform carriers) |
| Configuration / velocity dimensions | 129 / 128 |
| Active muscle-tendon elements | 416 |
| Route sites / materialized wrap geometries | 1,815 / 143 |
| Default-pose position / orientation error | `6.27e-08 m` / `9.88e-08 rad` |
| Source-oracle muscle length / force error | `2.56e-08 m` / `4.89e-04 N` |
| Inverse/forward dynamics round-trip error | `4.92e-13` |
| 416-muscle 1 µs state-coupling delta | `7.15e-02` velocity / `7.15e-08` configuration |

The last row is an unconstrained FP64 sensitivity comparison against the same
passive state, not a standing, walking, contact, or physiological-stability
result.

Run that exact native path without a Python process:

```sh
numi human myosim-native-probe Build/myosim-fullbody
```

## Apple-GPU full-body mechanics progress

The same fixed source pose is checked through the native Metal
kinematics/Jacobian plus MyoSim route-force route:

```sh
numi human myosim-native-probe Build/myosim-fullbody --metal
```

On the local Apple M4, this dispatched one 157-body / 128-DoF Human and
compared all body poses plus one nonzero point query per body against Core.

| GPU parity property | Measured maximum error |
| --- | --- |
| Body position | `6.3206736356e-07 m` |
| Body orientation component | `1.42935285885e-07` |
| Point position | `6.54161804947e-07 m` |
| Analytic point Jacobian | `7.34255547086e-07` |
| Spatial-muscle path length | `7.45058059692e-07 m` |
| Static actuator force | `2.62451171875e-03 N` |
| Applied spatial wraps | `90 / 90` |

This is actual Apple-GPU articulated execution, not a compile-only claim. The
kinematics-only route admits up to 192 bodies and 160 DoF because it does not
reserve the dense mass-factor scratch space. The 128-DoF dense mass solve and
forward-dynamics stage remain the CPU FP64 reference owner after the
device-side MyoSim `J^T` projection; no contact or locomotion is claimed here.

## Remaining visual/mechanical steps

1. Resolve the remaining C1/C2 and triquetrum source-geometry gaps, then review
   their parent-body choice before adding them; matching anatomical labels alone
   remain insufficient evidence.
2. Replace the current four-bone visual shell with calibrated deformable skin
   mechanics only after acquiring a cited material law and validation data; do
   not call the visual blend a physical shell.
3. Resolve the Mortensen spine-to-`cervical_spine` rest registration and make
   an explicit MyoSim neck/head replacement decision before applying its 72
   cervical/hyoid muscle forces.
4. Replace the one-step sensitivity capture with a deterministic native
   free-body replay: persist `q`/`v` and the 416 activation states, advance
   source-defined activation dynamics from an explicit recorded control stream,
   emit a bounded frame sequence, and compare it with the matched passive
   replay. Until a controller and contact are validated, call that evidence a
   free-body response—not a posture or gait.
5. Move the dense forward-dynamics update to a measured device-resident path,
   preserving CPU-vs-Metal replay parity before promoting the native capture
   to a live presentation sidecar.
6. Add registered anatomical colliders and calibrated contact before any
   standing or walking qualification.

This ordering keeps the Human more realistic by retaining source mechanics and
by refusing to turn a visually plausible mesh into an uncalibrated physical
body.
