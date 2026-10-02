# MyoSim left knee range overlay — 2026-10-02

The left `knee_angle_translation2_l` equality law was the exact sign mirror of
the right law, but its authored slide range had been copied with the right-side
sign. At 0.9 rad of knee flexion the left law therefore produced about `-4.273
mm`, outside its old `[0, 6.792] mm` interval.

The pinned upstream archive remains unchanged. A hash-bound overlay changes
only the left joint's range to `[-6.792, -7.69254e-8] mm` (stored in metres as
`[-0.006792, -7.69254e-11]`). The left equality polynomial remains untouched:

```text
-7.69254e-11 - 0.00587971 x + 0.00125622 x^2
+ 2.61846e-06 x^3 - 6.24355e-07 x^4
```

Across the authored driver interval `[0, 2.0944] rad`, its derivative remains
negative, rising from `-0.00587971` to `-0.000606142021`. The polynomial is
therefore monotone decreasing; its endpoint values are `-7.69254e-11 m` and
`-0.006791998837245416 m`. Reflecting the right-side interval covers this full
source law while preserving bilateral range symmetry.

The overlay binds to upstream revision `33c89c2bde282553dde3f526768eb3bdcfaa7649`
and archive SHA-256
`280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975`. The
original extracted chain file SHA-256 is
`da7c4a13eec2abfedc683210972ea423d630461a56909bee2884e181e1bf1d97`; the
corrected file SHA-256 is
`885fdc6fa0e8a2d24d590c3987856f4a17256f1d44f277f46ecf9dbfedc02b0a`. The
tracked overlay manifest SHA-256 is
`05f6b7698e571c83a62bdbc7055ff24322800a36e9b53e455a18b451e7b9dad4`.

The pinned MuJoCo 3.12.0 model compiled the left range exactly as
`[-0.006792, -7.69254e-11]`; the right range remains
`[7.69254e-11, 0.006792]`. The equality coefficients remain exact sign mirrors.
After equality projection, the left slide is `-0.0042727016589 m` in the
bilateral knee-flexion pose and `-0.00576461634485 m` in deep crouch; both are
inside the corrected interval. Neutral also passes. In functional crouch the
left slide passes and the two existing knee rotation coordinates still fail
their separate source ranges.

A refreshed local native reference artifact at
`Build/myosim-fullbody-left-knee-overlay-20261002` binds the overlay in its
manifest and carries the corrected `NHLIM1` source-compliance range. The
2026-10-02 accepted-step visual stand invocation does not take an `NHLIM1`
input, so this source correction has not changed that standing trajectory. The
loaded-knee source-compliance compiler and strict schema now bind the refreshed
limit payload; its regression suite passes against the tracked binary fixture
at `Docs/media/myosim-left-knee-range-overlay-20261002/`. This verifies only
the source-program binding. No Matter accepted-state transaction has run with
the new bytes. The artifact's rigid,
muscle, support, equality, extensor-hood, and support-primitive payload hashes
match the parent artifact. The 416 cached muscle-fit records were accepted only
after their source archive, ordering, fit objective, input arguments, and
source oracles matched the pinned export. The refreshed reference manifest SHA-
256 is `6f77958df63a9785f00b3837197d521290b5cfcb0d6640692e9cf47aac7b7913`, and
the corrected `NHLIM1` payload SHA-256 is
`fb799ed55a61c0d08a84fcb5ddd3f278ed8561e881318ca76d61e9e51268a29c`. The full
source driver-domain audit now finds 37 range conflicts with no unverified
domains; the left translation conflict is gone. Other authored dependent
ranges still conflict with their source laws at a few full-domain extrema,
including translation1, rotation2/3, and beta coordinates. Their bilateral
source laws do not show the same copied-sign error, so their intervals remain
unchanged pending stronger mechanical evidence. The local verification receipt
is `Build/myosim-fullbody-left-knee-overlay-20261002/refresh-receipt.json`
(SHA-256 `45e6ccad6e6b6ba7aeaf33a124679be9a64a9b2ae504d12c7d0ebab88d284e5e`).

This closes the source interval inconsistency and verifies its compiled range
and projected source poses. It does not qualify loaded knee mechanics,
patellofemoral contact or pressure, clinical anatomy, or sustained standing.
The separate patellar anteriority audit still covers sampled static poses;
cartilage intersection and loaded contact remain open.
