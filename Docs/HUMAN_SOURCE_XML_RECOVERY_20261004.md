# Pinned MyoSim XML source recovery — 2026-10-04

The cumulative Human source inventory now parses the pinned legacy elbow XML
that previously stopped at an invalid attribute boundary. The upstream archive
and checkout were left byte-identical.

The archived member is
`myo_sim/models/legacy/elbow/myoelbow_1dof6muscles_1dofSoftexo_sim2.xml` in
MyoSim revision `33c89c2bde282553dde3f526768eb3bdcfaa7649`. Its SHA-256 is
`9df2192a504b92b2785528cf786b049776bbbb91051805ea2013712ce8d90637`, bound by
archive SHA-256
`280d297aa496acccf3f1c5373a1304d23f9569362c2d6960910128bfba144975`. Two start
tags omit the required separator between attributes:
`name="exoUpperarm"type="box"` and `name="exoForearm"type="box"`.

Compiler `numilab-human.target-coverage.3` applies a repair only when both the
member path and raw member hash match those pins. It inserts one ASCII space at
raw byte offsets 3484 and 6612; the transformed member hash is
`81e676b1e17a994d58a96d330aa6e1650f9edda3b8bcfe1aad9f943a45384d00`. The XML
parser then retains 179 declarations. The current register records the raw and
transformed hashes, rule ID, offsets, and insertion count. The previous invalid
register and its 15 parsed declarations remain in cumulative history.

The new snapshot preserves all 28,415 leaves from the committed compiler-v2
manifest and adds 164 declarations from this source member. It contains 28,579
leaves and all 95 mandatory targets. Two source inventories remain unavailable:
the authenticated bimanual MoBL-ARMS release and the raw Z-Anatomy calf blend.
They keep source scope blocked. The snapshot records 28,484 nonmandatory source
leaves without target assignment; declarations do not establish anatomy
fidelity, physical ownership, mechanics, physiology, or qualification.

The machine-readable snapshot and checksum list are in
[the compiler-v3 coverage bundle](media/human-source-coverage-20261004-v3/receipt.json).
Focused validation passed 30 tests and 17 subtests across target coverage and
gap execution. The exact-source recovery test parses the archived member and
rejects any changed path or source hash.
