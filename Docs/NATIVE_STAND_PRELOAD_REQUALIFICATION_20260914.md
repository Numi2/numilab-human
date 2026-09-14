# Native Human static-preload requalification

The isolated Mac mini native branch `human-completion-static-preload` was rebuilt at `663c82a` after correcting a contradicted 64-step receipt. The run used the canonical one-adult payload set recorded by the native receipt:

- rigid core SHA-256 `6328f7e84663c611c5498624d1386b00b2d5b0e162c4cc2967c7b1dc49ab0c44`
- 416-route muscle SHA-256 `9a988f19a6fd8e533cd0f2bf3192cb8535fb008ccd394ffbf1a4432d3db76a05`
- support SHA-256 `4d54f8155cd83baaee7af536099824ac0da61e5d5e77544b42c6e5ce1b48c907`
- equality SHA-256 `b97f755c769d0af16e02ab5deb9d85bd0cc921649197f71d308e98130ac69b6a`
- NHTENDON3 SHA-256 `a594194f510eb4aa990a8767f868f999a10b4fedb745c8665368a231ed39b555`

The clean build passed. At the canonical 12.5 µs clock, the one-step persistent release measured `3.00897479057 m/s²` peak acceleration and `7.83274299465e-5 m/s` maximum velocity delta. A deterministic 64-step (`0.8 ms`) repeat measured `7.71253347397 m/s²`, `3.95996576117e-5` normal impulse, `0.00596422795206 m/s` maximum final velocity delta, and `3.80200799555e-6 m` maximum configuration delta.

The earlier `5.75767946243 m/s²` 64-step value remains preserved in the original native receipt. It is contradicted by the clean rebuild and repeat under the same code commit and payload hashes; the corrected immutable native record is `qualification-rerun-20260914.json` on the published native branch.

This closes evidence correction and deterministic bounded reaction handoff. It does not close the static `balanced=false` internal residual (`0.86077456182` RMS), 10–60 second assistance-free standing, perturbation recovery, walking, subject calibration, deformable materials, or blood-to-tissue mechanics.
