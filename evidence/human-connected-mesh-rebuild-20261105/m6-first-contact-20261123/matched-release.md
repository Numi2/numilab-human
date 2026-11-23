# Matched physical starting pose on SSH Numi

The supported and full-volume scenes used different bilateral shoulder elevations. Both used the same articulated and muscle sources. Setting MyoSim `shoulder_elv_r` and `shoulder_elv_l` to the full-volume reference of 0.10 rad changed only the initial condition. The native solver remained free to move all four limbs. The existing 0.18-rad supported start placed some finger bones 56.17 mm from their full-volume positions after pelvis alignment.

The preparation tool copies the existing joint overrides from a pinned full-volume launch. Its check requires all 157 native bodies, including their orientations, plus both physical material audits and the mass mapping. When a replay includes only its skin-bound bodies, a paired source capture must match the initial pack hash, runtime, pose overrides and every available native pose. A different initial pack failed this check. With matched shoulders, one rigid transform put every body within 0.0000671 mm and 1.95e−9 radians of the full-volume release. Its principal translation was 0.00566 mm vertically. Keeping this transform external preserves the original material coordinates, cell volumes and source mass mapping.

## Native result

The supported scene with the registered finger insertions accepted 8,000 free steps of 2 ms. The final-second mean bed reaction was 952.854 N against 952.864 N of weight. Maximum center-of-mass speed in the same interval was 0.0718 mm/s. The trace recorded three breaths, 19 filling–ejection cycles, a 528.82 mL tidal volume, 69.29 mL stroke volume and pleural pressure from −770.01 to −470.34 Pa. Root assistance was disabled.

| Metric | 7.998 s | 16.000 s |
| --- | ---: | ---: |
| Nonocular skin-to-tissue triangle pairs | 11,316 | 11,601 |
| Skin self-intersection pairs | 37 | 40 |
| Registered terminal sites outside skin | 4 | 4 |

End-to-end time was 169.27 s for 16 simulated seconds, or 0.09452×. The instrumented warm interval ran at 0.10183× and the complete body window ran at 0.09948×. The trial omitted the optional detailed bed quadrature diagnostic retained by the earlier 0.18-rad run. That observer difference and the changed pose preclude attributing the timing difference to either one alone.

## Material decision

The registered outer surface and the existing supported skin remain physically distinct. All four exposed index and little finger insertions were outside the supported skin at release, halfway through and at the end. An offline check carried each of the 311,020 material nodes with its recorded anatomical donor through those same native poses. At the late checkpoint that shortcut inverted 21,882 material tetrahedra. Even at release, re-emitting the nearly identical joint transforms separately at Float32 precision inverted two tetrahedra. Keeping one external release transform avoids introducing strain at initialization. The supported trajectory does not qualify a common moving material volume.

Keep this launch as a matched starting condition for coupled skin/fat development. The exact same detailed skin and fat must supply contact geometry, load transfer and the inspected surface through motion. A shared skeleton alone cannot supply that guarantee. Retain the original bed, the conservative fractional mass mapping and the current constitutive laws during that work.

The existing prep and run commands on SSH Numi operate on the installed `connected-digit-registration-064` directory. The release archive includes the launch commands, input and binary hashes, geometry witnesses, and three native replay frames. [Measured results](matched-release-results.json) · [Verified recovery](matched-release-recovery.json)
