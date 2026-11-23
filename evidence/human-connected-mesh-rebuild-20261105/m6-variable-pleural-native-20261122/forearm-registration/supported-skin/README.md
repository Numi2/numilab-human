# Supported arm skin through 16 seconds of native motion

The executable simulates the compact resting reference adult with the registered skin/fat bed sections, skeletal point Jacobians, local muscle mechanics, Brain control, variable pleural pressure and circulation. Its source correction redistributes selected trunk influence to existing ipsilateral arm donors. It preserves the registered neutral geometry, named skeletal and muscle attachments, and the source bindings at measured load-bearing skin points. Joints and contact vertices remain free to move. The dense skin/fat FEM uses a separate physical boundary.

| Native result | Control | First skin trial | Skin with bearing points preserved |
| --- | ---: | ---: | ---: |
| External skin/anatomy pairs at 16 s | 13,910 | 13,901 | 13,529 |
| Skin self-intersection pairs at 16 s | 40 | 47 | 37 |
| Skin/bone pairs at 16 s | 2,791 | 2,859 | 2,791 |

The first trial changed one bed section area by 4.0% and displaced a finger bone by 13.1 mm relative to the control. The selected revision retains the source bindings at the measured bearing surface and nearby triangles. All 8,000 physical steps completed. The accepted skeleton, respiration state and support impulse history matched the control byte for byte. Fifteen individual anatomy banks improved and none gained crossings. The largest reduction was 136 triangle pairs in muscle bank 111. At the separately captured 7.998-second pose, 13,310 external pairs and 47 skin self-intersections remained.

In the last simulated second the sampled bed reaction averaged 952.891 N against 952.864 N of body weight. Maximum center-of-mass speed was 0.050 mm/s. The 16-second trace contains three breaths and 19 complete filling/ejection cycles. The M6 simulation and viewer measured 0.1042× over the 15.936-second active window. The complete run including launch and terminal output took 166.57 seconds, or 0.0961×. The integrated owner reports presentation qualification pending.

The reported surface intersections and the degree of physical exposure are different measurements. On the final compact frame, approximately 24% of each brachioradialis and 42–45% of each extensor digitorum surface remained outside the skin. The selected registration is a development candidate. Anatomical clearance, dense equilibrium, sustained gas regulation and the long-duration qualification remain open.

The dense three-step reference identifies two failing subcutaneous fat cells with four corners on the bonded interface. At 10 mm radius the surrounding connected fat retained 99.33% and 99.88% of its installed volume. The two cells retained 35.00% and 225.96%, respectively. Local retriangulation on the installed mesh found 24 and one topology-valid alternatives. The best alternatives changed nodal force by up to 35.2% and 79.5% over the two measured configurations. Neither passes the existing force comparison. Keep the material equations and compare a reconstructed bonded patch by its force transfer and deformation before integrating it into the dense scene.

On the configured M6, run the verified development scene with the following command. It creates a fresh native evidence directory for each launch.

```sh
ssh numi '~/Work/run-numi-human-rest'
```

An independent launch through that command repeated all 8,000 accepted steps. The terminal geometry digest, body state, respiration state and four trace files matched exactly. Its cold wall time was 165.10 seconds (0.0969×).
