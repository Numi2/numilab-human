# Registered 1170 recording review

This is a read-only post-run review tool for the registered 1170 control and treatment arms. Run it only after an arm has a successful sealed science v2 receipt and the execution script's retention marker. It validates the registered chain with the pinned 1171 helper, requires the closed output files to carry macOS user-immutable flags, and confirms the movie and surface-audit CSV hashes against the exit receipt before reading video.

## Commands

Receipt and closure validation without hard-linking or decoding:

    /Users/n/numi-human-prep-venv-20261005/bin/python3.13 /Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170-review/review_registered_arm_1174.py --arm control --prepare-only

Full review for each closed arm:

    /Users/n/numi-human-prep-venv-20261005/bin/python3.13 /Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170-review/review_registered_arm_1174.py --arm control

    /Users/n/numi-human-prep-venv-20261005/bin/python3.13 /Users/n/numi-human-retained-delivery-20261009/native-integrated-resting-study-1170-review/review_registered_arm_1174.py --arm treatment

Outputs go to native-integrated-resting-study-1170-review/control and /treatment. Existing output directories are never overwritten. Movie and CSV are hard-linked from the receipt-bound scene; the registered source files are not modified. AVFoundation must finish the full reader pass, report frame count equal to surface-audit data rows, and pass the existing inspector's strictly increasing sorted-presentation-PTS and simulated-time checks. The inspector extracts initial, middle, final, and seven tour-layer frames (skin, muscles, skeleton, organs, lungs, heart, vessels) at the invocation's recorded tour period.

The inspector sorts presentation timestamps before checking them. The report explicitly makes no claim about the original compressed-sample decode order. It also does not infer geometric, physiological, or endurance qualification from a successful movie decode.

## Source pins and verification

- P18-bound 1171 receipt adapter: registered_receipt_adapter_1171.py, SHA-256 ba0b88416dbf9cab892f9daa6f954e931003f9db5075db6b758c209370efc1e8.
- Exact published 1096 V8 D-map reader: SHA-256 9a55338d874ecc4695e1545829c21266b9b5734d8ca8d64acab817906091c7bd.
- Existing AVFoundation reader: /Users/n/numi-human-resting-final-source-028/matter/tools/inspect_resting_movie.swift, SHA-256 2d6704dd5f06bffd0b8aa0072171af805fdb3dd00dcc11c02e0b692513b21649.
- Review orchestration tests: test_review_registered_arm_1174.py (3 tests); 1171 registered receipt and map suite: 11 tests. No study movie has been decoded by this preparation.
