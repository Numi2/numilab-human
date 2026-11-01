# Actual 008 evidence preservation note

The original run manifest and execution record are unchanged. The failed run is retained at `actual-008-refined-source-quadrature/`; its execution exit code is 6. The run log records rejection before solve: the inherited 1046c carrier witness used a lowercase role where the reader requires canonical `Skin`/`Fat` casing.

The run manifest refers to `receipt-builder-013/source-control-manifest-013.json` with SHA-256 `8694fa30cd1b81a9120369018e8ad7612b59fe17717b695cef13cebe46078b2d` and 5,894 bytes. That path now contains the later OUTPUT_DIR-filtered adapter pin and has been superseded. The exact earlier control-manifest bytes are preserved at `../receipt-builder-013/historical-013-before-output-dir.json`, SHA-256 `8694fa30cd1b81a9120369018e8ad7612b59fe17717b695cef13cebe46078b2d`, also 5,894 bytes.

The current control manifest remains at `../receipt-builder-013/source-control-manifest-013.json`, SHA-256 `28eb44f0a52066c2fd8aa129cf312326b1172bf15e39887921fd5703257c6079`. No run manifest or execution record was rewritten, and the 008 attempt did not produce a mass-mapping pass.
