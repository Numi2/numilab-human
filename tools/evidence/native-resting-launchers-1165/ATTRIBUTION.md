# Source attribution and scope

This review uses the completed 1159 scene, not a measured individual. The run's exact invocation, asset hashes, final NHA, anatomy receipt/manifest, respiration configuration, composition report, and accepted MRVPacks are bound in review.json. The source receipt records clinical_validation=false and an integrated candidate status; this successful 20-second native preflight does not establish the final 310-second study or clinical validity.

## Required upstream credits

- BodyParts3D: BodyParts3D, © The Database Center for Life Science licensed under CC Attribution 4.0 International. Direct database license/download: https://dbarchive.biosciencedbc.jp/en/bodyparts3d/lic.html and https://dbarchive.biosciencedbc.jp/en/bodyparts3d/download.html.
- Z-Anatomy: Z-Anatomy - The libre 3D atlas of anatomy - CC-BY-SA 4.0. Pinned upstream revision: 9a2ef22cc8443d14f9aa18ba91246416b1c6ab7a; archive SHA-256 e029688545627bd0214b269e1063143abb580aad72b2c2445d6d8a9a0d9da736; license https://github.com/Z-Anatomy/Models-of-human-anatomy/blob/9a2ef22cc8443d14f9aa18ba91246416b1c6ab7a/License.txt. The included BodyParts3D source credit is separately BodyParts3D - The Database Center for Life Science - CC-BY-SA 2.1 Japan.
- MyoSim myofullbody: pinned commit 33c89c2bde282553dde3f526768eb3bdcfaa7649, Apache-2.0; repository https://github.com/MyoHub/myo_sim.

## Current scene anatomy

The exact NHA loaded by this run is the 1159 candidate (SHA-256 c10dce4609be99fdc569801c2705e23c46120b610c8aa86c631165d38baf4713). Its receipt and manifest preserve source lineage and describe eight bounded free-apex operations on inferred reference anatomy. Stable IDs 305-309 are source-derived lung-lobe envelopes; 310 is a derived external lung-union/pleura proxy; 311 is source-derived diaphragm geometry. The 310 proxy is not parietal pleura, an interlobar lining, fluid, or pleural mechanics. The 1159 composition report SHA-256 is ce5948d2dfe6388fc64f656fa34abfd7f409e471ee272a4a8231eba38281e108.

This scene combines registered source anatomy and explicitly inferred corrections; it is not a patient-specific or clinically validated model. The 20-second review demonstrates only this completed native preflight, the captured recording, and associated telemetry.
