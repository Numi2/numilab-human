# Native left deep masseter coverage

The integrated resting scene gains the missing left deep masseter as passive anatomical row162, sourced from Z-Anatomy object Deep part of masseter.l. Its historical offline trial was labelled161; the new manifest explicitly maps that trial to canonical162 (FMA49005, left deep masseter). No previously admitted identity or geometry is changed. Right deep masseter remains a separate, incomplete structure.

The existing head registration and exact surface constructor reconcile the source with temporal bone, mandible and superficial masseter. Three cuts remove 1.389% of registered source volume in total. The 10 micrometre boundary spacing is an inferred precision reserve, not a measured anatomical gap. Triangle ancestry distinguishes retained source faces from generated cut boundaries. The unchanged source attachment-marker objects are proxies, not certified attachment footprints.

The added surface uses the existing head core23 transform. It does not add jaw actuation, MyoSim routes, physical mass or forces. All154 prior tissue rows remain exact.

A fresh16-second native run checks the composed scene. Across eight actual captures the added surface is closed, positively oriented and self-clear, with zero exact crossings against all864 other non-bed surfaces. All864 prior surfaces, all157 body poses and the1000-by60 coupled physiological trace match the parent sartorius scene exactly. The continuous invalid-triangle and bed-crossing audit passes; the movie has252 image frames and5 timing markers.

## Reproduce on the Mac mini

    python3 tools/evidence/native-anatomy-left-deep-masseter-1331/reproduce.py --prepare-only
    python3 tools/evidence/native-anatomy-left-deep-masseter-1331/reproduce.py

Requires pinned retained source assets and existing native runtime. Source rights and attribution are retained. This is reference anatomy assembled from registered sources, not a measured individual. No complete jaw function, certified entheses, whole-body anatomy completion or final five-minute qualification is claimed. The parent is native-anatomy-sartorius-precision-1330. Performance work remains deferred.
