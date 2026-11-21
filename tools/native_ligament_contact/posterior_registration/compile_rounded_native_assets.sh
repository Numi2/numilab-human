#!/bin/bash
set -euo pipefail
p=$1
prev=$2
base=$3
app=$4
py=$5
law="$app/matter/materials/passive_skeletal_muscle_unqualified.nmatter"
ref="$prev/native-release-motion/step-8.receipt.json"
"$py" prepare_native_contact.py "$p" "$prev" "$base" "$p/response" "$p/native-assets"
"$py" prepare_native_cells.py "$p" "$prev" "$p/response" "$law" "$p/native-assets"
"$py" prepare_contact_pool.py "$p" "$prev" "$p/response" "$p/native-assets" "$p/pool-assets" cortical_distance.dylib
"$py" enrich_cortical_modes.py "$p" "$p/response" "$p/pool-assets" "$ref" "$law" cortical_distance.dylib "$p/enriched"
"$py" prepare_native_contact.py "$p" "$prev" "$base" "$p/enriched" "$p/enriched-assets"
"$py" prepare_native_cells.py "$p" "$prev" "$p/enriched" "$law" "$p/enriched-assets"
"$py" prepare_contact_pool.py "$p" "$prev" "$p/enriched" "$p/enriched-assets" "$p/enriched-pool" cortical_distance.dylib
"$py" prepare_cortical_vertex_faces.py "$p" "$prev" "$p/enriched-pool" "$p/face-geometry"
"$py" factor_contact_columns.py "$p" "$p/response" "$p/face-geometry" "$ref" "$law" "$p/face-assets"
"$py" prepare_reverse_surface.py "$p" "$p/conforming-boundary" "$p/enriched-pool" "$p/closed-geometry"
"$py" use_registered_force_columns.py "$p/face-assets" "$p/closed-geometry" "$p/closed-qualified"
"$py" prepare_free_residual.py "$p" "$p/response" "$p/closed-qualified" "$ref" "$law" "$p/certified-assets"
"$py" prepare_pelvic_quotient.py "$p" "$p/response" "$p/certified-assets" "$ref" "$p/quotient-assets"
"$py" prepare_pelvic_reference_frame.py "$p" "$p/quotient-assets" "$ref" "$p/relative-assets"
"$py" prepare_finite_facets.py "$p" "$p/relative-assets" "$ref" "$p/facet-assets"
"$py" prepare_feature_facets.py "$p/facet-assets" "$p/native-features"
