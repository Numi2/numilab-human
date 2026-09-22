#!/bin/sh
# Three short native runs, one immutable binary and source package. No replay.
set -eu
[ "$#" = 3 ] || { printf '%s\n' 'usage: reproduce.sh <native-build> <source-input-directory> <new-output-directory>' >&2; exit 2; }
build=$1
inputs=$2
output=$3
mkdir "$output"
probe=$build/bin/metalrobo_numilab_human_myosim_visual_probe
export NUMI_HUMAN_EXECUTION_STAGES=1
hash_inputs() {
    shasum -a 256 "$probe" "$build/lib/libmetalrobo.dylib" "$build/shaders/MetalRobo.metallib" \
        "$inputs/myosim-fullbody-core-reference.nhrigid" \
        "$inputs/myosim-fullbody-muscle-reference.nhmyo" \
        "$inputs/numi-human-tendon-attachments.nhtendon" \
        "$inputs/myosim-fullbody-support-contact.nhcnt" \
        "$inputs/myosim-fullbody-joint-equalities.nheq"
}
hash_inputs > "$output/inputs-before.sha256"
run_prefix() {
    name=$1
    shift
    mkdir "$output/$name"
    set +e
    /usr/bin/time -l "$probe" \
        "$inputs/myosim-fullbody-core-reference.nhrigid" \
        "$inputs/myosim-fullbody-muscle-reference.nhmyo" "$output/$name/frames" \
        --tendon-payload "$inputs/numi-human-tendon-attachments.nhtendon" \
        --support-contact-payload "$inputs/myosim-fullbody-support-contact.nhcnt" \
        --joint-equality-payload "$inputs/myosim-fullbody-joint-equalities.nheq" \
        --support-stance-dof 2 0.02 \
        --support-stance-dof 108 0.1 --support-stance-dof 109 0.1 \
        --support-stance-dof 110 0.1 --support-stance-dof 122 0.1 \
        --support-stance-dof 123 0.1 --support-stance-dof 124 0.1 \
        --support-stance-contact 2 --support-stance-contact 3 \
        --support-stance-contact 4 --support-stance-contact 5 \
        --support-stance-contact 6 --support-stance-contact 7 \
        --muscle-step-seconds 0.001 --muscle-step-count 64 \
        --persistent-metal-stand --persistent-source-passive-joint-tissue \
        --stand-contact-iterations 64 --mechanics-only "$@" \
        > "$output/$name/stdout.txt" 2> "$output/$name/stderr.txt"
    code=$?
    set -e
    printf '%s\n' "$code" > "$output/$name/exit-code.txt"
    printf '%s exit=%s\n' "$name" "$code"
    [ "$code" = 0 ] || exit "$code"
}
run_prefix feedback-energy --stand-muscle-path-feedback 10 1 --stand-endpoint-energy
run_prefix feedback-plain --stand-muscle-path-feedback 10 1
run_prefix feedback-disabled-energy --stand-endpoint-energy
hash_inputs > "$output/inputs-after.sha256"
cmp "$output/inputs-before.sha256" "$output/inputs-after.sha256"
