#!/bin/bash
set -euo pipefail
run_root=/Users/n/human-standing-20260922
input=$run_root/current-toe-enthesis-input-20261002
output=$run_root/current-toe-enthesis-feedback-512ms-20261002
runtime=/Users/n/numi-human-standing-20260922
build=/Users/n/numi-human-standing-build-20260922
probe=$build/bin/metalrobo_numilab_human_myosim_visual_probe
mkdir -p "$output/mechanics"
export NUMI_HUMAN_EXECUTION_STAGES=1
export NUMI_LAB_ROOT=$runtime
export NUMI_BUILD_DIR=$build
{
  printf 'host='; hostname
  printf 'cpu='; sysctl -n machdep.cpu.brand_string
  printf 'utc='; date -u '+%Y-%m-%dT%H:%M:%SZ'
  cat "$run_root/provenance.txt"
} > "$output/execution-context.txt"
shasum -a 256 "$probe" "$build/shaders/MetalRobo.metallib" \
  "$input/myosim-fullbody-core-reference.nhrigid" \
  "$input/myosim-fullbody-muscle-reference.nhmyo" \
  "$input/myosim-fullbody-joint-equalities.nheq" \
  "$input/myosim-fullbody-support-contact.nhcnt" \
  "$input/numi-human-tendon-attachments.nhtendon" \
  > "$output/input-binary-sha256.txt"
if /usr/bin/time -l "$probe" \
  "$input/myosim-fullbody-core-reference.nhrigid" \
  "$input/myosim-fullbody-muscle-reference.nhmyo" "$output/mechanics" \
  --tendon-payload "$input/numi-human-tendon-attachments.nhtendon" \
  --support-contact-payload "$input/myosim-fullbody-support-contact.nhcnt" \
  --joint-equality-payload "$input/myosim-fullbody-joint-equalities.nheq" \
  --support-stance-dof 2 0.02 \
  --support-stance-dof 108 0.1 --support-stance-dof 109 0.1 \
  --support-stance-dof 110 0.1 --support-stance-dof 122 0.1 \
  --support-stance-dof 123 0.1 --support-stance-dof 124 0.1 \
  --support-stance-contact 2 --support-stance-contact 3 \
  --support-stance-contact 4 --support-stance-contact 5 \
  --support-stance-contact 6 --support-stance-contact 7 \
  --persistent-metal-stand --persistent-source-passive-joint-tissue \
  --stand-contact-iterations 64 --stand-muscle-path-feedback 10 1 \
  --muscle-step-count 512 --muscle-step-seconds 0.001 --dimension 512 \
  2> "$output/stderr.txt" | tee "$output/stdout.txt" | awk '
  /^human_standing_progress=/ {
    for (i=1; i<=NF; i++) { split($i, item, "="); value[item[1]]=item[2] }
    printf "%.3f / 0.512 simulated s | root speed %.3f mm/s | support %.1f N | contacts %d | assistance %s N\n", \
      value["simulated_seconds"], value["root_linear_speed_m_s"]*1000, \
      value["support_force_n"], value["contact_count"], value["root_assistance_force_n"]
    fflush()
  }'; then
  printf 'native_run=completed\noutput=%s\n' "$output"
else
  result=$?
  tail -20 "$output/stderr.txt" >&2 || true
  printf 'native_run=failed\nexit=%s\noutput=%s\n' "$result" "$output" >&2
  exit "$result"
fi
