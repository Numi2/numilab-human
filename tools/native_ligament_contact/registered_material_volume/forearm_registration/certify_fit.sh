set -eu
volume=$1
fit=$2
checker=$(awk '{print $2}' "$3")
"$PY" "$4" "$volume" "$fit" "$checker" "$5" >"$fit/interval.log" 2>&1 &
pid=$!
"$PY" "$6" "$volume" "$fit" "$7" >"$fit/walls.log" 2>&1
wait "$pid"
