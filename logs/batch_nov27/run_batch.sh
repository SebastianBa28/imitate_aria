#!/bin/bash
# post-process every raw_data/*.vrs into a fresh datasets/imitate-aria, keep stats + npz + seg.jpg, drop videos
cd "$(dirname "$0")/../.."
L=logs/batch_nov27
: > $L/stats.jsonl; : > $L/failed.txt
for v in $(ls raw_data/*.vrs | sort -V); do
  s=$(basename "$v" .vrs)
  extra=""
  case "$s" in rubiks_nov27_3) extra="--start 24";; esac   # objects out of view at the auto start
  echo "=== $s $(date +%T)"
  if uv run python packages/data/post_process.py --vrs "$v" --append-to --keep-output $extra > $L/$s.log 2>&1; then
    uv run python $L/health.py output/$s/${s}_aina.npz >> $L/stats.jsonl
    rm -f output/$s/*.mp4
  else
    echo "$s" >> $L/failed.txt; echo "  FAILED (see $L/$s.log)"; tail -3 $L/$s.log
  fi
done
echo "=== done $(date +%T)"
