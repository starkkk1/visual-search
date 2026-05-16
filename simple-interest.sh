#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 PRINCIPAL RATE TIME

Calculate simple interest.

  PRINCIPAL  Principal amount (e.g. 1000)
  RATE       Annual interest rate in percent (e.g. 5)
  TIME       Time in years (e.g. 2)

Example: $0 1000 5 2
EOF
  exit 1
}

if [ "$#" -ne 3 ]; then
  usage
fi

principal=$1
rate=$2
time=$3

interest=$(awk -v p="$principal" -v r="$rate" -v t="$time" 'BEGIN{printf "%.2f", p*(r/100)*t}')
total=$(awk -v p="$principal" -v i="$interest" 'BEGIN{printf "%.2f", p + i}')

printf "Principal: %s\nRate: %s%%\nTime: %s\nInterest: %s\nTotal: %s\n" "$principal" "$rate" "$time" "$interest" "$total"
