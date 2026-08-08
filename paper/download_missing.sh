#!/usr/bin/env bash
# Download remaining missing items from missing_list.txt (sequential, resume-aware)
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"

log() { echo "$1"; }

fetch() {
  local out="$1" url="$2"
  if [ -s "$out" ]; then
    log "SKIP  $(basename "$out")  (exists)"
    return 0
  fi
  local args=(-sL --compressed --fail --retry 5 --retry-delay 4 --retry-connrefused --max-time 600 -A "$UA")
  if [ -s "$out.tmp" ]; then
    args+=(-C -)
  fi
  if curl "${args[@]}" -o "$out.tmp" "$url" 2>/dev/null; then
    if [ -s "$out.tmp" ]; then
      case "$out" in
        *.pdf)
          if head -c 5 "$out.tmp" | grep -q '%PDF'; then
            mv "$out.tmp" "$out"
            log "OK    $(basename "$out")  <-  $url"
          else
            rm -f "$out.tmp"
            log "FAIL  $(basename "$out")  (not a PDF / blocked)  <-  $url"
          fi
          return 0
          ;;
        *)
          mv "$out.tmp" "$out"
          log "OK    $(basename "$out")  <-  $url"
          return 0
          ;;
      esac
    fi
  fi
  log "FAIL  $(basename "$out")  <-  $url"
  return 1
}

echo "==== Missing items (sequential) ===="
while IFS='|' read -r out url; do
  [ -n "$out" ] || continue
  fetch "$out" "$url"
done < "$DIR/missing_list.txt"
echo "==== Done ===="
