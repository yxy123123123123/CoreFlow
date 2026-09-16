#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_WORK_ROOT="${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/coreflow_m3_k_scaling_mini_workspace}"
PYTHON="${PYTHON:-$SOURCE_WORK_ROOT/runtime_env/bin/python}"
CONFIG="$ROOT/config/m3_k_scaling_mini_protocol.json"
REPORT_ROOT="$WORK_ROOT/reports/m3_k_scaling_mini"
PHASE="${1:-}"

die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
trap 'die "Stopped at line ${BASH_LINENO[0]} (exit $?); partial output was retained and no automatic retry was attempted."' ERR

verify_package() {
  [[ -x "$PYTHON" ]] || die "Frozen runtime Python not found: $PYTHON"
  "$PYTHON" "$SCRIPT_DIR/package_checksums.py" --root "$ROOT"
}

require_file() {
  [[ -f "$1" ]] || die "$2"
}

preflight() {
  mkdir -p "$REPORT_ROOT"
  "$PYTHON" "$SCRIPT_DIR/m3_k_scaling.py" preflight \
    --config "$CONFIG" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --work-root "$WORK_ROOT" \
    --output "$REPORT_ROOT/preflight.json"
}

build() {
  require_file "$REPORT_ROOT/preflight.json" "Run preflight first"
  "$PYTHON" "$SCRIPT_DIR/m3_k_scaling.py" build \
    --config "$CONFIG" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --work-root "$WORK_ROOT" \
    --output "$REPORT_ROOT/bank_lock.json"
}

analyze() {
  require_file "$REPORT_ROOT/bank_lock.json" "Run build first"
  "$PYTHON" "$SCRIPT_DIR/m3_k_scaling.py" analyze \
    --config "$CONFIG" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --work-root "$WORK_ROOT" \
    --bank-lock "$REPORT_ROOT/bank_lock.json" \
    --output "$REPORT_ROOT/reconstruction_metrics.json"
}

benchmark() {
  require_file "$REPORT_ROOT/bank_lock.json" "Run build first"
  "$PYTHON" "$SCRIPT_DIR/m3_k_scaling.py" benchmark \
    --config "$CONFIG" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --work-root "$WORK_ROOT" \
    --bank-lock "$REPORT_ROOT/bank_lock.json" \
    --output "$REPORT_ROOT/adapter_benchmark.json"
}

decide() {
  require_file "$REPORT_ROOT/reconstruction_metrics.json" "Run analyze first"
  require_file "$REPORT_ROOT/adapter_benchmark.json" "Run benchmark first"
  "$PYTHON" "$SCRIPT_DIR/m3_k_scaling.py" decide \
    --config "$CONFIG" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --work-root "$WORK_ROOT" \
    --metrics "$REPORT_ROOT/reconstruction_metrics.json" \
    --benchmark "$REPORT_ROOT/adapter_benchmark.json" \
    --output "$REPORT_ROOT/decision.json"
}

pack() {
  require_file "$REPORT_ROOT/decision.json" "Run decide first"
  local output="/root/autodl-tmp/coreflow_m3_k_scaling_mini_results.tar.gz"
  tar -czf "$output" \
    -C "$WORK_ROOT" \
    reports/m3_k_scaling_mini banks \
    -C "$ROOT" \
    config provenance docs README.md PACKAGE_MANIFEST.sha256 AutoDL_M3_K扩展最小实验_执行流程.md
  sha256sum "$output" > "$output.sha256"
  ls -lh "$output" "$output.sha256"
}

status() {
  echo "=== M3 K-scaling reports ==="
  for name in preflight.json bank_lock.json reconstruction_metrics.json adapter_benchmark.json decision.json; do
    path="$REPORT_ROOT/$name"
    if [[ -f "$path" ]]; then
      "$PYTHON" - "$path" <<'PY'
import json,sys
p=sys.argv[1]
d=json.load(open(p,encoding="utf-8"))
print(f"{p.split('/')[-1]}: {d.get('status', d.get('decision', 'UNKNOWN'))} {d.get('decision','')}")
PY
    else
      echo "$name: MISSING"
    fi
  done
  echo
  echo "=== bank dirs ==="
  find "$WORK_ROOT/banks" -maxdepth 4 -type f \( -name '*.safetensors' -o -name '*config.json' \) -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort | tail -30 || true
}

if [[ "$PHASE" != "status" ]]; then
  verify_package
fi

case "$PHASE" in
  preflight) preflight ;;
  build) build ;;
  analyze) analyze ;;
  benchmark) benchmark ;;
  decide) decide ;;
  pack) pack ;;
  status) status ;;
  all)
    preflight
    build
    analyze
    benchmark
    decide
    pack
    ;;
  *)
    die "Usage: bash scripts/run_m3_k_scaling.sh {preflight|build|analyze|benchmark|decide|pack|status|all}"
    ;;
esac
