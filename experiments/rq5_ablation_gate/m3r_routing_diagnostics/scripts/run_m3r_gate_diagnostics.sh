#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SOURCE_WORK_ROOT="${SOURCE_WORK_ROOT:-/root/autodl-tmp/coreflow_m0}"
M3_UPLOAD_ROOT="${M3_UPLOAD_ROOT:-/root/autodl-tmp/coreflow_m3_k_scaling_mini_upload}"
M3_WORK_ROOT="${M3_WORK_ROOT:-/root/autodl-tmp/coreflow_m3_k_scaling_mini_workspace}"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/coreflow_m3r_gate_diagnostics_workspace}"
PYTHON="${PYTHON:-$SOURCE_WORK_ROOT/runtime_env/bin/python}"
CONFIG="$ROOT/config/m3r_gate_diagnostics_protocol.json"
REPORT_ROOT="$WORK_ROOT/reports/m3r_gate_diagnostics"
PHASE="${1:-}"

die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
trap 'die "Stopped at line ${BASH_LINENO[0]} (exit $?); partial output was retained and no automatic retry was attempted."' ERR

verify_package() {
  [[ -x "$PYTHON" ]] || die "Frozen runtime Python not found: $PYTHON"
  "$PYTHON" "$SCRIPT_DIR/package_checksums.py" --root "$ROOT"
}

run_analysis() {
  mkdir -p "$REPORT_ROOT"
  "$PYTHON" "$SCRIPT_DIR/m3r_gate_diagnostics.py" \
    --config "$CONFIG" \
    --source-work-root "$SOURCE_WORK_ROOT" \
    --m3-upload-root "$M3_UPLOAD_ROOT" \
    --m3-work-root "$M3_WORK_ROOT" \
    --output "$REPORT_ROOT/gate_diagnostics.json"
}

pack() {
  [[ -f "$REPORT_ROOT/gate_diagnostics.json" ]] || die "Run analyze first"
  local output="/root/autodl-tmp/coreflow_m3r_gate_diagnostics_results.tar.gz"
  tar -czf "$output" \
    -C "$WORK_ROOT" reports/m3r_gate_diagnostics \
    -C "$ROOT" config docs README.md PACKAGE_MANIFEST.sha256 AutoDL_M3R_gate_diagnostics_执行流程.md
  sha256sum "$output" > "$output.sha256"
  ls -lh "$output" "$output.sha256"
}

status() {
  echo "=== M3R reports ==="
  if [[ -f "$REPORT_ROOT/gate_diagnostics.json" ]]; then
    "$PYTHON" - "$REPORT_ROOT/gate_diagnostics.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1],encoding='utf-8'))
print("gate_diagnostics.json:", d.get("status"), d.get("decision"))
for pool,item in d.get("pools",{}).items():
    print(pool, "modules=", item.get("module_count"), "draws/module=", item.get("draws_per_module"), "regime=", item.get("regime_summary"))
PY
  else
    echo "gate_diagnostics.json: MISSING"
  fi
}

if [[ "$PHASE" != "status" ]]; then
  verify_package
fi

case "$PHASE" in
  analyze) run_analysis ;;
  pack) pack ;;
  status) status ;;
  all)
    run_analysis
    pack
    ;;
  *)
    die "Usage: bash scripts/run_m3r_gate_diagnostics.sh {analyze|pack|status|all}"
    ;;
esac
