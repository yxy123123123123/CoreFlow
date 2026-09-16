#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
WORK_ROOT="${WORK_ROOT:-/root/autodl-tmp/coreflow_applsci_phase2b_workspace}"
MODEL="${MODEL:-/root/autodl-tmp/Model}"
SOURCE="${SOURCE:-/root/autodl-tmp/coreflow_m0}"
PY="${PY:-$SOURCE/runtime_env/bin/python}"
CONFIG="$ROOT/config/protocol.json"
DEV="$WORK_ROOT/data/dev_sets"
BANKS="$WORK_ROOT/data/qgrid_banks"
REPORTS="$WORK_ROOT/reports/phase2b"
RESULTS="$WORK_ROOT/results/phase2b"
LOG="$WORK_ROOT/logs"
PHASE="${1:-}"

die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }
verify() { "$PY" "$SCRIPT_DIR/package_checksums.py" --root "$ROOT" || die "manifest mismatch"; }

preflight() {
  verify
  mkdir -p "$LOG"
  "$PY" "$SCRIPT_DIR/preflight_phase2b.py" 2>&1 | tee "$LOG/01_preflight.log"
}

prepare() {
  verify
  mkdir -p "$DEV" "$BANKS" "$LOG"
  "$PY" "$SCRIPT_DIR/prepare_phase2b.py" --work-root "$WORK_ROOT" --source-assets "$SOURCE" \
    --output-dev "$DEV" 2>&1 | tee "$LOG/02_prepare.log"
  "$PY" "$SCRIPT_DIR/build_qgrid_banks.py" --source-assets "$SOURCE" --vendor-root "$SOURCE/vendor/LoRAFlow" \
    --output-root "$BANKS" 2>&1 | tee "$LOG/02b_build_qbanks.log"
}

seal() {
  verify
  [[ -f "$DEV/dev_set_audit.json" ]] || die "Run prepare first"
  mkdir -p "$REPORTS" "$LOG"
  "$PY" "$SCRIPT_DIR/seal_phase2b.py" --config "$CONFIG" --dev-data "$DEV" \
    --output "$REPORTS/PROTOCOL_SEAL.json" 2>&1 | tee "$LOG/03_seal.log"
}

b1_qgrid() {
  [[ -f "$REPORTS/PROTOCOL_SEAL.json" ]] || die "Run seal first"
  verify; mkdir -p "$RESULTS/b1_qgrid" "$LOG"
  "$PY" "$SCRIPT_DIR/run_qgrid.py" --work-root "$WORK_ROOT" --model "$MODEL" --source-assets "$SOURCE" \
    --vendor-root "$SOURCE/vendor/LoRAFlow" --dev-data "$DEV" --banks "$BANKS" \
    --output-root "$RESULTS/b1_qgrid" 2>&1 | tee "$LOG/04_b1_qgrid.log"
}

b3_prefill() {
  [[ -f "$REPORTS/PROTOCOL_SEAL.json" ]] || die "Run seal first"
  verify; mkdir -p "$RESULTS/b3_prefill" "$LOG"
  cfg_py="$PY -c"
  order="zh,ru,es,math,code"
  gate="$SOURCE/official_assets/Gates/code_k5_code_seed41.pt"
  bank="$BANKS/q185_bf16"
  for config in "1 128" "1 512" "1 1024" "2 512" "4 512"; do
    B=$(echo $config | awk '{print $1}'); L=$(echo $config | awk '{print $2}')
    for method in full_vectorized core_q185; do
      MB="$method"
      if [ "$method" = "core_q185" ]; then MB="core_q185"; fi
      "$PY" "$SCRIPT_DIR/run_prefill.py" --work-root "$WORK_ROOT" --model "$MODEL" \
        --source-assets "$SOURCE" --vendor-root "$SOURCE/vendor/LoRAFlow" \
        --method "$method" --bank "$bank" --gate "$gate" --batch "$B" --input-length "$L" \
        --output "$RESULTS/b3_prefill/${method}_b${B}_l${L}.json" 2>&1
    done
  done | tee "$LOG/05_b3_prefill.log"
}

analyze() {
  verify; mkdir -p "$REPORTS" "$LOG"
  "$PY" "$SCRIPT_DIR/analyze_phase2b.py" --config "$CONFIG" --result-root "$RESULTS" \
    --output "$REPORTS/decision.json" 2>&1 | tee "$LOG/08_analyze.log"
}

pack() {
  [[ -f "$REPORTS/decision.json" ]] || die "No decision; complete analyze first"
  cd "$WORK_ROOT"
  tar -czf /root/autodl-tmp/coreflow_applsci_phase2b_results.tar.gz \
    reports/phase2b results/phase2b data/dev_sets data/qgrid_banks
  sha256sum /root/autodl-tmp/coreflow_applsci_phase2b_results.tar.gz \
    > /root/autodl-tmp/coreflow_applsci_phase2b_results.tar.gz.sha256
  echo "PACKED: /root/autodl-tmp/coreflow_applsci_phase2b_results.tar.gz"
}

case "$PHASE" in
  preflight) preflight ;;
  prepare) prepare ;;
  seal) seal ;;
  b1) b1_qgrid ;;
  b3) b3_prefill ;;
  analyze) analyze ;;
  pack) pack ;;
  *) die "Usage: $0 {preflight|prepare|seal|b1|b3|analyze|pack}" ;;
esac