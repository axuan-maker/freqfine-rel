#!/usr/bin/env bash
#
# run_cloud.sh — 云端自动化流水线（按 v1.0 规范）。
# 流程：激活 conda -> 特征提取 -> H1b -> SVM 主方法 -> CNN 基线 -> 关机。

set -uo pipefail

CONDA_ENV_NAME="${CONDA_ENV_NAME:-freqfine}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PROCESSED_ROOT="${PROCESSED_ROOT:-processed/v1.0}"
MODALITY="${MODALITY:-audio}"
DATASET="${DATASET:-SEUMLD}"
OUT_DIR="${OUT_DIR:-results}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-1}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-/usr/bin/shutdown -h now}"

LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"
_SHUTDOWN_CALLED=0

log() {
    echo "[$(date '+%F %T')] $*" | tee -a "$LOG_FILE"
}

activate_conda() {
    if command -v conda >/dev/null 2>&1; then
        local conda_bin conda_sh
        conda_bin="$(command -v conda)"
        conda_sh="$(dirname "$conda_bin")/../etc/profile.d/conda.sh"
        if [[ -f "$conda_sh" ]]; then
            # shellcheck disable=SC1090
            source "$conda_sh"
        else
            # shellcheck disable=SC1091
            source "$(dirname "$conda_bin")/activate" "$CONDA_ENV_NAME"
            return
        fi
        conda activate "$CONDA_ENV_NAME"
        log "已激活 conda 环境: $CONDA_ENV_NAME"
    else
        log "WARN: 未找到 conda，使用系统 $PYTHON_BIN"
    fi
}

shutdown_now() {
    local code="${1:-0}"
    if [[ "$_SHUTDOWN_CALLED" == "1" ]]; then
        return
    fi
    _SHUTDOWN_CALLED=1
    log "准备关机 (exit=$code): $SHUTDOWN_CMD"
    if [[ "$AUTO_SHUTDOWN" == "1" ]]; then
        $SHUTDOWN_CMD >>"$LOG_FILE" 2>&1 \
            || sudo -n $SHUTDOWN_CMD >>"$LOG_FILE" 2>&1 \
            || log "ERROR: 关机命令执行失败，请手动关机"
    else
        log "AUTO_SHUTDOWN=0，跳过关机"
    fi
}

run_step() {
    local name="$1"
    shift
    local t0 t1 rc
    t0=$(date +%s)
    log ">>> 开始: $name"
    "$@" >>"$LOG_FILE" 2>&1
    rc=$?
    t1=$(date +%s)
    log "<<< 结束: $name (exit=$rc, 耗时=$((t1 - t0))s)"
    return "$rc"
}

main() {
    local status=0
    log "========== 流水线启动 =========="
    activate_conda

    run_step "1_cfc_extractor" "$PYTHON_BIN" cfc_extractor.py \
        --processed_root "$PROCESSED_ROOT" --modality "$MODALITY" \
        --dataset "$DATASET" --out_dir "$OUT_DIR/features" \
        --coupling AAC --n_subbands 8 || status=$?

    run_step "2_h1a" "$PYTHON_BIN" h1a.py \
        --features_dir "$OUT_DIR/features" --out_dir "$OUT_DIR/h1a" || status=$?

    run_step "3_h1b" "$PYTHON_BIN" h1b.py \
        --features_dir "$OUT_DIR/features" --out_dir "$OUT_DIR/h1b" \
        --model svm --weighting hsic || status=$?

    run_step "4_train_eval" "$PYTHON_BIN" train_eval.py \
        --features_dir "$OUT_DIR/features" --condition relation \
        --model svm --weighting hsic || status=$?

    run_step "5_dl_baseline" "$PYTHON_BIN" dl_baseline.py \
        --features_dir "$OUT_DIR/features" || status=$?

    log "========== 流水线结束 (exit=$status) =========="
    return "$status"
}

trap 'shutdown_now 1; exit 1' INT TERM
trap 'shutdown_now $?' EXIT

main
exit $?
