#!/usr/bin/env bash
set -euo pipefail

############################
# Usage
############################
usage() {
    cat <<EOF
Usage: $0 [options]

Options:
  -m <models>       Model(s), comma-separated
                    (default: wayformer, scenetransformer, scenetokens_student, causal_scenetokens, safe_scenetokens)
  -d <devices>      Devices (e.g. 0 or 0,1)
                    (default: 0)
  -s <strategies>   Strategy/strategies, comma-separated
                    (default: random_drop, token_random_drop, simple_token_jaccard_drop, gumbel_token_jaccard_drop, simple_token_hamming_drop, gumbel_token_hamming_drop)
  -p <percentages>  Percentage(s), comma-separated
                    (default: 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
  -n                Dry run (print commands, do not execute)
  -h                Show this help message

Examples:
  # Run with all defaults
  $0

  # Test specific model and strategy
  $0 -m scenetokens_student -s random_drop

  # Multiple models and devices
  $0 -m wayformer,scenetransformer -d 0,1

  # Custom strategies
  $0 -s token_random_drop,gumbel_token_hamming_drop

  # Custom percentages
  $0 -p 0.5,0.7,0.9

  # Dry run to preview commands
  $0 -m scenetokens_student  -n
EOF
    exit 1
}

############################
# Defaults
############################
DEFAULT_MODELS=(
    wayformer
    scenetokens_student
    causal_scenetokens
    safe_scenetokens
    scenetransformer
)
DEFAULT_DEVICES="0"
DEFAULT_STRATEGIES=(
  random_drop
  token_random_drop
  simple_token_jaccard_drop
  simple_token_hamming_drop
  gumbel_token_jaccard_drop
  gumbel_token_hamming_drop
)
DEFAULT_PERCENTAGES=(0.45 0.55 0.65 0.75 0.85 0.95)

dry_run=false

############################
# Parse arguments
############################
models=()
devices="$DEFAULT_DEVICES"
strategies=()
percentages=()

while getopts ":m:d:s:p:nh" opt; do
    case $opt in
        m) IFS=',' read -ra models <<< "$OPTARG" ;;
        d) devices="$OPTARG" ;;
        s) IFS=',' read -ra strategies <<< "$OPTARG" ;;
        p) IFS=',' read -ra percentages <<< "$OPTARG" ;;
        n) dry_run=true ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

############################
# Apply defaults if empty
############################
[[ ${#models[@]} -eq 0 ]] && models=("${DEFAULT_MODELS[@]}")
[[ ${#strategies[@]} -eq 0 ]] && strategies=("${DEFAULT_STRATEGIES[@]}")
[[ ${#percentages[@]} -eq 0 ]] && percentages=("${DEFAULT_PERCENTAGES[@]}")

############################
# Run experiments
############################
for model in "${models[@]}"; do
    for strategy in "${strategies[@]}"; do
        for pct in "${percentages[@]}"; do
            sweep_type="_${strategy}_${pct}"

            cmd=(
                uv run -m scenetokens.train
                model="$model"
                trainer.devices="[$devices]"
                sample_selection_strategy="$strategy"
                sample_selection_path="./meta"
                percentage="$pct"
                sweep_type="$sweep_type"
            )

            echo "--------------------------------------------------"
            echo "Model:      $model"
            echo "Devices:    $devices"
            echo "Strategy:   $strategy"
            echo "Percentage: $pct"
            echo

            if $dry_run; then
                echo "[DRY RUN] ${cmd[*]}"
            else
                "${cmd[@]}"
            fi
        done
    done
done
