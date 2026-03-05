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
                    (default: wayformer, scenetransformer, autobot, scenetokens, causal_scenetokens, safe_scenetokens)
  -d <devices>      Devices (e.g. 0 or 0,1)
                    (default: 0)
  -s <strategies>   Strategy/strategies, comma-separated
                    (default: random_drop, token_random_drop, simple_token_jaccard_drop, gumbel_token_jaccard_drop, simple_token_hamming_drop, gumbel_token_hamming_drop)
  -p <percentages>  Percentage(s), comma-separated
                    (default: 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
  -t <type>         Type of sample selection strategy (either of "st", "causalst", "safest")
                    (default: "st")
  -n                Dry run (print commands, do not execute)
  -h                Show this help message

Examples:
  # Run with all defaults
  $0

  # Test specific model and strategy
  $0 -m scenetokens -s random_drop

  # Multiple models and devices
  $0 -m wayformer,scenetransformer -d 0,1

  # Custom strategies
  $0 -s token_random_drop,gumbel_token_hamming_drop

  # Custom percentages
  $0 -p 0.5,0.7,0.9

  # Custom sample selection type
  $0 -t "st"

  # Dry run to preview commands
  $0 -m scenetokens -n
EOF
    exit 1
}

############################
# Defaults
############################
DEFAULT_MODELS=(
    wayformer
    scenetokens
    causal_scenetokens
    safe_scenetokens
    scenetransformer
    autobot
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
DEFAULT_SAMPLE_SELECTION_TYPE="st"
DEFAULT_SAMPLE_SELECTION_PATH="./meta/selection_strategies/scenetokens"
dry_run=false

############################
# Parse arguments
############################
models=()
devices="$DEFAULT_DEVICES"
strategies=()
percentages=()
selection_type="$DEFAULT_SAMPLE_SELECTION_TYPE"

while getopts ":m:d:s:p:t:nh" opt; do
    case $opt in
        m) IFS=',' read -ra models <<< "$OPTARG" ;;
        d) devices="$OPTARG" ;;
        s) IFS=',' read -ra strategies <<< "$OPTARG" ;;
        p) IFS=',' read -ra percentages <<< "$OPTARG" ;;
        t) selection_type="$OPTARG" ;;
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
if [[ "$selection_type" == "st" ]]; then
    selection_path="./meta/selection_strategies/scenetokens"
elif [[ "$selection_type" == "causalst" ]]; then
    selection_path="./meta/selection_strategies/causal-scenetokens"
elif [[ "$selection_type" == "safest" ]]; then
    selection_path="./meta/selection_strategies/safe-scenetokens"
else
    echo "Error: selection_type must be one of 'st', 'causalst', or 'safest', got '$selection_type'" >&2
    exit 1
fi

############################
# Run experiments
############################
for model in "${models[@]}"; do
    for strategy in "${strategies[@]}"; do
        for pct in "${percentages[@]}"; do
            sweep_type="_${strategy}_${pct}_${selection_type}"

            cmd=(
                uv run -m scenetokens.train
                model="$model"
                trainer.devices="[$devices]"
                sample_selection_strategy="$strategy"
                sample_selection_path="$selection_path"
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
