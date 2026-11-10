#!/bin/bash
#SBATCH --job-name=p2t_all_eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=gpu01
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=all_evaluations_%j.out
#SBATCH --error=all_evaluations_%j.err
 
set -euo pipefail
 
flight env activate conda
 
cd "$HOME/p2t_project/Code"
 
echo "========================================"
echo "P2T SAVED-MODEL EVALUATIONS"
echo "Start: $(date)"
echo "Host: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "========================================"
 
nvidia-smi
 
test -f inference.py
test -f evaluation.py
 
 
run_experiment () {
 
    NAME="$1"
    MODEL="$2"
    TEST="$3"
    RESULT="$4"
    INFERENCE_METRICS="$5"
    EVAL_DIR="$6"
 
    echo
    echo "========================================"
    echo "STARTING: $NAME"
    echo "Model: $MODEL"
    echo "Test: $TEST"
    echo "Result: $RESULT"
    echo "========================================"
 
    if [ ! -f "$MODEL/model.safetensors" ]; then
        echo "ERROR: Model not found: $MODEL/model.safetensors"
        exit 1
    fi
 
    if [ ! -f "$TEST" ]; then
        echo "ERROR: Test file not found: $TEST"
        exit 1
    fi
 
    rm -rf "$EVAL_DIR"
 
    conda run --no-capture-output -n p2t \
        python -u inference.py \
        --model_path "$MODEL" \
        --test_csv "$TEST" \
        --output_file "$RESULT" \
        --metrics_file "$INFERENCE_METRICS"
 
    conda run --no-capture-output -n p2t \
        python -u evaluation.py \
        --results_csv "$RESULT" \
        --output_dir "$EVAL_DIR" \
        --top_k 10
 
    echo
    echo "===== FINAL METRICS: $NAME ====="
    cat "$EVAL_DIR/evaluation_metrics.json"
 
    echo
    echo "COMPLETED: $NAME"
    echo "Time: $(date)"
}
 
 
run_experiment \
    "T5 Large" \
    "T5_large/final_merged_model" \
    "T5_large/test.csv" \
    "T5_large/new_test_results.csv" \
    "T5_large/new_inference_metrics.json" \
    "T5_large/new_evaluation"
 
 
run_experiment \
    "T5 Base" \
    "T5_base/final_merged_model" \
    "T5_large/test.csv" \
    "T5_base/new_test_results.csv" \
    "T5_base/new_inference_metrics.json" \
    "T5_base/new_evaluation"
 
 
run_experiment \
    "T5 XL" \
    "T5_XL/final_merged_model" \
    "T5_large/test.csv" \
    "T5_XL/new_test_results.csv" \
    "T5_XL/new_inference_metrics.json" \
    "T5_XL/new_evaluation"
 
 
run_experiment \
    "T5 Large - 10% Noisy Input" \
    "T5_large/final_merged_model" \
    "e_10_20_30/test_noisy_10PER.csv" \
    "e_10_20_30/new_test_results_10PER.csv" \
    "e_10_20_30/new_inference_metrics_10PER.json" \
    "e_10_20_30/new_evaluation_10PER"
 
 
run_experiment \
    "T5 Large - 20% Noisy Input" \
    "T5_large/final_merged_model" \
    "e_10_20_30/test_noisy_20PER.csv" \
    "e_10_20_30/new_test_results_20PER.csv" \
    "e_10_20_30/new_inference_metrics_20PER.json" \
    "e_10_20_30/new_evaluation_20PER"
 
 
run_experiment \
    "T5 Large - 30% Noisy Input" \
    "T5_large/final_merged_model" \
    "e_10_20_30/test_noisy_30PER.csv" \
    "e_10_20_30/new_test_results_30PER.csv" \
    "e_10_20_30/new_inference_metrics_30PER.json" \
    "e_10_20_30/new_evaluation_30PER"
 
 
run_experiment \
    "T5 Large - Stress + Tokens" \
    "stress_tokens+stress_pipe/stress_tokens/final_merged_model" \
    "stress_tokens+stress_pipe/stress_tokens/test_b.csv" \
    "stress_tokens+stress_pipe/stress_tokens/new_test_results.csv" \
    "stress_tokens+stress_pipe/stress_tokens/new_inference_metrics.json" \
    "stress_tokens+stress_pipe/stress_tokens/new_evaluation"
 
 
run_experiment \
    "T5 Large - Stress + Pipe" \
    "stress_tokens+stress_pipe/stress_pipe/final_merged_model" \
    "stress_tokens+stress_pipe/stress_pipe/test_c.csv" \
    "stress_tokens+stress_pipe/stress_pipe/new_test_results.csv" \
    "stress_tokens+stress_pipe/stress_pipe/new_inference_metrics.json" \
    "stress_tokens+stress_pipe/stress_pipe/new_evaluation"
 
 
echo
echo "========================================"
echo "ALL 8 EVALUATIONS COMPLETED"
echo "End: $(date)"
echo "========================================"
