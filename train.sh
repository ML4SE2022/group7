#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --time=02:00:00
#SBATCH --mem=32000
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

set -x
PY_ARGS=${@:1}

module load Python/3.7.4-GCCcore-8.3.0
cd /home/p307534/dan/group7
source /home/p307534/dan/group7/venv/bin/activate

srun python code/run_classifier.py --model_type roberta --do_train --do_eval --eval_all_checkpoints --train_file train_codesearchnet_7_short.json --dev_file dev_codesearchnet.json --max_seq_length 200 --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 1e-5 --num_train_epochs 3 --gradient_accumulation_steps 1 --warmup_steps 1000 --evaluate_during_training --data_dir ./data/ --output_dir model_codesearchnet --encoder_name_or_path microsoft/codebert-base
