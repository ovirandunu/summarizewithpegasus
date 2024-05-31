#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=30:00:00
#SBATCH --output=npc-trainer.out
#SBATCH -J spred2 

cd $HOME/tm/npc/tmgp/model-trainer

source $HOME/tm/venv/bin/activate

pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr -q
pip uninstall transformers accelerate -y
pip install transformers[torch] accelerate -U

python npc_summaries.py