#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=10:00:00
#SBATCH --output=pegasus-trainer.out
#SBATCH -J spred2 

cd $HOME/tm/tmgp/model-trainer

source $HOME/tm/tmgp/venv/bin/activate

pip install transformers[sentencepiece] datasets sacrebleu rouge_score py7zr -q
pip uninstall transformers accelerate -y
pip install transformers[torch] accelerate -U

python trainer.py