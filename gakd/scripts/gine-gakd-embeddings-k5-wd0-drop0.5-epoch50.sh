#!/bin/bash
#SBATCH --job-name=gine-gakd-embeddings-k5-wd0-drop0.5-epoch50
#SBATCH --partition=grete-h100
#SBATCH -G H100:1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --time=1-23:55:00
#SBATCH --output=~/GakD/logs/gine-gakd-embeddings-k5-wd0-drop0.5-epoch50.txt
#SBATCH --mail-user=muneeb.khan@stud.uni-goettingen.de
#SBATCH --mail-type=BEGIN,END

module load cuda/12.2.1

source $HOME/conda/bin/activate gml

BASE_DIR=~/GakD
TEACHER_KNOWLEDGE_PATH=~/GraphGPS/teacher_results/teacher-knowledge.pt

echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Active Python: $(which python)"
echo "Python version: $(python --version)"
echo "==========================================="

export PYTHONUNBUFFERED=TRUE
cd $BASE_DIR
# results will be available in $BASE_DIR/results
python gakd.py --teacher_knowledge_path $TEACHER_KNOWLEDGE_PATH --epochs 50 --batch_size 32 --n_runs 2 --student_virtual_node false --student_optimizer_weight_decay 0 --student_dropout 0.5 --discriminator_update_freq 5 --train_discriminator_logits false