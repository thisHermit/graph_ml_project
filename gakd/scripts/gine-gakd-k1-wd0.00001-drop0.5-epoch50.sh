#!/bin/bash
#SBATCH --job-name=gine-gakd-k1-wd0.00001-drop0.5-epoch50
#SBATCH --partition=grete
#SBATCH -G A100:1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --time=1-23:55:00
#SBATCH --output=/mnt/lustre-grete/projects/LLMticketsummarization/muneeb/rand_dir/GakD/logs/gine-gakd-k1-wd0.00001-drop0.5-epoch50.txt
#SBATCH --mail-user=muneeb.khan@stud.uni-goettingen.de
#SBATCH --mail-type=BEGIN,END

module load cuda/12.2.1

source $HOME/conda/bin/activate gml

BASE_DIR=/mnt/lustre-grete/projects/LLMticketsummarization/muneeb/rand_dir/GakD
TEACHER_KNOWLEDGE_PATH=/mnt/lustre-grete/projects/LLMticketsummarization/muneeb/rand_dir/GraphGPS/teacher_results/teacher-knowledge.pt

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
python gakd.py --teacher_knowledge_path $TEACHER_KNOWLEDGE_PATH --epochs 50 --batch_size 32 --n_runs 1 --student_virtual_node false