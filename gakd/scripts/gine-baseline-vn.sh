#!/bin/bash
#SBATCH --job-name=gine-gakd-embeddings-k5-wd0-drop0.5-epoch50
#SBATCH --partition=grete-h100
#SBATCH -G H100:1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --time=1-23:55:00
#SBATCH --output=/mnt/lustre-grete/projects/LLMticketsummarization/muneeb/rand_dir/GakD/logs/gine-gakd-embeddings-k5-wd0-drop0.5-epoch50.txt
#SBATCH --mail-user=muneeb.khan@stud.uni-goettingen.de
#SBATCH --mail-type=BEGIN,END

module load cuda/12.2.1

source $HOME/conda/bin/activate gml

BASE_DIR=/mnt/lustre-grete/projects/LLMticketsummarization/muneeb/rand_dir/GakD

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
python baseline.py --virtual_node true
