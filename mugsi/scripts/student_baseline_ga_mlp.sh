#!/bin/bash
#SBATCH --job-name=gwdg_kd_all_teacher_knowledge_pt
#SBATCH --partition=grete-h100
#SBATCH -G H100:1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --output=~/mugsi/logs/gml_teacher_knowledge_pt_all.txt
#SBATCH --mail-user=muneeb.khan@stud.uni-goettingen.de
#SBATCH --mail-type=BEGIN,END

module load cuda/12.2.1

source $HOME/conda/bin/activate new_gml

echo "==========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Active Python: $(which python)"
echo "Python version: $(python --version)"
echo "==========================================="

export PYTHONUNBUFFERED=TRUE
cd ~/mugsi

python student_baseline.py  --model_name MLP  --use_lappe --use_khop --max_epochs 30 --batch_size 256 --lr 1e-3 --weight_decay 1e-6 --pooling_method sum

