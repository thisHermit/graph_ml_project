#!/bin/bash
#SBATCH --job-name=gml_gwdg_gps_teacher
#SBATCH --partition=grete-h100
#SBATCH -G H100:1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/lustre-grete/projects/LLMticketsummarization/muneeb/rand_dir/logs/gml_gps_teacher.txt
#SBATCH --mail-user=muneeb.khan@stud.uni-goettingen.de
#SBATCH --mail-type=BEGIN,END

module load cuda/12.2.1

source $HOME/conda/bin/activate gml
export BASE_DIR=/mnt/lustre-grete/projects/LLMticketsummarization/muneeb/rand_dir/GraphGPS

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
python main.py --cfg gps_teacher_config.yaml --repeat 3 seed 42
