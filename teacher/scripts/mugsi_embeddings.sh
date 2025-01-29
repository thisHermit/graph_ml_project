#!/bin/bash
#SBATCH --job-name=embeddings_gml_gwdg_gps_teacher
#SBATCH --partition=grete-h100
#SBATCH -G H100:1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --output=~/logs/gml_gps_teacher_embeddings.txt
#SBATCH --mail-user=muneeb.khan@stud.uni-goettingen.de
#SBATCH --mail-type=BEGIN,END

module load cuda/12.2.1

source $HOME/conda/bin/activate gml
export BASE_DIR=~/GraphGPS
export SEED=44
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
# Will used pretained model from $teacher_results/$SEED and extract embeddings -> $teacher_results/node-embeddings.pt
python mugsi_embeds.py --cfg embedding_config.yaml seed $SEED embeddings.type node
# Will used pretained model from $teacher_results/$SEED and extract embeddings -> $teacher_results/logits-embeddings.pt
python embedding.py --cfg embedding_config.yaml seed $SEED embeddings.type logits
