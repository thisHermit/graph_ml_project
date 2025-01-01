#!/bin/bash
#SBATCH --job-name=graph_ml_job
#SBATCH --partition=gpu
#SBATCH -G V100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/job_output_%A_%a.txt
#SBATCH --mail-user=basharjaan.khan@stud.uni-goettingen.de
#SBATCH --mail-type=BEGIN,END

module load cuda/12.1.1
# Source the conda script that sets up 'conda activate' in the current shell
source $HOME/miniforge3/bin/activate graph

echo "==========================================="
echo "Active Python: $(which python)"
echo "Python version: $(python --version)"
echo "==========================================="

case $SLURM_ARRAY_TASK_ID in
    0)
        echo "Running baseline (student.py)..."
        python -m graph_ml_project.student.student
        ;;
    1)
        echo "Running teacher2.py..."
        python -m graph_ml_project.teachers.teacher2
        ;;
    2)
        echo "Running teacher1.py..."
        python -m graph_ml_project.teachers.teacher1
        ;;
    *)
        echo "Invalid array index. Nothing to do."
        ;;
esac
