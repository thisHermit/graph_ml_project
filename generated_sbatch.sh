#!/bin/bash
#SBATCH --job-name=graph_ml_job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4000M
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=job_output_%A_%a.txt
#SBATCH --mail-user=basharjaan.khan@stud.uni-goettingen.de
#SBATCH --mail-type=BEGIN,END

conda activate graph


case $SLURM_ARRAY_TASK_ID in
    0)
        echo "Running baseline (student.py)..."
        python graph_ml_project/student/student.py
        ;;
    1)
        echo "Running teacher2.py..."
        python graph_ml_project/teachers/teacher2.py
        ;;
    2)
        echo "Running teacher1.py..."
        python graph_ml_project/teachers/teacher1.py
        ;;
    *)
        echo "Invalid array index. Nothing to do."
        ;;
esac
