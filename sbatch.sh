#!/bin/bash
#SBATCH --job-name=graph_ml_job   # Optional: Set a job name
#SBATCH --partition=gpu              # Specify the partition
#SBATCH --G GTX500  # -G V100:1        # Request 1 GPU
#SBATCH --cpus-per-task=4            # Allocate 4 CPUs per task
#SBATCH --mem-per-cpu=4000M          # Allocate 4000 MB of memory per CPU
#SBATCH --time=02:00:00              # Set a time limit of 1 hour
#SBATCH --output=job_output_%j.txt   # Redirect output to a file named job_output_JOBID.txt
#SBATCH --mail-user=basharjaan.khan@stud.uni-goettingen.de
#SBATCH --mail-type=BEGIN,END               

# Load necessary modules or set up the environment if required
# module load your_module

# TODO: replace python command below
echo "Hello world"