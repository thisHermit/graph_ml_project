#!/usr/bin/env python3

import os

def generate_slurm_script(teachers_path='graph_ml_project/teachers'):
    """
    Generates a Slurm script that runs student.py at array index 0,
    and each Python file in 'teachers_path' at subsequent array indices.
    """
    # 1) Collect all .py files in teachers directory
    teachers = [
        f for f in os.listdir(teachers_path)
        if f.endswith('.py') and f != '__init__.py'
    ]

    total_tasks = len(teachers)

    # 2) Build script lines
    script_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=graph_ml_job",
        "#SBATCH --partition=gpu",          # or your GPU partition name
        "#SBATCH --gres=gpu:1",             # Request 1 GPU (modify as needed)
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem-per-cpu=4000M",
        "#SBATCH --time=02:00:00",
        f"#SBATCH --array=0-{total_tasks}",
        "#SBATCH --output=logs/job_output_%A_%a.txt",
        "#SBATCH --mail-user=basharjaan.khan@stud.uni-goettingen.de",
        "#SBATCH --mail-type=BEGIN,END",
        "",
        "module load cuda/12.1.1",
        "module load miniforge3",
        "# Source the conda script that sets up 'conda activate' in the current shell",
        "source \"$(conda info --base)/etc/profile.d/conda.sh\"",
        "conda activate graph",
        "",
        "",
        "case $SLURM_ARRAY_TASK_ID in",
        "    0)",
        "        echo \"Running baseline (student.py)...\"",
        "        python graph_ml_project/student/student.py",
        "        ;;"
    ]

    # 3) Add cases for teacher scripts (indices 1..N)
    for i, teacher_file in enumerate(teachers, start=1):
        script_lines += [
            f"    {i})",
            f"        echo \"Running {teacher_file}...\"",
            f"        python graph_ml_project/teachers/{teacher_file}",
            "        ;;"
        ]

    # Fallback case
    script_lines += [
        "    *)",
        "        echo \"Invalid array index. Nothing to do.\"",
        "        ;;",
        "esac",
        ""
    ]

    return "\n".join(script_lines)

if __name__ == "__main__":
    # Generate the script content
    slurm_script = generate_slurm_script()

    # Write it to file
    output_filename = "generated_sbatch.sh"
    with open(output_filename, "w") as f:
        f.write(slurm_script)

    print(f"Slurm script generated in {output_filename}")
