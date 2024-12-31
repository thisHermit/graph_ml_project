#!/usr/bin/env python3
import os

def generate_slurm_script(teachers_path='graph_ml_project/teachers'):
    """
    Generates an sbatch Slurm script that:
      - Runs student.py at array index 0 (via python -m graph_ml_project.student.student).
      - Runs each teacher .py file (via python -m graph_ml_project.teachers.<teacher>).
    """
    # 1) Collect all .py files in teachers/ directory
    teachers = [f for f in os.listdir(teachers_path) if f.endswith('.py') and f != '__init__.py']
    
    # Total tasks = 1 (for student.py) + number_of_teacher_scripts
    total_tasks = len(teachers)

    # 2) Build the Slurm script lines
    script_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=graph_ml_job",
        "#SBATCH --partition=gpu",
        "#SBATCH -G V100:1",
        "#SBATCH --cpus-per-task=4",
        "#SBATCH --mem-per-cpu=4000M",
        "#SBATCH --time=02:00:00",
        f"#SBATCH --array=0-{total_tasks}",
        "#SBATCH --output=logs/job_output_%A_%a.txt",
        "#SBATCH --mail-user=basharjaan.khan@stud.uni-goettingen.de",
        "#SBATCH --mail-type=BEGIN,END",
        "",
        "module load cuda/12.1.1",
        "# Source the conda script that sets up 'conda activate' in the current shell",
        "source $HOME/miniforge3/bin/activate graph",
        "",
        "echo \"===========================================\"",
        "echo \"Active Python: $(which python)\"",
        "echo \"Python version: $(python --version)\"",
        "echo \"===========================================\"",
        "",
        "case $SLURM_ARRAY_TASK_ID in",
        "    0)",
        "        echo \"Running baseline (student.py)...\"",
        "        python -m graph_ml_project.student.student",
        "        ;;"
    ]

    # 3) Add cases for each teacher script
    #    We run them as a module (python -m graph_ml_project.teachers.teacherN)
    for i, teacher_file in enumerate(teachers, start=1):
        # Remove ".py" extension for the module name
        module_name = teacher_file[:-3]
        script_lines += [
            f"    {i})",
            f"        echo \"Running {teacher_file}...\"",
            f"        python -m graph_ml_project.teachers.{module_name}",
            "        ;;"
        ]

    # 4) Fallback case
    script_lines += [
        "    *)",
        "        echo \"Invalid array index. Nothing to do.\"",
        "        ;;",
        "esac",
        ""
    ]

    # 5) Return as a single string
    return "\n".join(script_lines)

if __name__ == "__main__":
    slurm_script = generate_slurm_script()

    output_filename = "generated_sbatch.sh"
    with open(output_filename, "w") as f:
        f.write(slurm_script)

    print(f"Slurm script generated: {output_filename}")
