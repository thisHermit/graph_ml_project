# MuGSI: Distilling GNNs with Multi-Granularity Structural Information for Graph Classification

We use a multi-granular approach proposed in the paper to distill knowledge from the teacher model to the student model. We do this by adapting [their code](https://github.com/tianyao-aka/MuGSI) and using our GPS teacher while varying the Student models.
![muGSI](./imgs/Model.png)

- Our experiments here revolve around different student models and their ability to learn .
- MLP plain, MLP+LapPE, GA_MLP (graph augmented mlp) with one-hop neighbours
- All experiments are done on `OGBG-MolPCBA` dataset.

## Setup

Run the files in the scripts directory

## Experiments

In our experiments for MuGSI, we use variants of the student models while keeping the same teacher.

We use the following default configuration and mention the change in the particular experiment:

### Experiment #1: MLP

- We trained a simple MLP with the following configuration.

- We wanted to see the raw performance of the simplest student model possible.
- The results are summarized in following table:
  | Run | Valid AP | Test AP |
  |-----|------------|------------|
  | Baseline | 0.261 | 0.2568 |
  | KD | 0.074 | ? |
- The baseline for the simplest MLP is reasonable high indicating that the molecular tasks are very determined by the types of nodes present.
- This makes us infer that node features may play a more important role and node feature augmentation should provide better results.
- To reproduce the results, submit the following command via `sbatch`:
  ```
  sbatch scripts/student_baseline_mlp.sh
  ```

### Experiment #2: MLP+LapPE

- We trained a simple MLP with LapPE
- The results are summarized in following table:
  | Run | Valid AP | Test AP |
  |-----|------------|------------|
  | Baseline | 0.262 | 0.2583 |
- To reproduce the results, submit the following command via `sbatch`:
  ```
  sbatch scripts/student_baseline_mlp_lappe.sh
  ```

### Experiment #2: GA_MLP

- We trained a simple MLP with one hop encodings
- The results are summarized in following table:
  | Run | Valid AP | Test AP |
  |-----|------------|------------|
  | Baseline | 0.335 | 0.3218 |
- To reproduce the results, submit the following command via `sbatch`:
  ```
  sbatch scripts/student_baseline_ga_mlp.sh
  ```

### Results
