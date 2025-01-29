# MuGSI: Distilling GNNs with Multi-Granularity Structural Information for Graph Classification

We use a multi-granular approach proposed in the paper to distill knowledge from the teacher model to the student model. We do this by adapting [their code](https://github.com/tianyao-aka/MuGSI) and using our GPS teacher while varying the Student models.
![muGSI](./imgs/Model.png)

- Our experiments here revolve around different student models and their ability to learn .
- MLP plain, MLP+LapPE, GA_MLP (graph augmented mlp) with one-hop neighbours
- All experiments are done on `OGBG-MolPCBA` dataset.

## Setup

- Pre-requisite
  - `GML` Environment setup as described in the root [README.md](../README.md).

1. Create a new directory for the GakD experiments and `logs` directory inside it.
2. Copy [`baseline.py`](./baseline.py), [`mugsi.py`](./gakd.py) and `SBATCH` scripts from [`scripts/*`](./scripts/) directory to the new directory.
3. Modify the `SBATCH` script parameters according to your enviornment. Make sure to set the correct `BASE_DIR` (for all experiments) and `TEACHER_KNOWLEDGE_PATH` with the path to the teacher knowledge file (for `gakd` experiments).
4. The output of the experiments will be available in the `$BASE_DIR/results` directory with the name
   - `gine_results_<dataset_name>_<with/without>_virtual_node.csv` for `baseline` experiments.
   - `gine_student_gakd_<dataset_name>_<with/without>_virtual_node_discriminator_logits_<true/false>_discriminator_embeddings_<true/false>_k<discriminator_update_freq>_wd<student_optimizer_weight_decay>_drop<student_dropout>.csv` for `gakd` experiments.

## Experiments

In our experiments for MuGSI, we use variants of the student models while keeping the same teacher.

We use the following default configuration and mention the change in the particular experiment:

- Number of runs: `5`
- Starting seed: `42`
- Number of layers: `5`
- Hidden dimension: `400`
- Dropout: `0.5`
- Learning rate: `0.001`
- Batch size: `32`
- Epochs: `100`

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
- <why?>
- The results are summarized in following table:
  | Run | Valid AP | Test AP |
  |-----|------------|------------|
  | Baseline | 0.262 | 0.2583 |
- <what do we observe?>
- <inference>
- To reproduce the results, submit the following command via `sbatch`:
  ```
  sbatch scripts/student_baseline_mlp_lappe.sh
  ```

### Experiment #2: GA_MLP

- We trained a simple MLP with one hop encodings
- <why?>
- The results are summarized in following table:
  | Run | Valid AP | Test AP |
  |-----|------------|------------|
  | Baseline | 0.335 | 0.3218 |
- <what do we observe?>
- <inference>
- To reproduce the results, submit the following command via `sbatch`:
  ```
  sbatch scripts/student_baseline_ga_mlp.sh
  ```

### Results
