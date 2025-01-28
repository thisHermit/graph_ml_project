# Knowledge Distillation on Graphs
This repository contains the code and setup instructions for the project on Knowledge Distillation on Graphs as part of the course **Introduction to Graph Machine Learning**.

- We applied two different knowledge distillation frameworks on **OGBG-MolPCBA** dataset.
- We exclusively worked on Graph Classification tasks as the amount of literature on knowledge distillation for Graph Classification is limited and often focused on Node Classification tasks.
- The two frameworks are:
    - **GAKD**: [Graph Adversarial Knowledge Distillation](https://arxiv.org/pdf/2205.11678)
    - **MuGSI**: [Distilling GNNs with Multi-Granularity Structural Information for Graph Classification](https://dl.acm.org/doi/10.1145/3589334.3645542)
- We used the `A100` and `H100` GPUs for training and inference.
### Project Structure
- **`teacher/`**: Contains the instructions for producing the teacher knowledge on the `OGBG-MolPCBA` dataset. We used [`GraphGPS`](https://github.com/rampasek/GraphGPS) framework to build the teacher knowledge.
- **`gakd/`**: Contains the code, setup instructions and results for the `GAKD` framework.
- **`mugsi/`**: Contains the code, setup instructions and results for the `MuGSI` framework.

### Environment Setup

- #### `GML` Environment: (for `teacher` and `gakd` experiments)
  - Login to Cluster and Install `Miniforge`
    ```
    wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash Miniforge3.sh -b -p "${HOME}/conda"
    source "${HOME}/conda/etc/profile.d/conda.sh"
    conda activate
    ```
  - Create a new root directory for entire project. 
    ```
    mkdir -p $HOME/knowledge-distillation-on-graphs
    cd $HOME/knowledge-distillation-on-graphs
    ```
  - Setup the `conda` based `gml` environment with dependencies.
    ```
    conda create -n gml python=3.11 -y
    conda activate gml
    pip install  torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
    pip install torch-scatter torch-sparse torch-scatter torch-geometric ogb  -f https://data.pyg.org/whl/torch-2.4.1+cu121.html --no-cache-dir
    pip install numpy matplotlib scipy pandas
    pip install pyg-lib -f https://data.pyg.org/whl/torch-2.4.1+cu121.html --no-cache-dir
    conda install openbabel fsspec rdkit -c conda-forge -y
    pip install pytorch-lightning yacs torchmetrics
    pip install performer-pytorch
    pip install tensorboardX
    pip install ogb
    pip install wandb
    ```
  - Now you can move to the `teacher` directory and follow the [instructions](./teacher/README.md) to build the teacher knowledge.
  - After establishing the teacher knowledge, you can move to the `gakd` directory and follow the [instructions](./gakd/README.md) to perform the GAKD experiments.
  - The `sbatch` scripts of both experiments assumes the `gml` environment presence and activates them before running the experiment `python` script via:
    ```
    source $HOME/conda/bin/activate gml
    ```

- #### `MuGSI` Environment: (for `mugsi` experiments)









    


