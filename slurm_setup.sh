# 1) Create and activate new conda environment (Python 3.11)
conda create -n graph python=3.11 -y
conda activate graph

# 2) Install core packages
conda install -y numpy matplotlib scipy

# 3) Install PyTorch (GPU version example with CUDA 12.1)
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 4) Install PyTorch Geometric (using conda channel 'pyg')
conda install -y pyg -c pyg

# 5) Install datasets
 conda install -y conda-forge::ogb 
