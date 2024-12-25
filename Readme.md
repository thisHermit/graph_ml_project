# Graph Machine Learning Project

This is project for the course: Introduction to Graph Machine Learning

<What is knowledge distillation>

## Requirements

- pytorch
- pytorch geometric

## Methodology

- Weights and biases
- teacher and student
  - student backbone make node invariant
    - student with gcn layers
    - student with FF layers
- results mean +- std
  - #seeds
  - #random subsample
- node classification techniques x 2
- student baseline with training data and then with training and knowledge distillation
- base model

         /------------- graph transformer\

  input = = concat
  \ ++++++++++++++ MP /
  slide 23, lecture 6

## Tasks

## Running the code

Slurm used

srun command

Or run the jupyter notebook cell by cell

## Datasets

- Peptide func

<What is peptide func>

## Results

# References

- https://arxiv.org/pdf/2403.03483
- https://arxiv.org/pdf/2302.00219
- https://openreview.net/forum?id=vzZ3pbNRvh
- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9053986
- https://paperswithcode.com/sota/graph-classification-on-peptides-func

## Members

- [Bashar Jaan Khan](https://github.com/thishermit)
- [Muneeb Khan](https://github.com/MuneebKhan-7)
- [Shameer](https://github.com/syedshameersarwar)
