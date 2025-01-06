# aind-smartspim-classification
Code for classifying the cell candidate outputs from aind-smartspim-segmentation within the smartspim pipeline.
It uses the [aind-large-scale-prediction](https://github.com/AllenNeuralDynamics/aind-large-scale-prediction) package to efficiently process large amounts of image data.
This repository takes as input the cell proposals that will be classified by the CellFinder model. The output is a CSV with the following columns:

- Cell Counts: Number of positive cells.
- Cell Likelihood Mean: Mean of the probabilities of a cell being a cell.
- Cell Likelihood STD: Standard deviation of the probability of a cell being a cell.
- Noncell Counts: Number of negative cells.
- Noncell Likelihood Mean: Mean of the probabilities of negative cells.
- Noncell Likelihood STD; Standard deviation of the probabilities of negative cells.
