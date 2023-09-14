---
title: Pytorch Connectomics Learning Process Record
date: 2023-09-06 09:20:00 -0700
categories: [Machine Learning, Image Classification]
tags: [Project]     # TAG names should always be lowercase
---
# Learning Plan
I aim to train models in Pytorch Connectomics software application using the brain images and build a reliable pipeline for future use in electronical microscopy images. Firstly, to approach the Pytorch Connectomics, I start to read the published paper: PyTorch Connectomics: A Scalable and Flexible Segmentation Framework for EM Connectomics (https://arxiv.org/abs/2112.05754). In addition, I also browse through the official website (https://connectomics.readthedocs.io/en/latest/). After understanding the most terminologies and basic concepts, I continue to build a workflow as a blueprint for my future exploration.

## Paper Summary
PyTorch Connectomics (PyTC) is an opensource deep-learning framework for the semantic and instance segmentation of volumetric microscopy images, built upon PyTorch. It includes the semantic and instance segementation in Neurons, Synapses, Mitochondria, and Nuclei, and Artifacts. 

### System Design
They focus on data scalability that can handle datasets of different scales, model flexibility for learning
multiple targets simultaneously with various loss functions.
The system is supported by GPU and CPU parallelism.

**Data Scalability**: *Data Loading*, *Data Augmentation* 

**Model Flexibility**: *Hybrid-representation learning*, *Active and Semi-supervised Learning*, *Network Architectures*

**Training and Inference Parallelism**: *Distributed Training*, *Inference Parallelism*

### Experiment
**Synaptic Cleft Detection**: We first trained a customized 3D U-Net model using an SGD optimizer with linear warmup and cosine annealing in the learning rate scheduler. We use a weighted binary crossentropy (BCE) loss and applied rejection sampling to reject samples without synapse during training with a probability of 95% to penalize false negatives. The model input size
is 257 × 257 × 17 in (x, y, z) as CREMI is an anisotropic dataset with higher x and y resolution. The model was optimized for 150K iterations with a batch size of 6 and a base learning rate of 0.02. To further improve the performance, we also use a semi-supervised learning approach called self-training, which generates pseudo-labels on unlabeled images and combines labeled and pseudo-labeled data together in model finetuning. We optimize the model again using the same protocol for 150K iterations.

**Mitochondria Semantic Segmentation**: Similar to the CREMI experiments, the model output is the probability map of mitochondria. Since this dataset is isotropic (each voxel is a cube), we use an input size of 112 × 112 × 112, and we only use 3D convolutional filters in our custom 3D U-Net architecture instead of a combination of 2D and 3D kernels as for anisotropic data. For training augmentation, we enable transpose between every pair of three axes as the input is cubic. We use both weighted BCE and Dice losses with a ratio of 1:1 and train the model for 100K iterations with a batch size of 8. For post-processing, we applied median filtering with a kernel size of 7 × 7 × 7.

**Mitochondria Instance Segmentation**：We use the Tile Dataset to process MitoEM as the volumes are too large to be directly loaded
into memory. Different from the binary semantic segmentation models that have only one output channel, we build a U3D-BC architecture [51] that predicts the foreground mask and instance contour map at the same time. The instance contour map is informative in separating closely touching instances. We train the models for 150K iterations from scratch with a batch size of 8 and an input size of 257 × 257 × 17 as the data is also anisotropic. We train two models for two volumes separately in this comparison. The watershed segmentation algorithm is applied after merging overlapping chunks back to a single volume

**Neuronal Nuclei Instance Segmentation**：tricky part, a U3D-BCD model 

## Workflow
Data Loading -> Data Augmentation (supervised learning / unsupervised learning, ? reject sample) -> training (model choices) -> Infferences 
