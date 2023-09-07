---
title: Pytorch Connectomics Learning Process Record
date: 2023-09-06 09:20:00 -0700
categories: [Machine Learning, Image Classification]
tags: [Project]     # TAG names should always be lowercase
---
# Learning Plan
I aim to train models in Pytorch Connectomics software application using the brain images and build a reliable pipeline for future use in electronical microscopy images. Firstly, to approach the Pytorch Connectomics, I start to read the published paper: PyTorch Connectomics: A Scalable and Flexible Segmentation Framework for EM Connectomics (https://arxiv.org/abs/2112.05754). 

## Paper Summary
PyTorch Connectomics (PyTC) is an opensource deep-learning framework for the semantic and instance segmentation of volumetric microscopy images, built upon PyTorch. It includes the semantic and instance segementation in Neurons, Synapses, Mitochondria, and Nuclei, and Artifacts. 

### System Design
They focus on data scalability that can handle datasets of different scales, model flexibility for learning
multiple targets simultaneously with various loss functions.
The system is supported by GPU and CPU parallelism.
#### Data Scalability
*Data Loading*: 
