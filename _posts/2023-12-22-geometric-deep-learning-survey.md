---
title: geometric deep learning survey
date: 2023-12-22 13:18:00 -0700
categories: [Machine Learning]
tags: [note] # TAG names should always be lowercase
---

## 4. Blog on Graph Neural Networks (GNN)

- **Literature Review**:
  - Read papers in the model section in Notion.

- **PyG Skills**:
  - Learn PyG programming skills. [Blog](https://mlabonne.github.io/blog/posts/2022_02_20_Graph_Convolution_Network.html), [GitHub](https://github.com/mlabonne/graph-neural-network-course).
  - Implement GNN neural networks. Have implemented an introductory level GCN
  - try to implement GAN
  - PYG 算子分类 [什么是算子](https://zhuanlan.zhihu.com/p/533725319)
    - operator classification: [PYG cheatsheet](https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/gnn_cheatsheet.html)
  - [torch profiler](https://pytorch.org/docs/stable/profiler.html)

  ## the compiler of geometric deep learning about sparsity?
  ### main focus, probably
  - [SparseTir](https://arxiv.org/abs/2207.04606)
   - [video explanation](https://www.google.com/search?q=sparsetir+video&rlz=1C1ONGR_enUS1055US1055&oq=sparsetir+vi&gs_lcrp=EgZjaHJvbWUqCAgAEEUYJxg7MggIABBFGCcYOzIGCAEQRRg5qAIAsAIA&sourceid=chrome&ie=UTF-8#fpstate=ive&vld=cid:49517380,vid:dGeUOPh37gU,st:0)
  - learn [triton](https://triton-lang.org/main/index.html)

  task details:
  organize the workload of sparsetir: sddmm; spmm; gather+gemm+scatter

  ### the background of the geometric deep learning
  - [graph networks](https://arxiv.org/pdf/2310.11829.pdf)
  - [general intro to geometric deep learning](https://geometricdeeplearning.com/blogs/)

  #### SparseTir Workload
  - understand the structure of the [machine learning compilers](https://huyenchip.com/2021/09/07/a-friendly-introduction-to-machine-learning-compilers-and-optimizers.html)
  - workloads of sparsetir:
    1. sddmm: sampled dense-dense matrix multiplication sddmm for caculating attention score $$ B_{i,j} = \sum_{k=1}^{d} A_{i,j} X_{i,k} Y_{k,j} $$
    2. spmm: sparse-dense matrix multiplication, [visualExplanation]https://www.researchgate.net/figure/Conceptual-view-of-SpMM-and-SDDMM-sparse-matrix-the-values-may-change-but-the-sparsity_fig3_330891126) [mathematic explaination](https://www.google.com/imgres?imgurl=https%3A%2F%2Fars.els-cdn.com%2Fcontent%2Fimage%2F3-s2.0-B9780124201583000095-f09-14-9780124201583.jpg&tbnid=Ud5EYzvA8wLcfM&vet=10CAIQxiAoAGoXChMI-PDtx4qpgwMVAAAAAB0AAAAAEA8..i&imgrefurl=https%3A%2F%2Fwww.sciencedirect.com%2Ftopics%2Fcomputer-science%2Fsparse-matrix-vector-multiplication&docid=PHAZknEooJ31VM&w=433&h=390&itg=1&q=why%20we%20use%20sparse-dense%20matrix%20multiplication&ved=0CAIQxiAoAGoXChMI-PDtx4qpgwMVAAAAAB0AAAAAEA8#imgrc=diPN3FglDALu2M&imgdii=uoQnP7OopCqLOM). spmm works for the message passing
    mathematical formulas: $$Y_{i,k} = \sum_{j=1}^{n} A_{i,j} X_{j,k}$$
    3. gather + gemm + scatter: Relational Gather-Matmul-Scatter(RGMS). This method is to deal with 3 dimensions, The one more dimension comes from the number of relations. In the graph, there are multiple relation among each node. 
     - Relational Graph Convolution Network (RGCN): the essence of RGCN is 3D message passing mechanism.
     $$Y_{i,l} = \sum_{r=1}^{R} \sum_{j=1}^{n} \sum_{k=1}^{d_{in}} A_{r,i,j} X_{j,k} W_{r,k,l}$$
  
  #### SparseTir Workload Redo
  baseline classifications: 
  cuSPARSE - NVIDIA's offiical library for sparse tensor algebra

  dgSPARSE - SOTA sparse kernel implemenetation for GNNs; GE-SpMM, DA-SpMM, and PRedS

  a high-performing SpMM kernel on a GPU requires efficient memory access patterns and load balancing.

  PyG and DGL are two open-source frameworks. 

  [all relavant gnns](https://theaisummer.com/gnn-architectures/)

  ##### SPMM
  spmm: sparse-dense matrix multiplication, the most generic sparse operator in deep learning. 
  GE-SpMM, DA-SpMM in dgSPARSE are STOA kernel implementations.
  $$Y_{i,k} = \sum_{j=1}^{n} A_{i,j} X_{j,k}$$

  ![spmm](https://raw.githubusercontent.com/Yuxuan-Zhang-Dexter/Yuxuan-Zhang-Dexter.github.io/main/_imgs/spmm.png)
  k = n here. In graph, x multiplies each column in the feature matrix to get three n or k dimension vector and finally sum them together based on j to get the i th row in the output feature matrix.


  Typical Example: PyG - GCNConv() - message passing $$h_i = \sum_{j \in N_i} \frac{1}{\sqrt{\text{deg}(i) \text{deg}(j)}} W x_j$$ (not exactly)
  A is an adjacent matrix, represents the connections among nodes. 
  Y is the summation of message passing. 

  In the sparsetir, to load balancing, they reorganize sparse matrix. Based on the connection number of each node, the sparse matrix is separated into several dense matrix. (padding increases flops)
  ![ell](https://raw.githubusercontent.com/Yuxuan-Zhang-Dexter/Yuxuan-Zhang-Dexter.github.io/main/_imgs/ell.png)

  when hyp format increase the number of partittions, the memoery transations will increase and finally the benefitt of column partitioning saturates. Generally, the column partition is beneficial. 


  #### SDDMM 
  sddmm: sparse-dense matrix multiplication
  PRedS in dgSPARSE are STOA kernel implementation. 
  $$ B_{i,j} = \sum_{k=1}^{d} A_{i,j} X_{i,k} Y_{k,j} $$

  [summary of spmm and sddmm](https://www.researchgate.net/publication/330891126_Adaptive_sparse_tiling_for_sparse_matrix_multiplication)



  #### Triton code


