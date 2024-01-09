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
  ###### 1.GSPMM
  spmm: sparse-dense matrix multiplication, the most generic sparse operator in deep learning. 
  GE-SpMM and DA-SpMM in dgSPARSE are state-of-the-art (STOA) kernel implementations.
  $$Y_{i,k} = \sum_{j=1}^{n} A_{i,j} X_{j,k}$$

  ![spmm](https://raw.githubusercontent.com/Yuxuan-Zhang-Dexter/Yuxuan-Zhang-Dexter.github.io/main/_imgs/spmm.png)
  Here, \( k = n \). In a graph, \( x \) multiplies each column in the feature matrix to obtain three n or k dimensional vectors and finally sums them together based on \( j \) to get the \( i \)-th row in the output feature matrix.

  Applications: Locally Optimal Block Preconditioned Conjugate Gradient.
  Typical Example: PyG's `GCNConv()` for message passing is not exactly this but similar:
  $$h_i = \sum_{j \in N_i} \frac{1}{\sqrt{\text{deg}(i) \text{deg}(j)}} W x_j$$
  Here, \( A \) is an adjacency matrix, representing the connections among nodes. 
  \( Y \) is the summation of message passing.

  In sparsetir, for load balancing, they reorganize the sparse matrix. Based on the connection number of each node, the sparse matrix is separated into several dense matrices (padding increases FLOPs).
  ![ell](https://raw.githubusercontent.com/Yuxuan-Zhang-Dexter/Yuxuan-Zhang-Dexter.github.io/main/_imgs/ell.png)

  When the hyp format increases the number of partitions, the memory transactions will increase, and finally, the benefit of column partitioning saturates. Generally, column partitioning is beneficial.

  ###### 2.Multi-head SPMM
  Based on my understanding, after calculating the attention scores, we could perform a multi-head SPMM:
  $$h_i = \alpha_{i1} Wx_1 + \alpha_{i2} Wx_2 + \alpha_{i3} Wx_3 + \alpha_{i4} Wx_4$$

  ##### SDDMM
  ###### 1.GSDDMM
  sddmm: Sampled Dense-Dense Matrix Multiplication.
  PRedS in dgSPARSE are state-of-the-art (STOA) kernel implementations. (1. load/store intrinsics 2. intra-group and inter-group)
  Applications: gamma poisson, sparse factor analysis, and alternating least squares.
  $$B_{i,j} = \sum_{k=1}^{d} A_{i,j} X_{i,k} Y_{k,j}$$

  ###### 2.Multi-head SDDMM
  [SDDMM in multi-attention](https://docs.dgl.ai/en/1.1.x/notebooks/sparse/graph_transformer.html)

  ##### SPMM and SDDMM Visualization in the Conceptual View; sparsity in transformers
  ![visualization](https://raw.githubusercontent.com/Yuxuan-Zhang-Dexter/Yuxuan-Zhang-Dexter.github.io/main/_imgs/spmmAndSddmm.png)

  In SpMM, \( S \) is a sparse matrix. We only need to multiply \( O[i][:] = S[i][:] \) (here we only select non-zero elements on \( j \)) with \( D[j][:] \) and aggregate on \( j \).

  In SDDMM, we do \( D1[j][:] \) * \( D2[i][:] \) dot product to get one scalar and use this scalar to multiply \( S[i][j] = O[i][j] \).

  [Summary of SPMM and SDDMM](https://www.researchgate.net/publication/330891126_Adaptive_sparse_tiling_for_sparse_matrix_multiplication)

  Sparsity in Transformers comes from 1.sparse attention(multi-head spmm and sddmm) and 2.sparsity in the network weights after pruning. 

  ###### 1.multi-head spmm and sddmm in sparsetir(sparse attention)
  In sparsetir, longformer and pixelated butterfly transformer - sparse matrix: band matrix and butterfly matrix. (details, later explores)

  ###### 2.sparse weight(networking pruning)
  structured pruning:
    - [block pruning](https://aclanthology.org/2021.emnlp-main.829.pdf): the operator used in block-pruned transformer is SpMM
  
  unstructured pruning:



  ##### Relational Gather-Matmul-Scatter
  $$Y_{i,l} = \sum_{r=1}^{R} \sum_{j=1}^{n} \sum_{k=1}^{d_{in}} A_{r,i,j} X_{j,k} W_{r,k,l}$$

  A and W are 3D sparse matrix. R represents the number of relations. Under each relation, $A_{i,j}$ is a sparse matrix and $W_{k,l}$ is a dense matrix. X is a 2D feature matrix. 

  ###### 1.Relational Graph Convolution Network(RGCN)
  This is a generlalization of GCN mdoel to heterogeneous graphs with multiple relations/edge tyypes.

  Existing GNN libraries only use two stages:
  - $$ T_{r,j,l} = \sum_{k=1}^{d_{in}} X_{j,k} W_{r,k,l} $$ 
  The first stage fuses gathering and matrix multiplication
  - $$ Y_{i,l} = \sum_{r=1}^{R} \sum_{j=1}^{n} A_{r,i,j} T_{r,j,l} $$ 
  The second stage performs scattering. 

  In sparsetir, fuses two  stages into a single operator. 
  ![a single operator](https://raw.githubusercontent.com/Yuxuan-Zhang-Dexter/Yuxuan-Zhang-Dexter.github.io/main/_imgs/rgcn.png)

  Firstly, in the gathering and matmul, we do a similar hypo form to partition 2D sparse matrix and get ell forms to multiply with the corresponding weight matrix W. Repeat these steps r times( the number of relations). Finally, we aggregate together.

  generally perform better except increase some flops in the padding.


  ###### 2.Sparse Convolution (a special case of RGMS in 3D cloud point data)
  Assumption is that ELL(1)
  ![sparse convolution](https://raw.githubusercontent.com/Yuxuan-Zhang-Dexter/Yuxuan-Zhang-Dexter.github.io/main/_imgs/sparse_convolution.png)
  brief understanding: 
  Firstly, we have a sparse matrix. when move the sparse matrix with offset like coordinates. Then, we have a multiple relations here. 

  SparseTIR's RGMS cannot beat TorchSparse because the flops of matmul in sparseTIR is quadratic.(two matrix mulplication is cubic so there are R weight matrix. Totally, the matmul is quadratic)

  In the related work, there is a summary of current GNN systems and compilers.

  #### Triton code

  ### General Roadmap of the Survey of Sparse GNN
  - Step 1: model sparse input. 
    1. list all gnn function using sparse input and the corresponding mathematical formula from PYG or DGL
    2. bonus what's the sparse input matrix ?


  - Step 2: the current implementations of sparse operator like spmm, sddmm, gather-matl-scatter, scatter ...
    1. need to learn basic parallel computing like general matrix multiplication. 
    2. how the current implementation of spmm, sddmm ... in the engineering part. Using mathematical formula or visualization.

  - Step 3: sparse operator local design ...

  My current responsibility is to organize info in the step1 and step2. browse all relevant info and organize them in the structure. 

  #### Step 1
  models:  GCN, GraphSAGE, GIN, GAT, PNA, EdgeCNN
  
  PYG functions: GCNConv; SAGEConv; GINConv; GATConv / GATv2Conv; PNAConv; EdgeConv 

  Operators: spmm, sddmm, gather-matl-scatter, scatter

  | Models   | PYG Func   | Operators   |
  |------------|------------|------------|
  | GCN | GCNConv | SPMM |
  | GraphSAGE | SAGEConv | SPMM & scatter |
  | GIN | GINConv | gather-mul-scatter |
  | GAT | GATConv / GATv2Conv| SPMM & SDDMM |
  | PNA | PNAConv| SPMM & scatter |
  | EdgeCNN | EdgeConv | gather-mul-scatter | 

  Mathematical model formula corresponding to operators:

  **GCN**: 
  - Message Passing Stage: $$h_i = \sum_{j \in N_i} \frac{1}{\sqrt{\text{deg}(i) \text{deg}(j)}} W x_j$$ = SPMM
  
  **GraphSAGE**: 
  - [Neighbor Sampling](https://docs.dgl.ai/en/0.9.x/tutorials/large/L0_neighbor_sampling_overview.html) 
  - Aggregation: [scatter_add, scatter_mean, scatter_max](https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html) like ($$\text{mean}_{j \in \mathcal{N}_i} (h_j)$$) = SCATTER
  - linear transformation $$h_i' = W_1 h_i + W_2 \cdot \text{mean}_{j \in \mathcal{N}_i} (h_j)$$ = SPMM
  
  **GIN** :
  -  ![gin visualization](https://raw.githubusercontent.com/Yuxuan-Zhang-Dexter/Yuxuan-Zhang-Dexter.github.io/main/_imgs/gin.png) = gather-mul-scatter
  
  **GAT** :
  -  calculate node embedding  $$h_i = \alpha_{i1} Wx_1 + \alpha_{i2} Wx_2 + \alpha_{i3} Wx_3 + \alpha_{i4} Wx_4$$ = SPMM
  - [calculate attention scores](https://docs.dgl.ai/en/0.8.x/tutorials/models/1_gnn/9_gat.html) $$e_{ij}^{(l)} = \text{LeakyReLU}\left(\mathbf{a}^{(l)T} \left[ \mathbf{z}_i^{(l)} \| \mathbf{z}_j^{(l)} \right]\right)$$
$$\alpha_{ij}^{(l)} = \frac{\exp\left(e_{ij}^{(l)}\right)}{\sum_{k \in \mathcal{N}(i)} \exp\left(e_{ik}^{(l)}\right)}$$
$$h_i^{(l+1)} = \sigma\left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(l)} \mathbf{z}_j^{(l)} \right)$$
 = SDDMM (?)



  Engineering formula corresponding to operators:
  **SPMM** 
  **SDDMM**
  **SCATTER**

  

  #### CUDA CODE



