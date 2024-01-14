---
title: Code Experiment
date: 2024-01-13 14:59:00 -0700
categories: [Machine Learning]
tags: [code]     # TAG names should always be lowercase
---

# Code Experiment and Analysis

## Pytorch

### DimNet on PyG
Based on [paper](https://arxiv.org/abs/2003.03123),

Our input is the molecular graph. 

we know three important manipulations in Directional Message Passing Neural Network.

Embedding block; Interaction block; output block.

We only care about the forward method in the interaction block, representing the message passing

In high-level understanding of message embeddings,

![aggregation](https://raw.githubusercontent.com/Yuxuan-Zhang-Dexter/Yuxuan-Zhang-Dexter.github.io/main/_imgs/dimnet.png)

### Embedding block

```
    def forward(self, x: Tensor, rbf: Tensor, i: Tensor, j: Tensor) -> Tensor: # embedding - > linear transform rbf in embedding size -> activation function ->
                                                                              # concate node, neighbors, and transformed rbf and linear transform them to embedding size -> activation function
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        return self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))
```

### Interaction block
```
 # assuming x_ji = num_node x hidden_channels representing connections between node and neighbors; x_kj = num_node x hidden_channels representing connections between neighbor and neighbors' neighbors;

    def forward(self, x: Tensor, rbf: Tensor, sbf: Tensor, idx_kj: Tensor,
                idx_ji: Tensor) -> Tensor:
        rbf = self.lin_rbf(rbf) ### num_radial to hidden_channels:  num_node x hidden_channels
        sbf = self.lin_sbf(sbf) ### num_spherical * num_radial to num_bilinear

        x_ji = self.act(self.lin_ji(x)) ### hidden to hidden - connections between ith node and all jth neighbors
        x_kj = self.act(self.lin_kj(x)) ### hidden to hidden - connections between jth neighbor and all kth neighbor's neighbors
        x_kj = x_kj * rbf ### element-wise multiplication each embedding
        ### batch multiplication, j = num_bilinear; l = hidden embedding ; wjl and ijl, element-wise multiplcation, sum over i j
        ### idx_kj should be indexes of nodes are neighbors
        x_kj = torch.einsum('wj,wl,ijl->wi', sbf, x_kj[idx_kj], self.W)   ### wi - num_node x hidden channels
        ### idx_ji represent connections between neighbor and neighbors' neighbor
        ### we aggregate all neighbors'neighbors' message into neighbors 
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')
        ### the new embedding = the neighor's embedding + the sum of the neighbors'neighbors
        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h)) + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h
```
In summary, 

    Dimnet is not spmm because when we aggregate every node with their neighbor information, they don't use matrix multiplication. They use scatter to aggregate jth neighbors' message into ith node, by using index idx_ji connections between ith node and jth neighbors. The complex calculation of sum of mkj is not using any sparse adjacent matrix in einsum(). idx_kj avoids using sparse matrix here. 

x_ij: num_node * hidden_embedding, mij - the ith node embedding in the previous layer.

x_kj: num_node * hidden_embedding, sum of mkj - the jth node special embedding

Core manipulations:

```
x_kj = torch.einsum('wj,wl,ijl->wi', sbf, x_kj[idx_kj], self.W)
```
confused about idx_kj ... index of jth neighbors' connections with neighbors'kth neighbors.?

w: idx_kj

l: hidden_channels

i: hidden_channels

j: num_bilinear

special manipulations to aggregate all mkj message.

wj * wl = wjl -> element-wise multiplication wjl * ijl -> sum over j and l, wi

```
x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0), reduce='sum')
```
the idx_ji represents connections between ith node and jth neighbors.

Aggregate all messages between ith node and all jth neighbors