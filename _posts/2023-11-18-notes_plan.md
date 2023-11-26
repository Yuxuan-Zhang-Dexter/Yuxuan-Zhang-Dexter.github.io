---
title: Some Notes for the future update
date: 2023-09-13 17:00:00 -0700
categories: [Machine Learning, Techniques]
tags: [Note]     # TAG names should always be lowercase
---
Topics: 
1. update and reorganize the learning from pytorch connectomics:
 - retrain the pytorch connectomics model and try to inference the model, and then apply watershed function in the inferencce
 - why the inference hd5 file is just black and what volume [2, 10, 1000, 1000] means here?  for the second question, 2 is the number of channels. (debug code by code)
 - try the new 10000 checkpoint
2. create blog about the deployment of llm alpa opt175b in the docker:
 - learn how to add secrets to the container
 - debug the process of creating an docker image from the dockerfile
3. create blog about transfomer and batch effects
4. create blog about GNN or relavant topics
 - read papers in the model section in the notion
 - learn pyg programming sklls and implement gnn neural networks
5. create blog about CS229 and deep learning coursework
6. if possible, try to create blog to summarize the coursework that I have already taken like CSE 250A.
7. small idea or small research direction
 - memory capacity limit in the gpt long context inputs. answer: cache files and images are stored in the local browser