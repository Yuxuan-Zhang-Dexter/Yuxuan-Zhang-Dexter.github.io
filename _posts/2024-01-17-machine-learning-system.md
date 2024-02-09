---
title: Research Exploration in Machine Learning System.
date: 2024-01-17 11:31:00 -0700
categories: [Machine Learning]
tags: [llm]     # TAG names should always be lowercase
---


# General Approaches to ML System Research Area
1. cultivate the critical thinking to evaluate the current tech trending.

## Open Questions
1. How can we parallel llm token outputs? (output two token in one pass)
2. How can we schedule the workload of llm (like when the text generate stops)
3. Why does Amazon cloud services dominate in the market than google cloud service? (google invents many important cloud techniques)
4. Why does tesla start to build its own super computer like google using TPU? (cloud cluster is cheaper than super computer but Nvidia hardware to Amazon cloud services to big tech software, Nvidia monopoly profits so high)
5. since reduce_scatter and all-gather are important manipulations in gpu, ring algorithm dominate in reducing the effects of bandwidth. Is there are any better way to improve the bandwidth speed? (Full Shared Data Parallel - FSDP)
6. in-memory database. (ram cost goes down) Although the performance is not better,  1)data serialization is eliminated(difference data representations); 2)simpler implementations
7. data warehouse olap big markets databricks and snowflakes

## the ml system important surveys

- Above the Clouds: A Berkeley View of Cloud
Computing

- [Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems](https://arxiv.org/abs/2312.15234)

### General Analysis of GPT 
1. In the mathematics and engineering aspects, I need to understand how llm could improve the performance from compiler to parallel computing ...
[KV-Caching abusing use in llm](https://medium.com/@joaolages/kv-caching-explained-276520203249)

**essential mechanism**

### intro to paper of speculative decoding (Q2)
[speculative decoding](https://arxiv.org/abs/2308.04623)

new one [medusa blog](https://www.together.ai/blog/medusa)

how RAG works here?


