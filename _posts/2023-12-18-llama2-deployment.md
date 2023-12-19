---
title: llama2 deployment
date: 2023-12-18 13:50:00 -0700
categories: [Machine Learning, Techniques]
tags: [llm] # TAG names should always be lowercase
---
## install llama2 locally
![installation instructions](https://github.com/facebookresearch/llama?fbclid=IwAR1YR2P7h62OBvtUIHcsf4OPzgu_QEeK3QQnuARrJhwYG5TTkRfKRf6COI0)

1. request weights download from the meta website.
Follow the instructions, I have deployed a local ones.

## convert the local one to the local docker image and container.

1. create a dockerfile to generate the image.

2. use CI/CD to build docker image from docker file(research about CI/CD pipeline in the gitlab)
[gitlab, docker, and kubernete instructions](https://ucsd-prp.gitlab.io/userdocs/development/gitlab/)


