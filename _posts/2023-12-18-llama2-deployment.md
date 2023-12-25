---
title: llama2 deployment
date: 2023-12-18 13:50:00 -0700
categories: [Machine Learning, Techniques]
tags: [llm] # TAG names should always be lowercase
---
## install llama2 locally
[installation instructions](https://github.com/facebookresearch/llama?fbclid=IwAR1YR2P7h62OBvtUIHcsf4OPzgu_QEeK3QQnuARrJhwYG5TTkRfKRf6COI0)

1. request weights download from the meta website.
Follow the instructions, I have deployed a local ones.

## convert the local one to the local docker image and container.

1. create a dockerfile to generate the image.

2. use CI/CD to build docker image from docker file(research about CI/CD pipeline in the gitlab)

[gitlab, docker, and kubernete instructions](https://ucsd-prp.gitlab.io/userdocs/development/gitlab/)

- test the container from the docker file
- upload the local docker image to the gitlab (size limit, I can't upload the docker image with weight? )
- continue the private image registry pulling and using kubernetes

## future optimization

workflow: 

local valid docker image and set up s3 storage to save all weights -> upload docker image to gitlab container registry -> pulling from private image registry and building container on kubernetes by passing secrets of s3. 

(why not directly using .gitlab-ci.yml to build and push docker image with all weights in the container registry?)

docker image: has not weights but s3 setting
s3: store all weights
docker container: use weights from s3 and docker image environment.

1. using s3 to store all weights
2. using docker secret to use or download weights from s3 when we build a container


## the update of llama2 deployment

1. use s3 to store weights
2. use pvc 


