---
title: Some Notes for the Future Update
date: 2023-11-18 17:00:00 -0700
categories: [Machine Learning, Techniques]
tags: [Note] # TAG names should always be lowercase
---

# Blog Topics Organization -- Update Plan

## 1. PyTorch Connectomics Learning Update

- **Retrain and Inference**:
  - Retrain the PyTorch Connectomics model.
  - Perform inference with the model and apply the watershed function.

- **Debugging Inference Output**:
  - Understand why the inference HDF5 file appears black.
  - Analyze the volume `[2, 10, 1000, 1000]`, where 2 represents the number of channels.
  - Debug code by code.

- **Experiment with New Checkpoints**:
  - rebuild docker image and container if I can't figure out why it doesn't work.
  - modify dockerfile to rebuild a better docker image - dpkg problematic package only
  - give up debugging stupid nvidia
  - continue to reuse docker container and check if it works by scping the connectomics
  - Check tensorboard to visualize how the loss gradient descent.

- **look up a potential new collaborator**
  - check marco's email

- **check new survey about electron microscopy images**
  - start with a new survey: https://doi.org/10.1016/j.media.2023.102920
  - only pytorch connectomics: installation - https://github.com/zudi-lin/pytorch_connectomics/blob/master/docker/README.md
  - debug installation environment error message:
   - ImportError: libGL.so.1: cannot open shared object file: No such file or directory
   - ImportError: libgthread-2.0.so.0: cannot open shared object file: No such file or directory
   - NCCL version uninstall - not compatiable with cuda 11.3


## 2. Blog on Deployment of LLM Alpha Opt175B in Docker

- **Secrets Management**:
  - Learn how to add secrets to the Docker container.

- **GitLab as Container Registry**:
  - Build the Docker image in GitLab.
  - Utilize GitLab as a container registry for Kubernetes deployment.
  - [Alpha project deployment](https://alpa.ai/tutorials/opt_serving.html).
  - how to create a new docker image from the running [docer container](https://www.dataset.com/blog/create-docker-image/).
  - share data between host machine and container - [docker mount volume](https://www.freecodecamp.org/news/docker-mount-volume-guide-how-to-mount-a-local-directory/)
  - **Error Message**
   - cupy11x - supporting cuda version
   - Jaxlib== only 0.4 not 0.3  - install python 3.9 - find a better docker image cudnn >= 8.05 and cuda >= 11.1 python 3.7 - 3.9
    - firstly use pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel docker image
    - check all cuda and cudnn version
    - install by hands about all necessary dependencies - stuck in GPU environment and try to create docker container in the labtop supporting nvidia gpu.
    - already set up the requirements - need to run weights of pretrained models in the alpa
    - update docker image
    - try to use a private docker registry
    - work with kubernetes to deploy the alpa installation
    - upload the weights of pretrained models to the deployed docker container
### Kubernetes Deployment with GitLab Registry

#### Step 1: Build the Docker Image

- Build the Docker image either **locally** using the Docker CLI or within **GitLab using their CI/CD pipelines**.

#### Step 2: Tag the Docker Image

- Tag the Docker image with the GitLab registry's path using the `docker tag` command.

#### Step 3: Push the Image to the GitLab Registry

- Authenticate with the GitLab container registry using `docker login`, then push the tagged image.

#### Step 4: Configure Kubernetes to Use the Private Image

- Reference the image from the GitLab registry in the Kubernetes Deployment manifest.
- Create a Kubernetes secret with GitLab registry credentials for private images.

#### Step 5: Create a Kubernetes Deployment

- Apply the Kubernetes Deployment manifest using `kubectl apply`.

#### Step 6: Verify the Deployment

- Verify the deployment status using `kubectl get pods`.

##### Key Considerations

- **CI/CD Pipelines**: Automate building, tagging, and pushing using GitLab CI/CD.
  - Trying to create .gitlab-ci.yml to automatically go through the first three steps in the GitLab pipeline. [.gitlab-ci.yml reference keywords](https://docs.gitlab.com/ee/ci/yaml/index.html).
  - Add secrets to the CI/CD pipeline: [CI/CD variables](https://docs.gitlab.com/ee/ci/variables/#:~:text=1,Select%20Update%20variable).
  - set up s3 storage [s3 in nautlius](https://ucsd-prp.gitlab.io/userdocs/storage/ceph-s3/).

- **Security**: Secure credentials using Kubernetes secrets.
- **Version Tagging**: Use specific version tags for Docker images.
- **Access Control**: Manage access to the GitLab registry using GitLab's permissions. [GitLab access control](https://ucsd-prp.gitlab.io/userdocs/storage/ceph-s3/).

### Backup Plan
- create a working docker image and deploy alpa in the local container. Updating the docker image and using this working docker image to deploy the alpa in the kubernetes
- check servers mentioned by the professor Jishen Zhao - wait for responses
 - These nodes are reserved for the prism center (understand you are already communicating regarding this). You can add a toleration in your spec that will allow the usage
 '' tolerations:
  - key: "prism-center"
    operator: "Exists"
    effect: "NoSchedule"
''

## 3. Blog on Transformer and Batch Effects

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

## 5. Blog on CS229 and Deep Learning Coursework

## 6. Blog on Summarizing Past Coursework (e.g., CSE 250A)

## 7. Small Research Ideas

- **Memory Capacity Limit in GPT**:
  - Explore the memory capacity limit in GPT for long context inputs.
  - Discuss caching files and images in the local browser.

## 8. Open Source Projects Plans
  - practice programming skills in the engineering aspects 


