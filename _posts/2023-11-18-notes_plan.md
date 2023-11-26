---
title: Some Notes for the future update
date: 2023-09-13 17:00:00 -0700
categories: [Machine Learning, Techniques]
tags: [Note]     # TAG names should always be lowercase
---
# Blog Topics Organization -- update plan

## 1. PyTorch Connectomics Learning Update
- **Retrain and Inference**:
  - Retrain the PyTorch Connectomics model.
  - Perform inference with the model and apply the watershed function.
- **Debugging Inference Output**:
  - Understand why the inference hdf5 file appears black.
  - Analyze the volume `[2, 10, 1000, 1000]`, where 2 represents the number of channels.
  - Debug code by code.
- **Experiment with New Checkpoints**:
  - Test with the new 10000 checkpoint.

## 2. Blog on Deployment of LLM Alpa Opt175B in Docker
- **Secrets Management**:
  - Learn how to add secrets to the Docker container.
- **GitLab as Container Registry**:
  - Build the Docker image in GitLab.
  - Utilize GitLab as a container registry for Kubernetes deployment.

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
 - trying to create .gitlab-ci.yml to automatically go through first three steps in the gitlab pipeline.
- **Security**: Secure credentials using Kubernetes secrets.
- **Version Tagging**: Use specific version tags for Docker images.
- **Access Control**: Manage access to the GitLab registry using GitLab's permissions.

## 3. Blog on Transformer and Batch Effects

## 4. Blog on Graph Neural Networks (GNN)
- **Literature Review**:
  - Read papers in the model section in Notion.
- **PyG Skills**:
  - Learn PyG programming skills.
  - Implement GNN neural networks.

## 5. Blog on CS229 and Deep Learning Coursework

## 6. Blog on Summarizing Past Coursework (e.g., CSE 250A)

## 7. Small Research Ideas
- **Memory Capacity Limit in GPT**:
  - Explore the memory capacity limit in GPT for long context inputs.
  - Discuss caching files and images in the local browser.
