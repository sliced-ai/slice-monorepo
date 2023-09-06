#!/bin/bash

# Variables
CLUSTER_NAME="slice-cluster"
REGION="us-west-2"
FRONTEND_NODE_TYPE="t3.medium"
FRONTEND_NODE_MIN=1
FRONTEND_NODE_MAX=3
FRONTEND_NODE_DESIRED=1
BACKEND_NODE_TYPE="t3.medium"
BACKEND_NODE_MIN=1
BACKEND_NODE_MAX=3
BACKEND_NODE_DESIRED=1
GPU_NODE_TYPE="g4dn.metal"
GPU_NODE_MIN=0
GPU_NODE_MAX=2
GPU_NODE_DESIRED=0

# Create EKS Cluster with a frontend node group
eksctl create cluster \
  --name $CLUSTER_NAME \
  --region $REGION \
  --nodes $FRONTEND_NODE_DESIRED \
  --nodes-min $FRONTEND_NODE_MIN \
  --nodes-max $FRONTEND_NODE_MAX \
  --node-type $FRONTEND_NODE_TYPE \
  --node-labels role=frontend \
  --managed

# Add a backend node group
eksctl create nodegroup \
  --cluster $CLUSTER_NAME \
  --region $REGION \
  --name backend-nodes \
  --node-type $BACKEND_NODE_TYPE \
  --nodes $BACKEND_NODE_DESIRED \
  --nodes-min $BACKEND_NODE_MIN \
  --nodes-max $BACKEND_NODE_MAX \
  --node-labels role=backend

# Add a GPU node group
eksctl create nodegroup \
  --cluster $CLUSTER_NAME \
  --region $REGION \
  --name gpu-nodes \
  --node-type $GPU_NODE_TYPE \
  --nodes $GPU_NODE_DESIRED \
  --nodes-min $GPU_NODE_MIN \
  --nodes-max $GPU_NODE_MAX \
  --node-labels role=gpu

# Configure kubectl
aws eks update-kubeconfig --name $CLUSTER_NAME --region $REGION

# Print the details of the created resources
echo "EKS Cluster: $CLUSTER_NAME"
echo "Region: $REGION"
echo "Frontend node group:"
echo "  Node type: $FRONTEND_NODE_TYPE"
echo "  Min nodes: $FRONTEND_NODE_MIN"
echo "  Max nodes: $FRONTEND_NODE_MAX"
echo "  Desired nodes: $FRONTEND_NODE_DESIRED"
echo "Backend node group:"
echo "  Node type: $BACKEND_NODE_TYPE"
echo "  Min nodes: $BACKEND_NODE_MIN"
echo "  Max nodes: $BACKEND_NODE_MAX"
echo "  Desired nodes: $BACKEND_NODE_DESIRED"
echo "GPU node group:"
echo "  Node type: $GPU_NODE_TYPE"
echo "  Min nodes: $GPU_NODE_MIN"
echo "  Max nodes: $GPU_NODE_MAX"
echo "  Desired nodes: $GPU_NODE_DESIRED"
