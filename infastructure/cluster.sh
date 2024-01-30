#!/bin/bash


#ENSURE YOU ARE IN THE RIGHT VPC 
# Define Variables
CLUSTER_NAME="slice-ai"
REGION="us-west-2"
CPU_NODEGROUP_NAME="cpu-nodegroup"
GPU_NODEGROUP_NAME="gpu-nodegroup"
SSH_KEY="gputesting"

# Create EKS Cluster
eksctl create cluster \
  --name $CLUSTER_NAME \
  --region $REGION \
  --version 1.28 \
  --vpc-private-subnets=subnet-0e52b6f553295c587,subnet-06bfeb9ba93f3ce95,subnet-03b6480db73ed47ec \
  --without-nodegroup


# Create CPU Node Group
eksctl create nodegroup \
  --cluster $CLUSTER_NAME \
  --region $REGION \
  --name $CPU_NODEGROUP_NAME \
  --node-type m5.2xlarge \
  --nodes 1 \
  --nodes-min 0 \
  --nodes-max 4 \
  --node-private-networking \
  --subnet-ids subnet-0e52b6f553295c587,subnet-06bfeb9ba93f3ce95,subnet-03b6480db73ed47ec,subnet-0a70a21fdf39752d5


# Create GPU Node Group
eksctl create nodegroup \
  --cluster $CLUSTER_NAME \
  --region $REGION \
  --name $GPU_NODEGROUP_NAME \
  --node-type g5.xlarge \
  --nodes 1 \
  --nodes-min 0 \
  --nodes-max 4 \
  --ssh-access \
  --ssh-public-key $SSH_KEY

# Deploy CPU Example Application
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cpu-example
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cpu-example
  template:
    metadata:
      labels:
        app: cpu-example
    spec:
      containers:
      - name: cpu-container
        image: nginx
EOF

# Deploy GPU Example Application
cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-example
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gpu-example
  template:
    metadata:
      labels:
        app: gpu-example
    spec:
      containers:
      - name: gpu-container
        image: nvidia/cuda:latest
        resources:
          limits:
            nvidia.com/gpu: 1
EOF

# Delete the deployments post-execution (optional)
# kubectl delete deployment cpu-example
# kubectl delete deployment gpu-example
