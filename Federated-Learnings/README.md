# Federated Learning

Federated Learning is a distributed machine learning approach where multiple clients train models locally on their private data without sharing raw data with a central server. Instead of centralizing data, each client trains on their local dataset and sends only model weight updates to a central server, which aggregates these updates to create an improved global model.

## Notebooks Overview

### 1. Federated Learning Fundamentals (`1_Federated_Learning.ipynb`)

- **Non-IID Data Problem**: Each client may have different data distributions (e.g., in MNIST, client 1 excludes digits 1,3,7; client 2 excludes 2,5,8; client 3 excludes 4,6,9)
- **Model Architecture**: Simple fully connected neural network (784→128→10) for MNIST classification
- **Key Observation**: Individual models trained on non-IID data perform poorly on digits they never saw (0% accuracy on excluded digits)
- **Solution**: FedAvg algorithm aggregates model weights from multiple clients

### 2. Federated Training Process (`2_Federated_Training_Process.ipynb`)

Using the **Flower** framework for FL:

- **FedAvg Strategy**: Aggregates client model updates by computing weighted average based on number of training samples
- **Client-Server Communication**:
  1. Server sends global model to all clients
  2. Each client trains locally on their data partition
  3. Clients send updated weights back to server
  4. Server aggregates weights using FedAvg
- **Results**: Global model accuracy improved from ~12% (initial) to ~96% after 3 rounds
- Different subsets ([1,3,7], [2,5,8], [4,6,9]) all achieved ~95% accuracy

### 3. Hyperparameter Tuning (`3_Federated_Learning_Tuning.ipynb`)

- **Dynamic Configuration**: Server can send different training parameters per round
- **fit_config()**: Function that returns configuration dict (e.g., `local_epochs` varying by round)
- **Example**: 2 epochs for rounds 1-2, 5 epochs for round 3+
- This allows adaptive training strategies during federated learning

### 4. Data Privacy - Differential Privacy (`4_Federated_Learning_Data_Privacy.ipynb`)

- **Differential Privacy (DP)**: Adds noise to model updates to prevent reconstructing individual data from gradients
- **Key Components**:
  - **Adaptive Clipping**: Bounds client contribution sensitivity
  - **Noise Multiplier**: Controls privacy-utility tradeoff (set to 0.3)
  - **Client-side DP**: Privacy applied before sending to server
- **Implementation**: `DifferentialPrivacyClientSideAdaptiveClipping` wraps FedAvg strategy

### 5. Bandwidth Optimization (`5_Federated_Learning_Bandwidth.ipynb`)

- **Why Bandwidth Matters**:
  - Large models require significant transmission bandwidth
  - Impacts latency, scalability, and client heterogeneity

- **Bandwidth Tracking**: Measure data transferred between server and clients
- **Model Size Example**: Pythia-14m model is ~26 MB
- **Total Bandwidth**: For 2 clients in 1 round: ~104 MB (26 MB sent × 2 + 26 MB received × 2)

- **Optimization Techniques** (mentioned but not implemented):
  - Model compression (quantization, pruning)
  - Gradient sparsification
  - Sketching & low-rank updates

## Key Concepts Summary

| Concept | Description |
|---------|-------------|
| **FedAvg** | Federated Averaging - aggregates model weights from clients |
| **Non-IID Data** | Different clients have different data distributions |
| **Flower** | Framework for building FL systems |
| **Differential Privacy** | Adds noise to protect individual data |
| **Bandwidth** | Critical bottleneck - model size × clients × rounds |

## Requirements

- Python 3.10 (required for Flower framework compatibility)
- PyTorch
- Flower framework (`flwr`)
- Flower Datasets (`flwr-datasets`)

## References

- [Flower framework for federated learnings](https://flower.ai/docs/framework/)
