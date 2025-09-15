# GNN-CL for Few-Shot Satellite Image Classification

This repository contains the official PyTorch implementation for the paper: **"Enhancing Few-Shot Satellite Image Scene Classification with Graph Neural Network and Contrastive Learning"**.

Our work introduces a novel framework (FSL-GNN-CL) that integrates Graph Neural Networks (GNNs) and Contrastive Learning (CL)  to tackle the challenges of Few-Shot Learning (FSL) in Remote Sensing (RS) satellite imagery.

## The Challenge

Few-shot classification of satellite images is particularly difficult due to two main factors:

* **High Intra-Class Variance:** Images within the same class (e.g., "urban" or "residential") can appear vastly different across various geographic regions.
* **Low Inter-Class Discriminability:** Different classes (e.g., "farmland" and "wetlands") can share very similar spectral and textural features, making them hard to distinguish.

## Our Solution: FSL-GNN-CL

We frame the few-shot task as a graph-based relational reasoning problem. Instead of classifying images in isolation, our model constructs a task-specific graph for each *support set* and *query set*.

1.  **Feature Extraction:** A ResNet encoder extracts deep features from all images in the current task (both support and query).
2.  **Graph Construction:** These features become the **nodes** of a graph. **Edges** are constructed to represent relationships, primarily based on label equality for the known support set images.
3.  **Relational Reasoning:** A novel **GNN-Based UNet** propagates information across the graph. This allows query images (nodes) to be classified based on their relationship and similarity to the support images (nodes), effectively modeling the high intra-class variance.
4.  **Hybrid Loss Function:** The network is trained using a joint loss function:
    * **Cross-Entropy Loss:** Ensures correct classification of the query images.
    * **Contrastive Loss:** Enforces feature separability. It pulls embeddings of the same class closer together while pushing embeddings from different classes apart.

## Architecture

The overall architecture of our proposed framework is shown below:

![Model Architecture](assets/arch.png)

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/0aub/fsl-gnn.git
cd fsl-gnn
````

### 3\. Download Datasets

Our paper evaluates performance on four standard FSL-RS benchmarks:

  * **UCMerced (UCM)** 
  * **Aerial Image Dataset (AID)** 
  * **WHU-RS19** 
  * **NWPU-RESISC45 (NWPU45)** 

Please download the datasets and place them in a `data/` directory (or update the paths in the training script accordingly).

### 4\. Run Training

You can start training using the main script. The key arguments are `--dataset`, `--n_way` (W), and `--n_shot` (K), which correspond to the 5-way 1-shot and 5-way 5-shot tasks evaluated in the paper.

**Example: 5-way 1-shot training on UCMerced**

```bash
python train.py --dataset UCMerced --n_way 5 --n_shot 1
```

**Example: 5-way 5-shot training on NWPU-RESISC45**

```bash
python train.py --dataset NWPU-RESISC45 --n_way 5 --n_shot 5
```

## Results

Our model achieves state-of-the-art results by effectively combining relational graph reasoning with discriminative contrastive learning. We demonstrate significant accuracy improvements over existing methods on all four datasets.

  * **UCMerced (5-way 1-shot):** $96.90%\\pm0.00%$ accuracy , a **$29.63%$** improvement over the DEADN4 baseline.
  * **AID (5-way 1-shot):** $96.42%\\pm2.3%$ accuracy , a **$28.71%$** improvement over the TDNet baseline.
  * **NWPU-RESISC45 (5-way 5-shot):** $98.43%\\pm0.47%$ accuracy , a **$9.95%$** improvement over the $CS^{2}TFSL$ baseline.
  * **WHU-RS19 (5-way 5-shot):** $98.13%\\pm0.57%$ accuracy.

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{khelifi2024enhancing,
  title={Enhancing Few-Shot Satellite Image Scene Classification with Graph Neural Network and Contrastive Learning},
  author={Manel Khazri Khelifi and Ayyub Alzahem and Wadii Boulila and Anis Koubaa and Imed Riadh Farah},
  journal={},
  publisher={},
  year={2024},
  url={}
}
```
