# ⚽ Unsupervised Image Segmentation of Football Players using Clustering

## 📘 Overview
This project performs unsupervised **semantic segmentation** of football players in images using classic **clustering algorithms**. The goal is to segment players from the background without any pixel-level supervision. We experiment with multiple clustering strategies and features, generate binary masks, and evaluate against ground truth masks using standard metrics.

Dataset: [Football Player Segmentation on Kaggle](https://www.kaggle.com/datasets/ihelon/football-player-segmentation)

---

## 🗂️ Project Structure

- `images/`: Football images (1920×1080)
- `annotations/`: JSON masks with player locations
- `notebooks/`: Processing pipeline
- `outputs/`: Clustered segments, masks, evaluation metrics

---

## 🔁 Workflow Breakdown

### 1. Dataset Loading
- Sampled 50 images from the full dataset
- Downscaled to 1/8 of original size (240×135) to prevent memory issues

---

### 2. Feature Extraction
We extracted several types of features for pixel clustering:
- **RGB only**
- **RGB + Pixel coordinates (x, y)** ✅ Best performance
- **LAB color space**
- **Normalized RGB + Spatial Features**

Each feature was visualized and evaluated for clustering compatibility.

---

### 3. Clustering Algorithms
Applied 3 unsupervised clustering methods:
- **K-Means** (scikit-learn)
- **DBSCAN** (density-aware)
- **Agglomerative Clustering**

For K-means:
- Tuned **K** using inertia, silhouette score
- Optimal K: 4–6 depending on feature choice

Visualized segment maps and cluster boundaries per image.

---

### 4. Filtering & Merging
- Removed clusters with area below threshold (background/ground)
- Merged small adjacent clusters
- Final segments mostly contain player regions

---

### 5. Binary Mask & Re-clustering
- Created binary masks:
  - Player pixels → 1
  - Background → 0
- Reapplied clustering on binary mask
- Identified connected components
- Computed centroids of players
- Visualized masks + centroids

---

### 6. Advanced Feature Extraction
- Extracted **deep features** using `ResNet18` (intermediate layer)
- Downsampled images passed through CNN
- Applied **DBSCAN** on features
  - Noted players labeled as -1 (outliers)
  - Explained behavior based on DBSCAN’s density thresholding

---

### 7. Evaluation
Compared predicted binary masks with ground truth masks using:

| Metric         | Description                                        |
|----------------|----------------------------------------------------|
| **Dice Score** | 2×intersection / (pred + gt), higher is better     |
| **IoU**        | intersection / union, stricter than Dice           |

Reported average Dice and IoU scores across 50 sample images.

---

## ❓ Questions Answered

1. **Examples of Segmentation Types**:
   - Semantic: Tumor detection in MRI (single label)
   - Instance: Counting vehicles in traffic scene
   - Panoptic: Scene parsing in self-driving (combined)

2. **Dice vs IoU**:
   - Dice is more lenient, IoU penalizes mismatch more.
   - Dice preferred for small object segmentation (e.g. medical).

3. **Autoencoders for Pre-Clustering**:
   - Compress high-dimensional image → latent features
   - Apply K-Means on compressed features
   - Boosts clustering speed + quality

---

## 🛠️ Libraries Used
- `scikit-learn` (KMeans, DBSCAN)
- `opencv-python`, `matplotlib`, `numpy`, `json`
- `PyTorch`, `torchvision` (ResNet)
- `seaborn`, `PIL`

---

## 📌 Highlights
- No labels used for training — fully unsupervised pipeline
- Rich clustering + merging logic to extract players
- Evaluated segmentation against real masks
- Applied deep vision + traditional ML hybrid approach

---