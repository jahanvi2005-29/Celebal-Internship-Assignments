# 🛍️ Visual Product Recommendation System

> A deep learning–based visual search engine that recommends fashion products based on how they *look* — not on text, tags, or metadata.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📖 Overview

Traditional recommendation systems rely on text metadata (tags, categories, descriptions) to find similar products. This project takes a **purely visual** approach — given a query product image, the system retrieves the most visually similar items from the catalog using deep image embeddings and cosine similarity.

Three progressively stronger approaches are implemented and benchmarked against each other:

| Approach | Description |
|---|---|
| 🔹 **Baseline** | Pretrained ResNet50 (ImageNet weights), no fine-tuning |
| 🔹 **Transfer Learning** | ResNet50 fine-tuned on product category labels |
| 🔹 **Siamese Network** | Custom embedding network trained with **triplet loss** for metric learning |

---

## 🧠 How It Works
1. Images are preprocessed and passed through a CNN backbone (ResNet50)
2. Each image is converted into a compact embedding vector
3. Cosine similarity is computed between the query embedding and all catalog embeddings
4. The top-K most similar products are retrieved and displayed

The Siamese network takes this further by *learning* an embedding space specifically optimized for visual similarity — trained on anchor/positive/negative triplets so that visually similar items are pulled closer together and dissimilar items are pushed apart.

---

## 📂 Dataset

[Fashion Product Images Dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset) (Kaggle)

A subset of 5 categories was used: **Shirts, Shoes, Dresses, T-shirts, Watches** (up to 300 images per category).

---

## 📊 Evaluation

Models were evaluated using:
- **Precision@K** and **Recall@K** (K = 5, 20, 50)
- **t-SNE** visualization of the learned embedding space
- **Inference latency** benchmarking (single-query embedding + retrieval time)

### Results Snapshot

| Model | Precision@5 | Recall@5 |
|---|---|---|
| Baseline (pretrained) | *see notebook* | *see notebook* |
| Transfer Learning | *see notebook* | *see notebook* |
| **Siamese Network** | *see notebook* | *see notebook* |

*(Exact scores and charts are available in the notebook output — the Siamese network consistently outperforms the untrained baseline on category-level retrieval accuracy.)*

---

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `scikit-learn` · `pandas` · `NumPy` · `matplotlib` · `Kaggle API`

---

## ▶️ How to Run

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib kaggle
```

1. Get your Kaggle API credentials from **kaggle.com → Account → API**
2. Add them in the first code cell of `CelebalFinal.ipynb`
3. Run the notebook top to bottom

---
---

## 🚀 Key Highlights

- Compared 3 distinct embedding strategies on the same dataset for a fair, apples-to-apples benchmark
- Implemented a custom triplet loss function from scratch for metric learning
- Visualized embedding space quality using t-SNE (before vs. after training)
- Benchmarked real-time inference latency for production-readiness

---

## 👤 Author

**Jahanvi Gupta**
Data Science Intern @ Celebal Technologies
