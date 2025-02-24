# Sentiment Analysis Project

## Description
This project implements **Sentiment Analysis** using **Convolutional Neural Networks (CNNs) and Google News Word2Vec embeddings**. The model classifies text into **positive or negative sentiments** based on the provided dataset.

The pipeline includes **text preprocessing, word embedding generation using pre-trained Google News Word2Vec, CNN-based feature extraction, and classification**. CNNs help in capturing local and hierarchical text representations, improving sentiment prediction accuracy.

---

## Features
- **Preprocessing**:
  - Tokenization, stopword removal, and text normalization.
  - Word sequence padding for uniform input length.

- **Word Embeddings**:
  - Uses **pre-trained Google News Word2Vec embeddings** to improve word representations.
  - Converts text into vectorized form for deep learning models.

- **Convolutional Neural Networks (CNNs)**:
  - **1D convolutional layers** extract relevant sentiment patterns.
  - **Global Max Pooling** for dimensionality reduction.
  - **Dense layers** for final sentiment classification.

- **Training & Evaluation**:
  - Uses **train-test split** for evaluation.
  - Implements `ModelCheckpoint` to save the best-performing model.

---

## Results

### Model Performance
| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| CNN + Word2Vec    | **93.5%** | **92.9%** | **93.7%** | **93.3%** |
| Naïve Bayes       | 85.2%    | 84.5%     | 85.0%  | 84.7%    |
| SVM               | 88.6%    | 88.0%     | 88.9%  | 88.4%    |

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Sentiment Distribution
![Sentiment Distribution](results/sentiment_distribution.png)

### Key Insights
- **CNN + Word2Vec achieved the highest accuracy of 93.5%**, outperforming traditional models.
- **Word embeddings significantly improved classification performance**, capturing contextual meaning better than TF-IDF or one-hot encoding.
- **CNN-based architectures effectively extracted local text patterns**, leading to better sentiment classification.

---

