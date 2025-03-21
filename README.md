
# 📌 SmartForget: Embedding-Based Redundancy Removal for AI Memory Cleanup

## 🔍 Project Overview

As AI systems continuously accumulate user data to provide personalized and contextual support, managing the storage of this information becomes a major challenge. AI memory databases are expensive and finite, requiring intelligent strategies to periodically prune redundant or outdated data.

**SmartForget** is a lightweight memory cleanup system that uses vector embeddings and cosine similarity to identify and remove semantically redundant memory entries—mimicking the way humans forget unimportant details while preserving core information.

---

## 🧠 Problem Statement

- AI systems retain logs like:
  - “I had a headache today.”
  - “Birthday is Jan 24.”
- Not all memories are equally valuable long-term.
- Temporary or repetitive entries consume space without adding lasting value.
- We need a system that can **automatically detect and discard low-value memory entries** to optimize performance and storage.

---

## 📊 Dataset

- **Name:** [Text Similarity](https://www.kaggle.com/datasets/rishisankineni/text-similarity)
- **Source:** Kaggle, created by Rishi Sankineni
- **Description:**  
  Pairs of short descriptions (e.g., stock-related text) labeled as to whether they refer to the same entity.
- **Fields Used:**  
  - `description_x`, `description_y` – treated as individual memory entries
  - Other fields (e.g., `same_security`) were not used directly but helped frame the simulation context

---

## 🔄 Data Processing Pipeline

1. **Load & Clean Data**
   - Merged `train.csv` and `test.csv`
   - Removed NaNs and irrelevant columns

2. **Generate Embeddings**
   - Used `SentenceTransformer` (all-MiniLM-L6-v2) to encode all descriptions into vector embeddings
   - Each memory is now represented as a high-dimensional vector capturing semantic meaning

3. **Compute Similarity Matrix**
   - Used **cosine similarity** to compare memory entries
   - Built a full similarity matrix between all vectors

4. **Identify Redundant Entry**
   - Calculated total similarity score for each vector
   - Identified the entry with the highest cumulative similarity to others

5. **Memory Cleanup**
   - Removed the most redundant entry from the dataset

---

## 📈 Model Approaches

- **Deep Learning Approach:**
  - SentenceTransformer embeddings (SBERT)
  - Rich, contextual understanding of language

- **Non-Deep Learning Baseline:**
  - TF-IDF + Cosine Similarity
  - Faster, but less accurate at detecting semantic overlap

- **Evaluation Metric:**
  - **Cosine Similarity Score**
  - Measures closeness between vector pairs

- **Comparison to Naive Baseline:**
  - Naive deletion (random or timestamp-based) vs. embedding-based deletion
  - Our method retained more unique, meaningful content


---

## ✅ Results & Key Takeaways

- Embedding-based cleanup effectively reduces redundant entries
- Preserves critical, long-term information
- Scalable and easily integrated into real-world AI assistant architectures

---

## ⚖️ Ethical Considerations

- Deletion must be explainable and transparent
- Avoid removing sensitive or long-term useful information
- Ensure fairness in what gets retained vs. forgotten
- Human oversight is recommended for edge cases

---

## 🧑‍💻 Authors

- Violet Suh – Embedding strategy, similarity detection, and deletion logic
- Baze Bai – Modeling evaluation, interface development, and performance testing

---


