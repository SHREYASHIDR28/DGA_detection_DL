

# Deep Learning-Based DGA Detection & Malware Family Attribution

## Detecting Malware Command & Control Domains Using Character-Level Deep Learning

---

## Problem Overview

Modern malware frequently uses **Domain Generation Algorithms (DGAs)** to dynamically generate thousands of pseudo-random domain names for Command & Control (C2) communication.

Traditional blacklist-based detection fails because:

* Domains change frequently
* Patterns are algorithmically generated
* High domain entropy evades simple filters

This project builds and evaluates a deep learning-based detection system capable of:

* Detecting DGA domains (Binary Classification)
* Attributing domains to specific malware families (Multi-Class Classification)
* Comparing CNN vs BiLSTM architectures
* Evaluating adversarial robustness

---

## Dataset

Balanced dataset containing:

* **80,000 legitimate domains**
* **80,000 DGA domains**
* **7 malware families**

### Malware Families Included

| Family       | Samples |
| ------------ | ------- |
| cryptolocker | 37,254  |
| newgoz       | 9,276   |
| gameoverdga  | 8,461   |
| nivdort      | 8,456   |
| necurs       | 8,331   |
| goz          | 6,136   |
| bamital      | 2,086   |

### Preprocessing Steps

* Lowercasing
* TLD removal
* Duplicate removal
* Class balancing
* Character-level tokenization
* Padding to fixed sequence length

---

## Exploratory Statistical Analysis

Before modeling, domain characteristics were analyzed:

| Feature     | Legit | DGA   |
| ----------- | ----- | ----- |
| Avg Length  | 10.61 | 17.21 |
| Entropy     | 2.84  | 3.46  |
| Vowel Ratio | 0.338 | 0.198 |
| Digit Ratio | 0.041 | 0.089 |

### Key Insight

DGA domains are:

* Longer
* More random (higher entropy)
* Digit-heavy
* Lower vowel density

This validates character-level modeling for detection.

---

## Model Architectures

### Binary Classification (Legit vs DGA)

**Architecture:**

* Character Embedding Layer
* Conv1D layers
* Global Max Pooling
* Dense + Dropout
* Sigmoid Output

**Performance:**

* Accuracy: **97%**
* ROC-AUC: **0.995**
* False Negative Rate: ~2.7%

This model reliably distinguishes algorithmically generated domains from legitimate ones.

---

### Multi-Class Malware Family Attribution

Classifies domains into:

* Legit
* 7 DGA families

**CNN Results:**

* Accuracy: **86%**
* Macro F1: **0.74**

**Important Finding:**
Class weighting significantly improved minority family recall:

* necurs recall improved from **1% → 48%**

---

### CNN vs BiLSTM Comparison

| Metric        | CNN      | BiLSTM   |
| ------------- | -------- | -------- |
| Accuracy      | **86%**  | 82%      |
| Macro F1      | **0.74** | 0.73     |
| Legit Recall  | **0.96** | 0.81     |
| necurs Recall | 0.48     | **0.64** |

### Interpretation

* CNN captures local character n-gram patterns effectively.
* BiLSTM improves minority recall for specific families.
* CNN demonstrates superior overall stability and generalization.

Conclusion:
For DGA detection, convolutional modeling is more robust than sequential modeling.

---

## Adversarial Robustness Evaluation

Tested modified domains such as:

* g00gle
* faceb00k
* micr0soft
* secure-google-login

### Result

The model classified them as legitimate.

### Interpretation

The system detects **algorithmic randomness**, not semantic impersonation.

Important distinction:

> DGA detection ≠ phishing detection.

This project specifically targets algorithmically generated domains used in botnet infrastructure.

---

## Project Structure

```text
notebooks/
    00_project_overview.ipynb
    01_data_preprocessing.ipynb
    02_binary_cnn.ipynb
    03_multiclass_cnn.ipynb
    04_bilstm_comparison.ipynb

models/
results/
data/processed/
```

The project is modular and reproducible.

---

## Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Matplotlib

---

## How to Reproduce

```bash
git clone https://github.com/yourusername/DGA-Detection-DeepLearning.git
cd DGA-Detection-DeepLearning
pip install -r requirements.txt
```

Then execute notebooks in order:

1. Data preprocessing
2. Binary CNN
3. Multi-class CNN
4. BiLSTM comparison

---

## Key Takeaways

* Character-level CNN achieves near-perfect DGA detection.
* Multi-family attribution is significantly more complex.
* Class imbalance strongly impacts minority performance.
* Sequential modeling provides limited advantage over convolutional modeling.
* Model effectively detects algorithmic randomness but not semantic impersonation.

---

## Future Enhancements

* Transformer-based character models
* Real-time DNS stream integration
* Phishing similarity detection
* Graph-based C2 infrastructure modeling

---



