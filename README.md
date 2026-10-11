# Naive Bayes Classifier: Play Golf Dataset

An implementation and walkthrough of Naive Bayes classification based on the GeeksforGeeks tutorial:
[Naive Bayes Classifiers](https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/).

---

## Overview

This repository demonstrates how Naive Bayes classification works from first principles using the classic weather/golf dataset (`golf.csv`). It covers theory, probability tables, step-by-step manual calculation, custom implementation from scratch, and comparison with scikit-learn.

### Key Concepts Covered

1. **Bayes' Theorem & The Naive Assumption:**
   $$P(y \mid X) = \frac{P(X \mid y)\,P(y)}{P(X)}$$
   The conditional independence assumption decomposes the joint likelihood into a simple product of individual feature likelihoods.

2. **Categorical Naive Bayes:**
   * Prior calculation: $P(\text{Yes}) = 9/14$, $P(\text{No}) = 5/14$.
   * Frequency and conditional probability tables for Outlook, Temperature, Humidity, and Wind.
   * Step-by-step manual verification of the tutorial query: `(Outlook: Sunny, Temp: Hot, Humidity: Normal, Wind: Weak)`.
   * Prediction output: **Yes** ($P \approx 0.673$).

3. **Zero-Frequency Problem & Laplace Smoothing:**
   * Demonstrates how unseen combinations (such as `Outlook: Overcast` with `Play: No`) produce zero probabilities.
   * Compares unsmoothed results ($\alpha = 0$) with Laplace smoothing ($\alpha = 1$).

4. **Validation against Scikit-Learn:**
   * Direct verification showing that our custom `NaiveBayesClassifier` produces identical posterior probabilities to `sklearn.naive_bayes.CategoricalNB`.

5. **Model Evaluation:**
   * Training accuracy and classification report.
   * Leave-One-Out Cross-Validation (LOOCV) analysis highlighting sample size effects.

6. **Bonus: Gaussian Naive Bayes for Continuous Data:**
   * Full implementation of Gaussian Naive Bayes using class means and variances, verified on the Iris dataset alongside `sklearn.naive_bayes.GaussianNB`.

---

## File Structure

* `golf.csv`: The 14-row weather dataset with single-quoted categorical values.
* `naive_bayes_golf.ipynb`: Jupyter notebook containing the complete walkthrough, code, and pre-rendered plots.
* `requirements.txt`: Python package dependencies.

---

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the Notebook / Or view in Google Colab/VSCode

```bash
jupyter notebook naive_bayes_golf.ipynb
```
