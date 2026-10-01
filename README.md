# Feature Scaling, Normalization and Standardization in Machine Learning

This repository contains my practical work on **Feature Scaling, Normalization, and Standardization** in Machine Learning.

Feature scaling is an important data preprocessing step when numerical features have very different ranges. For example, one feature might contain values between 0 and 100, while another feature might contain values between 10,000 and 1,000,000.

Without appropriate scaling, some Machine Learning algorithms can be influenced more strongly by features with larger numerical ranges.

## What is Feature Scaling?

Feature scaling is the process of transforming numerical features so that their values are placed on a comparable scale.

For example:

```text
Age       → 18 to 80
Income    → 20,000 to 500,000
```

The original ranges are very different. Scaling can transform these features into more comparable numerical ranges.

---

# 1. Normalization

Normalization commonly refers to scaling values to a fixed range, often **0 to 1**.

One common method is **Min-Max Scaling**.

The formula is:

```text
X_scaled = (X - X_min) / (X_max - X_min)
```

After Min-Max Scaling, the minimum value becomes 0 and the maximum value becomes 1.

### Scikit-learn Example

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Min-Max Scaling can be useful when a model benefits from features being within a bounded range.

---

# 2. Standardization

Standardization transforms a feature so that it is centered around a mean of approximately 0 with a standard deviation of approximately 1.

The formula is:

```text
Z = (X - μ) / σ
```

Where:

- `X` = original value
- `μ` = mean
- `σ` = standard deviation

### Scikit-learn Example

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Standardization is commonly used with algorithms that are sensitive to feature scale.

---

# 3. Robust Scaling

I also practiced **Robust Scaling**, which uses the median and interquartile range (IQR).

It can be useful when numerical features contain significant outliers.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Robust Scaling is less affected by extreme values compared with methods based on the mean and standard deviation.

---

# Normalization vs Standardization

| Method | Main Idea | Typical Range |
|---|---|---|
| Min-Max Scaling | Rescales values using minimum and maximum | 0 to 1 |
| Standardization | Centers and scales using mean and standard deviation | No fixed range |
| Robust Scaling | Uses median and IQR | No fixed range |

---

# Why Feature Scaling is Important

Feature scaling can be important for algorithms that use distances, gradients, or regularization.

Examples include:

- K-Nearest Neighbors
- K-Means
- Support Vector Machines
- Logistic Regression
- Linear Regression with regularization
- Neural Networks
- Principal Component Analysis

Tree-based models such as Decision Trees and Random Forests generally do not require feature scaling in the same way because their splits are based on feature thresholds rather than distances or feature magnitude.

---

# Important Practice: Avoid Data Leakage

The scaler should be **fitted only on the training data**.

Correct approach:

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Do not fit the scaler separately on the test data.

The test set should be transformed using the scaling parameters learned from the training set.

---

# Feature Scaling Workflow

```text
Raw Dataset
     ↓
Select Numerical Features
     ↓
Split Data into Train and Test
     ↓
Choose Scaling Technique
     ↓
Fit Scaler on Training Data
     ↓
Transform Training Data
     ↓
Transform Test Data
     ↓
Train ML Model
     ↓
Evaluate Model
```

---

# Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Jupyter Notebook

# Learning Goal

The goal of this repository is to build a practical understanding of **Feature Scaling, Normalization, and Standardization** and learn how different scaling techniques can be applied to numerical features before training Machine Learning models.

Through this practice, I learned how Min-Max Scaling, Standardization, and Robust Scaling work, when they are useful, and how to correctly apply them without causing data leakage.

This repository is part of my ongoing **Machine Learning and AI learning journey**.
