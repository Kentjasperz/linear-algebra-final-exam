# -*- coding: utf-8 -*-
"""for_Part_D.ipynb

# Part D — Linear Transformations

**Final Project: The Linear Algebra Behind AI — A Regression Problem**

Preprocessing steps like standardization aren't just "cleaning" — they are linear maps with their own matrix representation.

---
**Student name:** _______________________   **Date:** _______________________

---

**What to do:** Define at least one linear transformation (e.g., standardization, normalization, or a projection) and apply it to your data.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

npz_path = '/content/drive/MyDrive/partC_data.npz' if os.path.exists('/content/drive/MyDrive/partC_data.npz') else 'partC_data.npz'
data = np.load(npz_path, allow_pickle=True)
X, y = data['X'], data['y']
feature_names = data['feature_names'].tolist()
rank_X = int(data['rank_X'])
print('Loaded X:', X.shape, ' y:', y.shape, ' rank_X:', rank_X)

# Build the matrix representation of your chosen linear transformation.
# Standardization T(x) = S(x - mean), whose LINEAR part is
# multiplication by S = diag(1/std):
mu = X.mean(axis=0)
sigma = X.std(axis=0)
S = np.diag(1.0 / sigma)   # diagonal matrix S = diag(1/sigma)

X_centered = X - mu
X_transformed = X_centered @ S   # apply T(x) = S x  (i.e. X_centered @ S)
print("Feature means (mu):", mu)
print("Feature stds (sigma):", sigma)
print("Transformation matrix S = diag(1/sigma):\n", S)
print("X_transformed shape:", X_transformed.shape)

# Show the transformation applied to ONE real data point
idx = 0   # first student in the dataset
x_before = X[idx]
x_centered = X_centered[idx]
x_after = X_transformed[idx]
print(f"Sample index {idx} ({feature_names}):")
print("Before (raw x):             ", x_before)
print("Centered (x - mu):          ", x_centered)
print("After T(x - mu) = S(x - mu):", x_after)
print("Check S @ (x - mu):         ", S @ x_centered)

# Split your (transformed) data into train and test sets (80/20 split).
rng = np.random.default_rng(42)
n = X_transformed.shape[0]
idx_perm = rng.permutation(n)
train_size = int(0.8 * n)
train_idx, test_idx = idx_perm[:train_size], idx_perm[train_size:]
X_train, X_test = X_transformed[train_idx], X_transformed[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

"""### Answer these in your report

**Q13.** Which linear transformation did you apply to your data? Write its matrix representation M such that T(x) = Mx.

> *Your answer:* After centering each feature by subtracting the mean vector μ = [79.9774, 19.9753, 75.0705, 7.0291]ᵀ so that x_centered = x − μ, I applied feature scaling (standardization) via the linear transformation T(x_centered) = M x_centered where M = S ∈ ℝ⁴ˣ⁴ is the diagonal scaling matrix of reciprocal standard deviations (with σ = [11.5466, 5.9901, 14.3987, 1.4680]ᵀ):
M = S = diag(1/σ₁, 1/σ₂, 1/σ₃, 1/σ₄) = diag(0.0866, 0.1669, 0.0695, 0.6812).
For the whole centered matrix, X_transformed = (X − μ)S. Multiplying by S rescales each feature axis independently into standard-deviation units (z-scores), putting all four features on a common dimensionless scale with unit variance.

**Q14.** Using one actual data point from your dataset, show the before vector x and the after vector T(x) = Mx.

> *Your answer:* Using the first student (`idx = 0`), the raw feature vector (`[Attendance, Hours_Studied, Previous_Scores, Sleep_Hours]`) is x_before = [84.0000, 23.0000, 73.0000, 7.0000]ᵀ. Subtracting the mean vector μ = [79.9774, 19.9753, 75.0705, 7.0291]ᵀ gives the centered vector x_centered = [4.0226, 3.0247, -2.0705, -0.0291]ᵀ, and multiplying by M = S = diag(0.0866, 0.1669, 0.0695, 0.6812) gives the transformed vector T(x_centered) = S(x − μ) = [0.3484, 0.5049, -0.1438, -0.0198]ᵀ. Each coordinate of the transformed vector represents how many standard deviations above or below the dataset average that student is on each predictor.

**Q15.** Is your transformation invertible? How do you know, and why would invertibility matter if you wanted to convert a prediction back to original units?

> *Your answer:* Yes, the transformation is invertible. Because S = diag(0.0866, 0.1669, 0.0695, 0.6812) is a diagonal matrix with strictly positive diagonal entries (since every feature has σⱼ > 0), its determinant is det(S) = 1/(σ₁σ₂σ₃σ₄) = 0.0007 ≠ 0 and its rank is 4. Its inverse is simply S⁻¹ = diag(σ₁, σ₂, σ₃, σ₄) = diag(11.5466, 5.9901, 14.3987, 1.4680). Invertibility matters because no information is lost: we can recover any student's original feature values via x = S⁻¹ x_transformed + μ, and we can convert regression coefficients β̂_std learned in standardized space back into the original physical units (exam points per percent of attendance, per study hour, etc.) via β̂_orig = S β̂_std.

**Q16.** Would your transformation's matrix change if computed from a different subset of your data (e.g., a different train/test split)? What does that imply about how you must apply it consistently to both training and test data?

> *Your answer:* Yes, because μ and σ are sample statistics, computing them on a different subset of students will produce slightly different means and standard deviations, and therefore a slightly different matrix S = diag(1/σ). This implies that in a machine learning pipeline we must compute μ and S once (on the training set) and apply that exact same μ and S to both the training data and the test data (and any future unseen inputs). If we recomputed a separate transformation matrix on the test set, the coordinate system of the test features would not match the coordinate system in which the regression weights β̂ were trained, causing inconsistent predictions and data leakage.

### Save your work for the next notebook
"""

np.savez('partD_data.npz',
         X_transformed=X_transformed, X_train=X_train, X_test=X_test,
         y_train=y_train, y_test=y_test, feature_names=feature_names)
if os.path.exists('/content/drive/MyDrive'):
    np.savez('/content/drive/MyDrive/partD_data.npz',
             X_transformed=X_transformed, X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test, feature_names=feature_names)
print("Saved X_transformed, X_train, X_test, y_train, y_test for Part E.")
