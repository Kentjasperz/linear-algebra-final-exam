# -*- coding: utf-8 -*-
"""for_Part_E.ipynb

# Part E — Determinants

**Final Project: The Linear Algebra Behind AI — A Regression Problem**

Before you can solve for regression coefficients using matrix inversion, you need to know the inversion is actually possible.

---
**Student name:** _______________________   **Date:** _______________________

---

**What to do:** Compute det(XᵀX) on your training data (after your Part D transformation) before attempting to invert it.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
pd.set_option('display.float_format', lambda x: f'{x:.4f}')

search_dirs = [Path.cwd(), Path('/content'), Path('/content/drive/MyDrive')]
npz_path = next((d / 'partD_data.npz' for d in search_dirs if (d / 'partD_data.npz').exists()), None)
if npz_path is None:
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive/MyDrive'):
            drive.mount('/content/drive')
        if os.path.exists('/content/drive/MyDrive/partD_data.npz'):
            npz_path = Path('/content/drive/MyDrive/partD_data.npz')
    except Exception:
        pass

data = np.load(npz_path or 'partD_data.npz', allow_pickle=True)
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
print('Loaded X_train:', X_train.shape, ' X_test:', X_test.shape)

# Compute X_train^T X_train and its determinant
XtX_train = X_train.T @ X_train
det_XtX_train = np.linalg.det(XtX_train)
print("X_train^T X_train:\n", XtX_train)
print(f"det(X_train^T X_train) = {det_XtX_train:.4f}")

# Decide, based on the determinant, whether XtX_train is invertible.
# Also compute its condition number for a stability check.
cond_number = np.linalg.cond(XtX_train)
print(f"Condition number = {cond_number:.4f}")
if not np.isclose(det_XtX_train, 0.0):
    print("Conclusion: det(X_train^T X_train) != 0, so X_train^T X_train is invertible.")
else:
    print("Conclusion: det(X_train^T X_train) is close to 0; matrix is singular or nearly singular.")

"""### Answer these in your report

**Q17.** What is det(XᵀX) for your training data?

> *Your answer:* For our standardized training matrix X_train ∈ ℝ⁵²⁸⁵ˣ⁴ (80% of the 6,607 students), det(X_trainᵀ X_train) = 785,932,759,224,407.3750 (approximately 7.8593 × 10¹⁴). Because each standardized feature has unit variance, the diagonal entries of X_trainᵀ X_train are close to n_train = 5,285 ([5286.8236, 5307.7899, 5293.5145, 5298.5532]) and the off-diagonal entries are relatively small, making the determinant close to (5,285)⁴ and far from zero.

**Q18.** Based on that value, is XᵀX invertible? If it were numerically singular, what would you do instead?

> *Your answer:* Yes, because det(X_trainᵀ X_train) = 7.8593 × 10¹⁴ ≠ 0, X_trainᵀ X_train is nonsingular and invertible, so the normal equations β̂ = (X_trainᵀ X_train)⁻¹ X_trainᵀ y_train have a unique closed-form solution. If X_trainᵀ X_train were numerically singular (det ≈ 0), we could instead: (1) drop or combine the redundant/collinear features (or use PCA to keep only directions with nonzero variance), (2) compute the least-squares solution using the Moore–Penrose pseudoinverse (`np.linalg.pinv`) or SVD-based solver (`np.linalg.lstsq`), or (3) use Ridge regression (L₂ regularization), replacing XᵀX with XᵀX + λI (λ > 0) so all eigenvalues are shifted above zero and the matrix becomes invertible.

**Q19.** Describe a realistic situation in your own dataset (e.g., two nearly duplicate features) that could push this determinant close to zero.

> *Your answer:* In our student dataset, det(X_trainᵀ X_train) would be pushed close to zero if we included two features that measure nearly the same quantity—for example, keeping `Hours_Studied` (weekly study hours) while also adding a column `Daily_Study_Hours` ≈ `Hours_Studied / 7` (with slight rounding noise), or including both `Attendance` (percentage of classes attended) and `Classes_Missed` out of a fixed semester schedule. Those two columns would be nearly collinear, causing the smallest eigenvalue of X_trainᵀ X_train—and thus the determinant, which equals the product of the eigenvalues—to drop close to zero.

**Q20.** Look up or compute the condition number of XᵀX. How does a large condition number relate to the determinant, and what does it suggest about the stability of your model?

> *Your answer:* The condition number of X_trainᵀ X_train is κ(X_trainᵀ X_train) = λ_max / λ_min = 1.0747, the ratio of its largest to smallest eigenvalue. Because our four standardized features are nearly orthogonal, all four eigenvalues are close to n_train = 5,285, giving a condition number of 1.0747 (very close to the ideal value of 1), which indicates an exceptionally well-conditioned and numerically stable model. In general, since det(XᵀX) = ∏ᵢ λᵢ, near-collinearity drives the smallest eigenvalue λ_min → 0, which simultaneously pushes det(XᵀX) → 0 (for normalized columns) and causes κ(XᵀX) = λ_max / λ_min → ∞. A large condition number means the matrix is close to singular and the regression model is numerically unstable: small changes or noise in the data can cause large swings in the estimated coefficients β̂.

### Save your work for the next notebook
"""

np.savez('partE_data.npz', XtX_train=XtX_train, det_XtX_train=det_XtX_train, cond_number=cond_number)
if os.path.exists('/content/drive/MyDrive'):
    np.savez('/content/drive/MyDrive/partE_data.npz',
             XtX_train=XtX_train, det_XtX_train=det_XtX_train, cond_number=cond_number)
print("Saved XtX_train for Part F.")
