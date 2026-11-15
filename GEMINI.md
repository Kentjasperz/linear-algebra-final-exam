# Linear Algebra Final Exam — The Linear Algebra Behind AI: A Regression Problem

## Project Overview
This workspace contains the final exam project **"The Linear Algebra Behind AI — A Regression Problem"**, structured as a sequence of Jupyter notebooks (`Part_A` through `Part_E`) and reference Python scripts (`for_part_a.py`, `for_part_b.py`, `for_part_c.py`).

The project explores linear regression from a linear algebra perspective using the **Student Performance Factors** dataset from Kaggle (`https://www.kaggle.com/datasets/lainguyn123/student-performance-factors`).

## Dataset & Problem Formulation
- **Dataset:** `Student_Performance_Factors.csv` ($n = 6,607$ samples, 20 total dataset columns; $p = 4$ selected numeric predictors).
- **Selected Features (`feature_names`):**
  1. `Attendance` (percentage of classes attended)
  2. `Hours_Studied` (weekly study hours)
  3. `Previous_Scores` (scores on earlier exams)
  4. `Sleep_Hours` (average hours of sleep per night)
- **Target Variable (`y`):** `Exam_Score` (continuous numerical exam score).

## Workspace Structure

### Jupyter Notebooks (Templates / Deliverables)
- `Part_A_Problem_and_Dataset (2).ipynb` — **Part A: Problem & Dataset Selection** (Q1–Q4). Loads `X_raw`, `y`, and `feature_names`, inspects summary statistics, and stores variables via `%store` (or `.npz` on Google Drive when running in Colab).
- `Part_B_Matrices.ipynb` — **Part B: Matrices** (Q5–Q8). Constructs the design matrix $X \in \mathbb{R}^{n \times p}$, computes the Gram matrix $X^T X \in \mathbb{R}^{p \times p}$, and inspects off-diagonal inner products.
- `Part_C_Vector_Spaces.ipynb` — **Part C: Vector Spaces** (Q9–Q12). Computes $\text{rank}(X)$, singular values, condition number, column space interpretation, and linear independence of features.
- `Part_D_Linear_Transformations (1).ipynb` — **Part D: Linear Transformations** (Q13–Q16). Applies standardization via diagonal scaling matrix $S = \text{diag}(1/\sigma)$ on centered data $X_{\text{centered}} = X - \mu$, inspects individual sample transformations, and performs an 80/20 train/test split (`rng = np.random.default_rng(42)`).
- `Part_E_Determinants (2).ipynb` — **Part E: Determinants** (Q17–Q20). Computes $X_{\text{train}}^T X_{\text{train}}$, its determinant $\det(X_{\text{train}}^T X_{\text{train}})$, and its condition number to verify invertibility before solving normal equations.

### Reference Scripts (Completed Drafts / Colab Exports)
- `for_part_a.py` — Colab script for Part A (loads `/content/Student_Performance_Factors.csv`, defines `feature_names`, `X_raw`, `y`, and draft answers to Q1–Q4).
- `for_part_b.py` — Completed Colab script and Q5–Q8 written answers for Part B.
- `for_part_c.py` — Completed Colab script and Q9–Q12 written answers for Part C.
- `for_part_d.py` — Completed Colab script and Q13–Q16 written answers for Part D.
- `for_part_e.py` — Completed Colab script and Q17–Q20 written answers for Part E.

## Conventions & Guidelines
- **Numeric Formatting:** Always configure NumPy and Pandas for 4 decimal places:
  ```python
  np.set_printoptions(precision=4, suppress=True)
  pd.set_option('display.float_format', lambda x: f'{x:.4f}')
  ```
- **Data Passing Across Notebooks:**
  - Local Jupyter sessions use IPython `%store` (`%store X_raw`, `%store -r X_raw y feature_names`).
  - Google Colab sessions mount Google Drive (`/content/drive`) and save/load `.npz` archives (`partA_data.npz`, `partB_data.npz`, etc.).
- **Written Answers Style:** Keep report answers (Q1–Q20) clear, mathematically accurate, and directly tied to the concrete numbers and features (`Attendance`, `Hours_Studied`, `Previous_Scores`, `Sleep_Hours`, `Exam_Score`, $n = 6,607$, $p = 4$) of the dataset.
