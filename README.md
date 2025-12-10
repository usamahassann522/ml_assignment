# Support Vector Machines: Finding the Optimal Decision Boundary

**Name:** Usama Hassan  
**Course:** 24060653
**Assignment:** Machine Learning

---

## 📋 Overview

This tutorial provides a comprehensive exploration of **Support Vector Machines (SVMs)** and how kernel choice and hyperparameters dramatically affect classification performance. Through systematic experimentation on three distinct datasets, we demonstrate that proper kernel selection and parameter tuning are critical for SVM success.

**Key Question:** How do kernel choice (linear vs RBF) and hyperparameters (C and gamma) control SVM decision boundaries?

---


## 📂 Files in This Directory

```
ml_assignment/
├── README.md                           # This file
├── svm.ipynb                           # Google Colab notebook (full code)
├── report.pdf                          # PDF version of tutorial
```

---

## Experimental Design

### Datasets

Three synthetic datasets testing SVM flexibility:

**1. Linear Dataset (Easy)**
- 200 samples, 2 classes
- Perfectly separable with a straight line
- Tests: Linear kernel performance

**2. Circles Dataset (Hard)**
- 200 samples, 2 classes
- Concentric circles (impossible to separate linearly)
- Tests: Necessity of non-linear kernels

**3. Moons Dataset (Medium)**
- 200 samples, 2 classes
- Two interleaving crescent shapes
- Tests: RBF kernel parameter sensitivity

### Experiments Conducted

1. **Experiment 1:** Linear SVM with C parameter tuning
   - Tested C = [0.01, 0.1, 1, 10, 100, 1000]
   - Observed margin width changes
   - Counted support vectors

2. **Experiment 2:** Kernel comparison (Linear vs RBF)
   - Applied both kernels to all three datasets
   - Demonstrated when linear kernels fail

3. **Experiment 3:** RBF gamma parameter tuning
   - Tested gamma = [0.01, 0.1, 1, 10, 100, 1000]
   - Observed boundary complexity progression

4. **Experiment 4:** Grid search optimization
   - Systematic search over C × gamma combinations
   - Found optimal parameters for best generalization

---

## Key Results

### Main Findings

| Experiment | Parameter | Too Small | Optimal | Too Large |
|------------|-----------|-----------|---------|-----------|
| **Linear SVM** | C (regularization) | C=0.01 (122 SVs, wide margin) | C=0.1-10 (55-72 SVs) | C=100+ (52 SVs, narrow margin) |
| **Kernel Choice** | Kernel type | Linear on circles: **46.7%** | RBF on all data: **98.3%** | N/A |
| **RBF Gamma** | Gamma (width) | γ=0.01 (too smooth, 85%) | γ=0.1-1 (optimal, 98.3%) | γ=100+ (too wiggly, 61.7%) |

### Performance Summary

**Best Configuration:**
- **Kernel:** RBF (Radial Basis Function)
- **C:** 0.1 (regularization strength)
- **Gamma:** 1.0 (kernel coefficient)
- **Test Accuracy:** 98.3%
- **Support Vectors:** 44 (reasonable complexity)

**Key Insight:** Kernel choice matters MORE than parameter tuning. Wrong kernel = fundamental failure (46.7% on circles), but right kernel with proper tuning = excellent performance (98.3%).

---

## Visualizations

### 1. Dataset Overview
![Datasets](ml_assignment/image_0.png)

Three dataset types showing progression from easy (linear) to impossible (circles) to medium (moons) difficulty.

### 2. C Parameter Impact (Linear Kernel)
![C Parameter](images/svm_linear_c_comparison.png)

Six subplots showing how C controls margin width:
- **Low C (orange):** Wide margin, many support vectors, tolerates errors
- **Medium C (green):** Balanced margin and accuracy
- **High C (red):** Narrow margin, few support vectors, minimizes errors

### 3. Kernel Comparison
![Kernel Comparison](images/svm_kernel_comparison.png)

2×3 grid comparing linear and RBF kernels across all datasets. Shows dramatic failure of linear kernel on circles dataset (46.7% → 98.3% with RBF).

### 4. Gamma Parameter Impact (RBF Kernel)
![Gamma Parameter](images/svm_rbf_gamma_comparison.png)

Six subplots showing gamma's effect on decision boundary complexity:
- **Low gamma (orange):** Smooth, almost linear boundary (underfits)
- **Medium gamma (green):** Captures curved patterns perfectly
- **High gamma (red):** Extremely wiggly, overfits training noise

### 5. Grid Search Results
![Grid Search](images/svm_grid_search_heatmap.png)

Heatmap showing test accuracy for all C × gamma combinations. Darker green = better performance. Reveals optimal region at C=0.1-1, gamma=0.1-1.

### 6. Performance Summary
![Performance Summary](images/svm_performance_summary.png)

Three-panel summary showing:
- Left: C parameter effect on accuracy
- Middle: Kernel comparison across datasets
- Right: Support vectors vs C (inverse relationship)

### 7. Confusion Matrices
![Confusion Matrices](images/svm_confusion_matrices.png)

Error analysis for three configurations:
- Linear kernel: 83.3% accuracy, 10 errors
- RBF default: 86.7% accuracy, 8 errors
- RBF optimal: 98.3% accuracy, 1 error

---

## Running the Code

### Local Jupyter Notebook

```bash
# Clone repository
git clone https://github.com/usamahassann522/ml_assignment.git
cd ml-tutorials/4-svm

# Install dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# Launch Jupyter
jupyter notebook svm.ipynb

# Run all cells
```


##  Key Concepts Explained

### 1. What is SVM?

Support Vector Machines find the **optimal decision boundary** by maximizing the margin (distance) to the nearest data points from each class. These nearest points are called **support vectors**.

**Analogy:** Imagine drawing the widest possible street between two neighborhoods. The houses on the edge of the street (support vectors) determine where the street goes.

### 2. The C Parameter

**C controls the regularization strength** (trade-off between margin width and training accuracy):

- **Small C:** Wide margin, tolerates some misclassifications (soft margin)
- **Large C:** Narrow margin, minimizes misclassifications (hard margin)

**Formula:** SVM minimizes: `½||w||² + C × Σ(penalties for errors)`

### 3. The Kernel Trick

Kernels allow SVMs to create non-linear boundaries without explicitly transforming features:

**Linear Kernel:** `K(x, y) = x·y`
- Fast and simple
- Only works for linearly separable data

**RBF Kernel:** `K(x, y) = exp(-γ||x-y||²)`
- Can create any curved boundary
- Requires tuning gamma

### 4. The Gamma Parameter

**Gamma controls how far the influence of a single training example reaches:**

- **Small gamma:** Wide influence → smooth boundary
- **Large gamma:** Narrow influence → complex boundary

**Think of it as:** "How local should decisions be?"

### 5. Support Vectors

Only the points closest to the decision boundary (support vectors) matter for the final model. Other points could be removed without changing the boundary!

---

## Understanding the Results

### When Linear Kernels Work

**Linear kernel is perfect when:**
-  Data is linearly separable
-  Speed is critical
-  Interpretability matters
-  High-dimensional data (text classification)

**Linear kernel fails when:**
- Classes form circles, curves, or complex shapes
-  XOR-like patterns exist

### When to Use RBF Kernel

**RBF kernel is needed when:**
- Data has curved boundaries
- Linear kernel performs poorly
- Willing to tune hyperparameters
- Have sufficient training data

**Warning:** RBF without tuning often overfits!

### The Bias-Variance Trade-off

**High Bias (Underfitting):**
- Linear kernel on circles: **46.7%**
- Small C, small gamma: **85%**
- **Problem:** Model too simple

**Balanced:**
- RBF with C=0.1-1, gamma=0.1-1: **98.3%**
- **Success:** Captures patterns without memorizing

**High Variance (Overfitting):**
- Very large gamma (100+): **61.7%**
- **Problem:** Memorizes training noise

---

## Practical Guidelines

### Step-by-Step Tuning Strategy

**Step 1: Try Linear Kernel First**
```python
svm = SVC(kernel='linear', C=1.0)
svm.fit(X_train, y_train)
```
- If accuracy > 85%, you're done!
- If accuracy < 80%, proceed to Step 2

**Step 2: Switch to RBF Kernel**
```python
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)
```
- Check if performance improves significantly
- If yes, proceed to Step 3

**Step 3: Grid Search for Optimal Parameters**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10]
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.3f}")
```

**Step 4: Evaluate on Test Set**
```python
best_model = grid.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.3f}")
```

### Best Practices

**Always:**
1. **Scale features** before training SVM (use StandardScaler)
2. **Use cross-validation** for parameter selection
3. **Start with linear kernel** (faster, simpler)
4. **Check support vector count** (too many = overfit risk)
5. **Try default gamma='scale'** before manual tuning

**Never:**
1. Forget to scale features (SVM is distance-based!)
2. Use RBF without tuning gamma
3. Tune parameters on test set (data leakage!)
4. Use very large gamma (>10 usually overfits)
5. Apply SVM to very large datasets (>100k samples) without consideration

### Common Pitfalls

**Pitfall 1: Forgetting to Scale**
```python
# BAD - features on different scales
svm.fit(X_train, y_train)

# GOOD - scale first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
svm.fit(X_train_scaled, y_train)
```

**Pitfall 2: Wrong Kernel Choice**
- If linear gives 50% on circles → MUST switch to RBF
- Don't try to fix with different C values

**Pitfall 3: Extreme Parameter Values**
- gamma=1000 → Almost always overfits
- C=0.001 → Almost always underfits

---

## 📚 References

[1] Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

[2] Boser, B. E., Guyon, I. M., & Vapnik, V. N. (1992). A training algorithm for optimal margin classifiers. *Proceedings of the 5th Annual Workshop on Computational Learning Theory*, 144-152.