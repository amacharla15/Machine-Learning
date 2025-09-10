#Support Vector Machines

"""
support vector machines are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis.
    They are a set of related supervised learning methods used for classification and regression.
    They work by finding the optimal separating boundary between data points of different classes 

    1)  data points are polotted in 2D or 3D space or etc,
    2) SVM tries to find a hyperplane that best divides a dataset into classes and the seperation is done with largest possible margin 
    3) The margin in the distance from the hyperplane to the closest data points from each class, these closest points are called seperate vectors

Margin:

The gap between the two nearest points from different classes and the hyperplane.
SVM aims to maximize this margin for better generalization.

Types of SVM:

1) Linear SVM
Works when data is linearly separable (a straight line can separate classes).
2) Non-Linear SVM
Uses the kernel trick to map data into a higher-dimensional space where it becomes linearly separable.



## Mathematical Intuition:
- Decision boundary equation: wᵀx + b = 0
- Margin width: 2 / ||w||
- Maximizing the margin ⇔ Minimizing ||w||.

---

## Soft Margin vs Hard Margin:
- **Hard Margin**: Perfectly separates classes with no errors (only works when data is clean & separable).
- **Soft Margin**: Allows some misclassifications to handle noisy or overlapping data.
  Controlled by **C** (regularization parameter).

---

## Kernels (for Non-Linear Separation):
- **Why**: Avoids manually creating new features by implicitly mapping to higher dimensions.
- **Common Types**:
  - Linear Kernel: K(x, x') = xᵀx'
  - Polynomial Kernel: captures polynomial relationships.
  - RBF (Gaussian): flexible and widely used.
  - Sigmoid Kernel: behaves like a neural network activation.

---

## Hyperparameters:
- **C**: Controls trade-off between margin size and classification error.
- **gamma**: In RBF/poly kernels, controls influence of each training sample.
- **degree**: For polynomial kernels, defines curve complexity.
"""