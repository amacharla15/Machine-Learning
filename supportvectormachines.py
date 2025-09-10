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

"""