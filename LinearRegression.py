# Content from Oreilly's Media

"""Machine Learning Basics — From Features to RMSE

1. Features and Labels
Features (X): Input variables used by a model to make predictions.
Example (house price prediction): Area, number of rooms, location, age of house.
Labels (y): The correct output we want the model to predict.
Example: Actual house price.
In supervised learning, both features and labels are provided to the model during training.

2. Supervised Learning
Definition: The model learns from labeled examples (X, y) to predict y for new, unseen X.
Types:
  - Regression: Predict a continuous value (e.g., price, temperature).
  - Classification: Predict a category (e.g., spam or not spam).

3. Linear Regression
Goal: Predict a continuous label as a linear combination of features.
Equation for single feature:
  y_hat = m * x + c
Where:
  m = slope (how much y changes per unit change in x)
  c = intercept (value of y when x = 0)
Equation for multiple regression:
  y_hat = w1*x1 + w2*x2 + ... + wn*xn + b
"Fitting a line" means finding m and c (or w and b) so the line best matches the training data.

4. Training and Testing
1. Split data into:
      Training set (e.g., 80%)
      Test set (e.g., 20%)
2. Train the model on training set (X_train, y_train).
3. Test the model on unseen test set (X_test) and compare predictions to y_test.

5. Performance Measurement — RMSE
RMSE = Root Mean Square Error, a common metric for regression.
Purpose: Measures how far, on average, predictions are from actual values, penalizing large errors more.

RMSE Formula:
  RMSE = sqrt( (1/m) * sum( (y_hat(i) - y(i))^2 ) )
Where:
  m = number of samples
  y_hat(i) = predicted value for sample i
  y(i) = actual value for sample i

Why divide by m? To get the average squared error (MSE).
Why take square root? To bring the units back to the same scale as y.
"""


#Machine Learning Performance Metrics — MAE vs RMSE
"""
1. Mean Absolute Error (MAE)
- Definition: The average of the absolute differences between predictions and actual values.
- Formula:
    MAE = (1/m) * Σ |y_hat(i) - y(i)|
  where:
    m = number of samples
    y_hat(i) = predicted value for sample i
    y(i) = actual value for sample i
- Properties:
    * Measures the average size of errors.
    * Treats all errors equally, regardless of size.
    * Less sensitive to outliers.

2. Root Mean Square Error (RMSE)
- Definition: The square root of the average of the squared differences between predictions and actual values.
- Formula:
    RMSE = sqrt( (1/m) * Σ (y_hat(i) - y(i))² )
  where:
    m = number of samples
    y_hat(i) = predicted value for sample i
    y(i) = actual value for sample i
- Properties:
    * Penalizes large errors more due to squaring.
    * More sensitive to outliers.
    * Same units as the target variable.

3. Relationship and Interpretation
- If RMSE ≈ MAE:
    * Errors are consistent.
    * No major outliers affecting predictions.
- If RMSE >> MAE:
    * Indicates presence of large errors or outliers.
    * Worth investigating those cases to improve model performance.
- If both MAE and RMSE are large:
    * Model generally performs poorly.
    * May need better features, more data, or a different algorithm.

4. Choosing Between MAE and RMSE
- Use MAE when:
    * All errors are equally important.
    * Outliers are common and you don’t want them to dominate the metric.
- Use RMSE when:
    * Large errors are especially undesirable.
    * Outliers are rare, and you want to penalize them more.


"""

#Stable Train/Test Split Using Hashing
"""
Purpose:
When datasets are updated over time (new rows added, some removed), a normal random split can shuffle data differently each time.
This can cause:
1. Data leakage — rows that were in the test set before might end up in the training set later.
2. Inconsistent evaluation — your test set changes, so results can’t be compared fairly across runs.

Solution:
Use a stable, deterministic method to decide whether each row goes into the training set or test set.
One common method is hashing.

Process:
1. Each row in the dataset has a unique and unchanging identifier (e.g., ID number, customer ID, product code).
2. Apply a hash function to this ID.
   - A hash function turns the ID into a large integer.
   - The same ID will always produce the same hash value.
   - Different IDs will produce different values, evenly spread over a range.
3. Normalize the hash value:
   - Divide the hash result by the maximum possible hash value.
   - This converts it to a range between 0 and 1 (normalized value).
4. Compare to a threshold:
   - Decide the percentage of data for the test set (e.g., 20% → threshold = 0.2).
   - If the normalized value ≤ threshold → assign to the test set.
   - If it is greater → assign to the training set.

Why it is stable:
- The hash for the same ID never changes.
- The train/test assignment depends only on the ID, not on random sampling.
- If new data is added, old rows keep their assignment.
- About 20% of any new rows will go to the test set, the rest to the training set.

Example:
- ID 1 → hash result = 2212294583 → normalized = 0.5149 → Train (since > 0.2)
- ID 4 → hash result = 682475091  → normalized = 0.1588 → Test (since ≤ 0.2)
Even if you add IDs 11–20 later, IDs 1 and 4 will remain in their original sets.

#Normalizing the hash value:

We divide by the maximum possible CRC32 value: for example if hashing give 1= 2212294583, we divide by 2^32 = 4294967296 to get 0.5149.
"""

#StratifiedShuffleSplit vs train_test_split with stratify
"""
1. Regular train_test_split
- Splits the dataset randomly into train and test sets.
- If the dataset is large and well-distributed, this is usually fine.
- Problem: For smaller datasets or when certain categories are important, random splitting can change category proportions between train and test sets, causing sampling bias.

2. Stratified sampling
- Goal: Keep the same proportion of an important category in both train and test sets.
- This is useful when a specific feature strongly influences the target variable and you want both subsets to represent the overall population fairly.

3. StratifiedShuffleSplit
- Allows multiple different stratified splits.
- Preserves the proportion of the chosen category (stratum) in each split.
- Returns train and test indices; you use these indices to select rows from the dataset.
- Parameters:
    * n_splits: Number of different splits to generate.
    * test_size: Fraction of data to put in the test set.
    * random_state: Ensures reproducibility of splits.
- Use case: Cross-validation or repeated experiments with stratification.

4. train_test_split(..., stratify=...)
- One-time stratified split.
- Shorter syntax: directly returns the train and test sets without needing to handle indices manually.
- Parameters:
    * test_size: Fraction of data to put in the test set.
    * stratify: The column (Series/array) whose proportion you want to preserve.
    * random_state: Ensures reproducibility of the split.
- Use case: Quick, one-time stratified split without multiple repetitions.

"""

#Data Cleaning — Handling Missing Values - SimpleImputer vs IterativeImputer
"""
Definition:
Data cleaning is the process of identifying and fixing or removing incorrect, corrupted, or missing parts of the dataset to ensure quality before model training.

Missing values:
Missing values can occur due to data entry errors, incomplete measurements, or merging datasets. They are often represented as NaN (Not a Number) in pandas.

Scikit-learn provides tools to handle missing values, including SimpleImputer and IterativeImputer.

1. SimpleImputer
- Purpose: Fill missing values using a fixed strategy.
- Strategies:
    * "mean" — replace missing values with the column mean.
    * "median" — replace with the column median.
    * "most_frequent" — replace with the most common value in the column.
    * "constant" — replace with a constant value specified by fill_value.
- Example usage:
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    housing_num = imputer.fit_transform(housing_num)

2. IterativeImputer
- Purpose: Predict missing values using other features in the dataset.
- Method:
    1. Select a feature with missing values.
    2. Use all other features as predictors in a regression model.
    3. Predict and fill the missing values.
    4. Repeat for other features with missing values.
    5. Iterate the process multiple times to refine estimates.
- Parameters:
    * estimator — model used to predict missing values (default: BayesianRidge for continuous data).
    * max_iter — number of imputation iterations.
- Example usage:
    from sklearn.experimental import enable_iterative_imputer
    from sklearn.impute import IterativeImputer
    imputer = IterativeImputer(max_iter=10, random_state=42)
    housing_num = imputer.fit_transform(housing_num)

"""
#Feature Scaling — StandardScaler vs MinMaxScaler

"""

Definition:
Feature scaling is the process of transforming input features so they are on comparable numerical ranges. This prevents features with larger values from dominating features with smaller values in machine learning algorithms.

Why it matters:
- Many ML algorithms (e.g., Linear Regression with gradient descent, SVM, KNN, PCA, Neural Networks) are sensitive to the scale of features.
- Without scaling, large-range features can bias the model or slow convergence during training.
- Scaling ensures that all features contribute equally to the learning process.

When it matters most:
- Algorithms based on distances (KNN, K-Means, SVM with RBF kernel).
- Algorithms that use gradient descent optimization.
- PCA (Principal Component Analysis), since it is variance-based.

When it matters less:
- Tree-based algorithms (Decision Trees, Random Forests, Gradient Boosted Trees) are not scale-sensitive.

Common Scaling Methods:

1. Standardization (Z-score scaling)
   Formula:
       x' = (x - μ) / σ
       where μ = mean of the feature, σ = standard deviation
   Result:
       - Mean = 0
       - Standard deviation = 1
   Notes:
       - Works well for normally distributed features.
       - Values can be negative.

2. Min-Max Scaling (Normalization)
   Formula:
       x' = (x - min) / (max - min)
   Result:
       - Scales values to a fixed range, usually [0, 1]
   Notes:
       - Preserves relative relationships.
       - Sensitive to outliers.

3. Robust Scaling
   Formula:
       x' = (x - median) / IQR
       where IQR = Interquartile Range (Q3 - Q1)
   Result:
       - Less sensitive to outliers.
   Notes:
       - Useful when data has extreme values.

Example:
Suppose we have:
Size (sqft): 1400, 2000, 3000
Bedrooms: 3, 4, 5

Min-Max Scaling to [0, 1]:
Size_scaled:
    1400 → (1400 - 1400) / (3000 - 1400) = 0.000
    2000 → (2000 - 1400) / (3000 - 1400) = 0.375
    3000 → (3000 - 1400) / (3000 - 1400) = 1.000
Bedrooms_scaled:
    3 → (3 - 3) / (5 - 3) = 0.00
    4 → (4 - 3) / (5 - 3) = 0.50
    5 → (5 - 3) / (5 - 3) = 1.00

Standardization (Z-score):
Size mean = 2133.33, std ≈ 802.77
    1400 → (1400 - 2133.33) / 802.77 ≈ -0.915
    2000 → (2000 - 2133.33) / 802.77 ≈ -0.166
    3000 → (3000 - 2133.33) / 802.77 ≈  1.081
Bedrooms mean = 4, std ≈ 0.816
    3 → (3 - 4) / 0.816 ≈ -1.225
    4 → (4 - 4) / 0.816 =  0.000
    5 → (5 - 4) / 0.816 ≈  1.225

"""
"""Gradient Descent — Intuition to Practice

1) Problem Setup (Linear Regression)
We want to learn parameters θ = [θ0, θ1, …, θn]^T so predictions match targets y.
Prediction (vector form): ŷ = θ^T x  (for 1D: ŷ = θ0 + θ1 x)

Loss we minimize (training objective): Mean Squared Error (MSE)
    J(θ) = (1/m) * Σ_i (ŷ(i) − y(i))^2
Where:
  m = number of training examples
  x^(i) = features of example i (x0 = 1 for bias)
  ŷ(i) = θ^T x^(i)

2) Why “Gradient” Descent?
- The gradient ∇_θ J(θ) points in the direction of steepest INCREASE of the loss.
- To DECREASE the loss, we move in the opposite direction:
      θ ← θ − η * ∇_θ J(θ)
  η (“eta”) is the learning rate: a hyperparameter we choose to control step size.

3) Gradients for Linear Regression with MSE
For each parameter θ_j:
    ∂J/∂θ_j = (2/m) * Σ_i ( (θ^T x^(i) − y^(i)) * x_j^(i) )
Vectorized (X_b = design matrix with bias col, shape m×(n+1)):
    ∇_θ J(θ) = (2/m) * X_b^T (X_b θ − y)

4) Learning Rate (η)
- Not computed from data; YOU set it (hyperparameter).
- Too small → slow convergence; too large → divergence/oscillation.
- Typical starting points: 0.1, 0.01, 0.001
- Often use a schedule to reduce η over time (see “Schedules” below).

5) Three Flavors of Gradient Descent
A) Batch Gradient Descent (BGD)
   - Uses ALL m examples to compute each gradient step.
   - Update:
       θ ← θ − η * (2/m) * X_b^T (X_b θ − y)
   - Stable direction, but each step is expensive for large m.

B) Stochastic Gradient Descent (SGD)
   - Uses ONE example per step.
   - For a single (x^(i), y^(i)):
       θ ← θ − η * 2 * ( (θ^T x^(i) − y^(i)) * x^(i) )
     (No division by m in pure SGD.)
   - Very fast updates, noisy trajectory; good for huge datasets/out-of-core.

C) Mini-batch Gradient Descent (MBGD)
   - Uses a small batch of b examples per step (2 ≤ b ≪ m).
   - Compromise: faster than batch, smoother than SGD.
   - Works well with modern hardware (vectorized math on batches).

6) Tiny Worked Example (Makes the math concrete)
Dataset (perfect line y = 2x):
   (x, y): (1,2), (2,4), (3,6)
Add bias: x0=1 → x^(i) = [1, x]^T. Start θ0=0, θ1=0. Let η=0.1.

First Batch-GD step:
  Predictions at start: ŷ = 0
  Errors: e = ŷ − y = [-2, -4, -6]
  Gradient components:
    ∂J/∂θ0 = (2/3) * (-2 + -4 + -6) = (2/3) * (-12) = -8
    ∂J/∂θ1 = (2/3) * [(-2)*1 + (-4)*2 + (-6)*3] = (2/3)*(-28) = -18.666...
  Update:
    θ0 ← 0 − 0.1 * (-8)      = 0.8
    θ1 ← 0 − 0.1 * (-18.667) = 1.8667
  Interpretation:
    One step already moved slope near the true 2. Repeating converges quickly.

First SGD pass (in order 1→2→3), same η=0.1:
  Start θ=[0,0]^T
  i=1: x=[1,1], y=2 → error = (0)−2 = -2
       grads: dθ0=2*(-2)*1=-4, dθ1=2*(-2)*1=-4
       θ ← θ − 0.1*[ -4, -4 ]^T = [0.4, 0.4]
  i=2: x=[1,2], y=4 → ŷ=0.4+0.4*2=1.2, error=-2.8
       dθ0=-5.6, dθ1=-11.2
       θ ← [0.4,0.4] − 0.1*[-5.6,-11.2] = [0.96, 1.52]
  i=3: x=[1,3], y=6 → ŷ=0.96+1.52*3=5.52, error=-0.48
       dθ0=-0.96, dθ1=-1.44
       θ ← [0.96,1.52] − 0.1*[-0.96,-1.44] = [1.056, 1.664]
  Interpretation:
    Faster early movement but noisy; over many passes (epochs), it hovers near optimum.

Mini-batch example (batch size b=2):
  Batch 1: points 1 & 2 → compute gradient over those 2, update.
  Batch 2: point 3 (and optionally wrap with point 1) → gradient, update.
  Interpretation:
    Steps are less noisy than SGD, cheaper than full batch; common default in practice.
    """