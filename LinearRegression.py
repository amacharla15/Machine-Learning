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

"""
Machine Learning Performance Metrics — MAE vs RMSE

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
