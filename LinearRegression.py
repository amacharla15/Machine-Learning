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

"""