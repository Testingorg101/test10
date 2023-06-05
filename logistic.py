import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample input data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 1, 1, 1])

# Create a logistic regression model
model = LogisticRegression()

# Fit the model to the data
model.fit(X, y)

# Predict probabilities
X_test = np.array([[6], [7]])
y_prob = model.predict_proba(X_test)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Print the predicted probabilities
print("Predicted probabilities:", y_prob)
