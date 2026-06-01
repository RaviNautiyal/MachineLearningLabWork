
First, let's load the wine dataset from scikit-learn and prepare it for our linear regression model.
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Load the wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

print("Features (X) head:")
display(X.head())
print("\nTarget (y) head:")
display(y.head())
print("\nDataset shape:", X.shape, y.shape)
Next, we will split the dataset into training and testing sets. This allows us to train the model on one part of the data and evaluate its performance on unseen data.
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
Now, let's initialize and train a `LinearRegression` model using our training data.
# Initialize the Linear Regression model
model = LinearRegression()

# Train the model using the training sets
model.fit(X_train, y_train)

print("Model trained successfully.")
Finally, we'll make predictions on the test set and evaluate the model's performance using metrics like Mean Squared Error and R-squared.
# Make predictions using the testing set
y_pred = model.predict(X_test)

# The coefficients
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)

# The mean squared error
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination (R-squared): %.2f' % r2_score(y_test, y_pred))
To visualize the model's performance, let's create a scatter plot comparing the actual `y_test` values with the predicted `y_pred` values. A perfect model would show all points lying on the diagonal line.