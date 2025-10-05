# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```

Program to implement the linear regression using gradient descent.
Developed by: A.Abinesh
RegisterNumber:  25017255
import numpy as np
from sklearn.preprocessing import StandardScaler

# Linear Regression using Gradient Descent
def linear_regression(X1, y, learning_rate=0.1, num_iters=1000):
    X = np.c_[np.ones(len(X1)), X1]  # Add intercept term
    theta = np.zeros(X.shape[1]).reshape(-1, 1)
    
    for _ in range(num_iters):
        predictions = X.dot(theta).reshape(-1, 1)
        errors = (predictions - y).reshape(-1, 1)
        theta -= learning_rate * (1 / len(X1)) * X.T.dot(errors)  # Gradient descent update
    
    return theta

# Sample data (similar to 50_Startups dataset)
# Columns: R&D Spend, Administration, Marketing Spend
X = np.array([
    [165349.2, 136897.8, 471784.1],
    [162597.7, 151377.6, 443898.5],
    [153441.5, 101145.5, 407934.5],
    [144372.4, 118671.9, 383199.6],
    [142107.3, 91391.77, 366168.4],
    [131876.9, 99814.71, 362861.4],
    [134615.5, 147198.9, 127716.8],
    [130298.1, 145530.1, 323876.7],
    [120542.5, 148718.9, 311613.3],
    [123334.9, 108679.2, 304981.6]
], dtype=float)

# Target variable: Profit
y = np.array([
    192261.83,
    191792.06,
    191050.39,
    182901.99,
    166187.94,
    156991.12,
    156122.51,
    155752.6,
    152211.77,
    149759.96
]).reshape(-1, 1)

# Scaling features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Train Linear Regression model
theta = linear_regression(X_scaled, y_scaled, learning_rate=0.1, num_iters=10000)

print("Learned parameters (theta):")
print(theta)

# Predicting for a new data point
new_data = np.array([165349.2, 136897.8, 471784.1]).reshape(1, -1)
new_scaled = scaler_X.transform(new_data)  # scale new input
prediction_scaled = np.dot(np.c_[1, new_scaled], theta)  # add intercept
prediction = scaler_y.inverse_transform(prediction_scaled)  # inverse scale to get original value

print(f"Predicted profit for {new_data.flatten()}: {prediction.flatten()[0]:.2f}")



```

## Output:
```
Learned parameters (theta):
[[-6.66133815e-18]
 [ 8.42189317e-01]
 [-5.77735216e-03]
 [ 1.63704478e-01]]
Predicted profit for [165349.2 136897.8 471784.1]: 196813.34
```


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
