import pandas as pd
import numpy as np
from scipy.stats import t

import linear_regression as lr

data = pd.read_csv("datasets/b1.csv")[["x2", "x7", "x8", "y"]]

x_data = data[["x2", "x7", "x8"]].values
y_data = data["y"].values

# add a column of ones to x_data, as first column
X_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))

model = lr.MultipleLinearRegression(x_data, y_data)

C = np.linalg.inv(np.matmul(X_data.T, X_data))

x0 = np.array([1, 2300, 56, 2100]).T
prediction = model.predict(np.array([2300, 56, 2100]))

t_stat = t.ppf(0.975, 24)

temp = pow(model.MSRes * (1+np.dot(x0, np.matmul(C, x0))), 0.5)

lower_bound = prediction - t_stat * temp
upper_bound = prediction + t_stat * temp
print(lower_bound, upper_bound)