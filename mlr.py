import linear_regression as lr
import numpy as np


delivery_times = [16.68, 11.50, 12.03, 14.88, 13.75, 18.11, 8.0, 17.83, 79.24, 21.5, 40.33, 21.0, 13.50, 19.75, 24.0, 29.0, 15.35, 19.0, 9.5, 35.1, 17.9, 52.32, 18.75, 19.83, 10.75]
number_cases = [7, 3, 3, 4, 6, 7, 2, 7, 30, 5, 16, 10, 4, 6, 9, 10, 6, 7, 3, 17, 10, 26, 9, 8, 4]
distances = [560, 220, 340, 80, 150, 330, 110, 210, 1460, 605, 688, 215, 255, 462, 448, 776, 200, 132, 36, 770, 140, 810, 450, 635, 150]

X = np.array([np.ones(len(delivery_times)), number_cases, distances]).T
y = np.array(delivery_times)

model = lr.MultipleLinearRegression(X, y)
print(model.prediction_confidence_interval(np.array([1, 8, 275]), 0.05))