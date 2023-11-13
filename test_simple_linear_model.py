# ---------------------------------------------------------------------------------
# SIMPLE LINEAR REGRESSION MODEL TESTING
# 
# Tests the SimpleLinearRegression class from the linear_regression module, for the following:
#     - correct calculation of model parameters using the least squares method
#     - proper analysis of variance reporting, along with the F-statistic and p-value for the model fit
#     - proper t-tests for the model parameters (H0: b1 = 0, H0: b0 = 0)
#     - calculation of R^2 score
#     - CI for the mean response
#     - CI for the model parameters
#     - CI for the predicted response
# 
# ---------------------------------------------------------------------------------

import linear_regression as lr
import pandas as pd
from scipy.stats import t



def test_model_parameter_estimates_01():
    """
    Tests the accuracy of the model parameter estimates for a sample dataset.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_slope = -0.007025
    expected_intercept = 21.7882

    assert abs(model._slope - expected_slope) < 0.001
    assert abs(model._intercept - expected_intercept) < 0.001


def test_analysis_of_variance_01():
    """
    Tests the analysis of variance table for sample dataset.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SST = 326.9643
    expected_SSR = 178.0923
    expected_SSRes = 148.8720
    expected_MSRes = 5.7258
    expected_Sxx = 3608611.42857

    assert abs(model.SST - expected_SST) < 0.001
    assert abs(model.SSR - expected_SSR) < 0.001
    assert abs(model.SSRes - expected_SSRes) < 0.001
    assert abs(model.MSRes - expected_MSRes) < 0.001
    assert abs(model.Sxx - expected_Sxx) < 0.001



def test_confidence_interval_slope_01():

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SEb1 = 0.0012596
    model_SEb1 = pow(model.MSRes / model.Sxx, 0.5)

    boundary_t = t.ppf(0.975, model.n - 2)
    expected_slope = -0.007025

    expected_lower_bound = expected_slope - boundary_t * expected_SEb1
    expected_upper_bound = expected_slope + boundary_t * expected_SEb1

    lower_bound, upper_bound = model.regressor_parameter_confidence_interval(j = 1, alpha = 0.05)

    assert abs(model_SEb1 - expected_SEb1) < 0.001
    assert abs(lower_bound - expected_lower_bound) < 0.001
    assert abs(upper_bound - expected_upper_bound) < 0.001


def test_confidence_interval_intercept_01():

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    n = len(x_data)
    mean_x = sum(x_data) / n

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SEb0 = 2.6962
    model_SEb0 = pow(model.MSRes * (1/n + pow(mean_x, 2) / model.Sxx), 0.5)

    expected_intercept = model._intercept
    boundary_t_statistic = t.ppf(0.975, model.n - 2)

    expected_lower_bound = expected_intercept - boundary_t_statistic * expected_SEb0
    expected_upper_bound = expected_intercept + boundary_t_statistic * expected_SEb0

    lower_bound, upper_bound = model.regressor_parameter_confidence_interval(j = 0, alpha = 0.05)

    assert abs(model_SEb0 - expected_SEb0) < 0.001
    assert abs(lower_bound - expected_lower_bound) < 0.001
    assert abs(upper_bound - expected_upper_bound) < 0.001
