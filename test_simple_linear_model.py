# --------------------------------------------------------------------------------------------------------
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
# The tests are performed on 5 different datasets, and the file is therefore structured into
# 5 different sections, each corresponding to a different dataset, and each section tests all of the 
# above for that dataset.
# --------------------------------------------------------------------------------------------------------

import linear_regression as lr
import pandas as pd
from scipy.stats import t



# --------------------------------------------------------------------------------------------------------
# DATASET #1: National Football League data (1976)
# --------------------------------------------------------------------------------------------------------


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


def test_r2_value_01():

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SST = 326.9643
    expected_SSR = 178.0923

    r2 = model.coefficient_of_determination()
    expected_r2 = expected_SSR / expected_SST

    assert abs(r2 - expected_r2) < 0.001


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


def test_confidence_interval_mean_response_01():

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    x0 = 2000

    mean_response_ll, mean_response_ul = model.mean_response_confidence_interval(x0, alpha = 0.05)

    expected_ll = 6.7658
    expected_ul = 8.7103

    assert abs(mean_response_ll - expected_ll) < 0.001
    assert abs(mean_response_ul - expected_ul) < 0.001


def test_confidence_interval_predicted_response_01():

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    x0 = 2000

    prediction_ll, prediction_ul = model.prediction_confidence_interval(x0, alpha = 0.05)

    expected_ll = 2.7242
    expected_ul = 12.7519

    assert abs(prediction_ll - expected_ll) < 0.001
    assert abs(prediction_ul - expected_ul) < 0.001


def test_model_parameters_significance_01():

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SEb1 = 0.0012596
    expected_SEb0 = 2.6962

    expected_slope = -0.007025
    expected_intercept = 21.7882

    t_statistic_b0 = model._intercept / expected_SEb0
    t_statistic_b1 = model._slope / expected_SEb1

    expected_t_b0 = expected_intercept / expected_SEb0
    expected_t_b1 = expected_slope / expected_SEb1

    assert abs(t_statistic_b0 - expected_t_b0) < 0.001
    assert abs(t_statistic_b1 - expected_t_b1) < 0.001



# --------------------------------------------------------------------------------------------------------
# DATASET #2: Solar energy project (Georgia Tech)
# --------------------------------------------------------------------------------------------------------


def test_model_parameter_estimates_02():

    data = pd.read_csv("datasets/b2.csv")

    x_data = data["x4"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_intercept = 607.1032
    expected_slope = -21.402458

    assert abs(model._intercept - expected_intercept) < 0.001
    assert abs(model._slope - expected_slope) < 0.001


def test_analysis_of_variance_02():

    data = pd.read_csv("datasets/b2.csv")

    x_data = data["x4"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SST = 14681.928
    expected_SSR = 10578.684
    expected_SSRes = 4103.243
    expected_MSRes = 151.972
    expected_Sxx = 23.0943

    assert abs(model.SST - expected_SST) < 0.001
    assert abs(model.SSR - expected_SSR) < 0.001
    assert abs(model.SSRes - expected_SSRes) < 0.001
    assert abs(model.MSRes - expected_MSRes) < 0.001
    assert abs(model.Sxx - expected_Sxx) < 0.001

def test_r2_value():

    data = pd.read_csv("datasets/b2.csv")

    x_data = data["x4"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SST = 14681.928
    expected_SSR = 10578.684

    expected_r2 = expected_SSR / expected_SST

    r2 = model.coefficient_of_determination()

    assert abs(r2 - expected_r2) < 0.001


def test_confidence_interval_slope_02():

    data = pd.read_csv("datasets/b2.csv")
    x_data = data["x4"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)


    expected_SEb1 = pow(model.MSRes / model.Sxx, 0.5)
    t_threshold = t.ppf(0.995, model.n - 2)

    expected_slope = -21.402458

    expected_lower_bound = expected_slope - t_threshold * expected_SEb1
    expected_upper_bound = expected_slope + t_threshold * expected_SEb1

    lower_bound, upper_bound = model.regressor_parameter_confidence_interval(j = 1, alpha = 0.01)

    assert abs(lower_bound - expected_lower_bound) < 0.001
    assert abs(upper_bound - expected_upper_bound) < 0.001


def test_confidence_interval_intercept_02():
    pass


def test_confidence_interval_mean_response_02():

    data = pd.read_csv("datasets/b2.csv")
    x_data = data["x4"].values
    y_data = data["y"].values

    x_mean = sum(x_data) / len(x_data)

    model = lr.SimpleLinearRegression(x_data, y_data)

    x0 = 16.5
    prediction = model.predict(x0)

    t_threshold = t.ppf(0.975, model.n - 2)

    c = model.MSRes * (1/len(x_data) + pow(x0 - x_mean, 2) / model.Sxx)
    c = pow(c, 0.5)

    expected_ll = prediction - t_threshold * c
    expected_ul = prediction + t_threshold * c

    ll, ul = model.mean_response_confidence_interval(x0, alpha = 0.05)

    assert abs(ll - expected_ll) < 0.001
    assert abs(ul - expected_ul) < 0.001


def test_confidence_interval_predicted_response_02():

    data = pd.read_csv("datasets/b2.csv")
    x_data = data["x4"].values
    y_data = data["y"].values

    x_mean = sum(x_data) / len(x_data)

    model = lr.SimpleLinearRegression(x_data, y_data)


    x0 = 16.5
    prediction = model.predict(x0)

    prediction_ll, prediction_ul = model.prediction_confidence_interval(x0, alpha = 0.05)

    c = model.MSRes * (1 + 1 / len(x_data) + (x0 - x_mean) ** 2 / model.Sxx)
    c = pow(c, 0.5)

    t_threshold = t.ppf(0.975, model.n - 2)

    expected_ll = prediction - t_threshold * c
    expected_ul = prediction + t_threshold * c

    assert abs(prediction_ll - expected_ll) < 0.001
    assert abs(prediction_ul - expected_ul) < 0.001