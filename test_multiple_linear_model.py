# --------------------------------------------------------------------------------------------------------
# MULTIPLE LINEAR REGRESSION MODEL TESTING
# 
# Tests the MultipleLinearRegression class from the linear_regression module, for the following:
#     - correct calculation of model parameters using the least squares method
#     - proper analysis of variance reporting, along with the F-statistic and p-value for the model fit
#     - proper t-tests for the model parameters (H0: b_j = 0, j=0,1,...,p)
#     - calculation of R^2 score, and adjusted R^2 score
#     - CI for the mean response
#     - CI for the model parameters
#     - CI for the predicted response
# 
# The tests are performed on 2 different datasets, and the file is therefore structured into
# 2 different sections, each corresponding to a different dataset, and each section tests all of the 
# above for that dataset.
# --------------------------------------------------------------------------------------------------------

import pandas as pd
import linear_regression as lr
import numpy as np
from scipy.stats import t


# --------------------------------------------------------------------------------------------------------
# DATASET 1
# --------------------------------------------------------------------------------------------------------


def test_model_parameters_01():
    """
    Initializes the MultipleLinearRegression class with the data from dataset 1, and tests that the
    model parameters have been calculated correctly using the least squares method.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data[["x2", "x7", "x8"]].values
    y_data = data["y"].values

    model = lr.MultipleLinearRegression(x_data, y_data)

    expected_b0 = -1.80837206
    expected_b1 = 0.00359807
    expected_b2 = 0.19396021
    expected_b3 = -0.00481549

    assert abs(model._parameters[0] - expected_b0) < 1e-3
    assert abs(model._parameters[1] - expected_b1) < 1e-3
    assert abs(model._parameters[2] - expected_b2) < 1e-3
    assert abs(model._parameters[3] - expected_b3) < 1e-3


def test_analysis_of_variance():
    """
    Tests if the model correctly calculated the values for: SST, SSR, SSRes, MSRes.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data[["x2", "x7", "x8"]].values
    y_data = data["y"].values

    model = lr.MultipleLinearRegression(x_data, y_data)

    expected_SST = 326.9642857
    expected_SSR = 257.0940708
    expected_SSRes = 69.87000418
    expected_MSRes = 2.911250174

    assert abs(model.SST - expected_SST) < 1e-3
    assert abs(model.SSR - expected_SSR) < 1e-3
    assert abs(model.SSRes - expected_SSRes) < 1e-3
    assert abs(model.MSRes - expected_MSRes) < 1e-3
    

def test_parameter_significance():
    """
    Runs a t-test for significance of each model parameter (including intercept), and checks the value
    against manually calculated value.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data[["x2", "x7", "x8"]].values
    y_data = data["y"].values

    model = lr.MultipleLinearRegression(x_data, y_data)

    t0 = model._parameters[0] / np.sqrt(model.MSRes * model.C[0, 0])
    t1 = model._parameters[1] / np.sqrt(model.MSRes * model.C[1, 1])
    t2 = model._parameters[2] / np.sqrt(model.MSRes * model.C[2, 2])
    t3 = model._parameters[3] / np.sqrt(model.MSRes * model.C[3, 3])

    expected_t0 = -0.22888295653851262
    expected_t1 = 5.177090239988905
    expected_t2 = 2.198261682783285
    expected_t3 = -3.7710364517287562

    assert abs(t0 - expected_t0) < 1e-3
    assert abs(t1 - expected_t1) < 1e-3
    assert abs(t2 - expected_t2) < 1e-3
    assert abs(t3 - expected_t3) < 1e-3


def test_coefficient_of_determination():
    """
    Calculates the R^2 score and the adjusted R^2 score, and checks the values against manually
    calculated values.
    """
    
    data = pd.read_csv("datasets/b1.csv")
    x_data = data[["x2", "x7", "x8"]].values
    y_data = data["y"].values

    model = lr.MultipleLinearRegression(x_data, y_data)

    r2 = model.coefficient_of_determination_analysis()

    expected_SST = 326.9642857
    expected_SSR = 257.0940708
    expected_SSRes = 69.87000418
    expected_MSRes = 2.911250174

    r2_standard = r2["r2"]
    r2_adjusted = r2["r2_adj"]

    r2_expected = expected_SSR / expected_SST
    r2_adj_expected = 1 - (expected_SSRes / expected_SST) * (model.n - 1) / (model.n - model.p)

    assert abs(r2_standard - r2_expected) < 1e-3
    assert abs(r2_adjusted - r2_adj_expected) < 1e-3


def test_confidence_interval_mean_response():
    """
    Constructs a confidence interval for the mean response, and checks the values against manually calc. values.
    x0 = (2300, 56, 2100)
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data[["x2", "x7", "x8"]].values
    y_data = data["y"].values

    model = lr.MultipleLinearRegression(x_data, y_data)

    x0 = np.array([2300, 56, 2100])

    expected_ll = 6.436202776657977
    expected_ul = 7.996644889334663

    mean_response_ll, mean_response_ul = model.mean_response_confidence_interval(x0)

    assert abs(mean_response_ll - expected_ll) < 1e-3
    assert abs(mean_response_ul - expected_ul) < 1e-3


def test_confidence_interval_prediction():
    """
    Constructs a 95% confidence interval for the prediction at x0 = (2300, 56, 2100), and checks the values
    against manually calculated values.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data[["x2", "x7", "x8"]].values
    y_data = data["y"].values

    model = lr.MultipleLinearRegression(x_data, y_data)

    x0 = np.array([2300, 56, 2100])

    expected_ll = 3.6095233493318544
    expected_ul = 10.823324316660784

    mean_response_ll, mean_response_ul = model.prediction_confidence_interval(x0)

    assert abs(mean_response_ll - expected_ll) < 1e-3
    assert abs(mean_response_ul - expected_ul) < 1e-3


def test_confidence_intervals_model_parameters():
    """
    Constructs 95% confidence intervals for all model parameters (including intercept),
    and compares the values with manually calculated values.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data[["x2", "x7", "x8"]].values
    y_data = data["y"].values

    model = lr.MultipleLinearRegression(x_data, y_data)

    b0_ll, b0_ul = model.regressor_parameter_confidence_interval(0)
    b1_ll, b1_ul = model.regressor_parameter_confidence_interval(1)
    b2_ll, b2_ul = model.regressor_parameter_confidence_interval(2)
    b3_ll, b3_ul = model.regressor_parameter_confidence_interval(3)

    expected_MSRes = 2.911250174

    X_data = np.hstack((np.ones((x_data.shape[0], 1)), x_data))
    C = np.linalg.inv(np.matmul(X_data.T, X_data))

    SEb0 = np.sqrt(model.MSRes * C[0, 0])
    SEb1 = np.sqrt(model.MSRes * C[1, 1])
    SEb2 = np.sqrt(model.MSRes * C[2, 2])
    SEb3 = np.sqrt(model.MSRes * C[3, 3])

    t_statistic = t.ppf(0.975, model.n - model.p)

    expected_b0 = -1.80837206
    expected_b1 = 0.00359807
    expected_b2 = 0.19396021
    expected_b3 = -0.00481549

    b0_ll_expected, b0_ul_expected = expected_b0 - t_statistic * SEb0, expected_b0 + t_statistic * SEb0
    b1_ll_expected, b1_ul_expected = expected_b1 - t_statistic * SEb1, expected_b1 + t_statistic * SEb1
    b2_ll_expected, b2_ul_expected = expected_b2 - t_statistic * SEb2, expected_b2 + t_statistic * SEb2
    b3_ll_expected, b3_ul_expected = expected_b3 - t_statistic * SEb3, expected_b3 + t_statistic * SEb3

    assert abs(b0_ll - b0_ll_expected) < 1e-3
    assert abs(b0_ul - b0_ul_expected) < 1e-3
    assert abs(b1_ll - b1_ll_expected) < 1e-3
    assert abs(b1_ul - b1_ul_expected) < 1e-3
    assert abs(b2_ll - b2_ll_expected) < 1e-3
    assert abs(b2_ul - b2_ul_expected) < 1e-3
    assert abs(b3_ll - b3_ll_expected) < 1e-3
    assert abs(b3_ul - b3_ul_expected) < 1e-3

