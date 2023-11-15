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
# The tests are performed on 2 different datasets, and the file is therefore structured into
# 2 different sections, each corresponding to a different dataset, and each section tests all of the 
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
    Tests the accuracy of the model parameter estimates (b0, b1) for a sample dataset. Compares the estimates
    with manually calculated values.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_slope = -0.007025
    expected_intercept = 21.7882

    assert abs(model._slope - expected_slope) < 0.001
    assert abs(model._intercept - expected_intercept) < 0.001


def test_model_parameters_significance_01():
    """
    Calculates the t-statistics for tests of significance for the model parameters, and compares them with the manually
    calculated values.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    t_stat_slope = model._slope / model.SEb1
    t_stat_intercept = model._intercept / model.SEb0

    t_stat_intercept_expected = 8.080996253
    t_stat_slope_expected = -5.577027563

    assert abs(t_stat_slope - t_stat_slope_expected) < 0.001
    assert abs(t_stat_intercept - t_stat_intercept_expected) < 0.001


def test_analysis_of_variance_01():
    """
    Tests the analysis of variance table for sample dataset, by comparing the SST, SSR, SSRes, MSRes, and Sxx
    values calculated by the model with manually checked values (labelled as expected).
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SST = 326.9642857
    expected_SSR = 178.0923135
    expected_SSRes = 148.8719722
    expected_MSRes = 5.725845085
    expected_Sxx = 3608611.429

    assert abs(model.SST - expected_SST) < 0.001
    assert abs(model.SSR - expected_SSR) < 0.001
    assert abs(model.SSRes - expected_SSRes) < 0.001
    assert abs(model.MSRes - expected_MSRes) < 0.001
    assert abs(model.Sxx - expected_Sxx) < 0.001


def test_r2_value_01():
    """
    Compares the R2 value of the model with the manually calculated value.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SST = 326.9642857
    expected_SSR = 178.0923135
    expected_r2 = expected_SSR / expected_SST

    r2 = model.coefficient_of_determination()

    assert abs(r2 - expected_r2) < 0.001


def test_confidence_interval_slope_01():
    """
    Calculates the 95% confidence interval for the slope parameter, and compares it with the manually calculated
    value.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SEb1 = 0.001259649553
    model_SEb1 = model.SEb1

    boundary_t = t.ppf(0.975, model.n - 2)
    expected_slope = -0.007025100276

    expected_lower_bound = expected_slope - boundary_t * expected_SEb1
    expected_upper_bound = expected_slope + boundary_t * expected_SEb1

    lower_bound, upper_bound = model.regressor_parameter_confidence_interval(j = 1, alpha = 0.05)

    assert abs(model_SEb1 - expected_SEb1) < 0.001
    assert abs(lower_bound - expected_lower_bound) < 0.001
    assert abs(upper_bound - expected_upper_bound) < 0.001


def test_confidence_interval_intercept_01():
    """
    Uses the model to calculate the 95% confidence interval for the intercept parameter, and compares it with the manually
    calculated value.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    n = len(x_data)
    mean_x = sum(x_data) / n

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SEb0 = 2.696233261
    model_SEb0 = model.SEb0

    expected_intercept = 21.78825088
    boundary_t_statistic = t.ppf(0.975, model.n - 2)

    expected_lower_bound = expected_intercept - boundary_t_statistic * expected_SEb0
    expected_upper_bound = expected_intercept + boundary_t_statistic * expected_SEb0

    lower_bound, upper_bound = model.regressor_parameter_confidence_interval(j = 0, alpha = 0.05)

    assert abs(model_SEb0 - expected_SEb0) < 0.001
    assert abs(lower_bound - expected_lower_bound) < 0.001
    assert abs(upper_bound - expected_upper_bound) < 0.001


def test_confidence_interval_mean_response_01():
    """
    Calculates the 95% confidence interval for the mean response at x0 = 2000, and compares it with the manually
    calculated value.
    """


    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    x0 = 2000

    mean_response_ll, mean_response_ul = model.mean_response_confidence_interval(x0, alpha = 0.05)

    expected_ll = 6.765753088
    expected_ul = 8.710347573

    assert abs(mean_response_ll - expected_ll) < 0.001
    assert abs(mean_response_ul - expected_ul) < 0.001


def test_confidence_interval_predicted_response_01():
    """
    Calculates the 955 confidence interval for the predicted response at x0 = 2000, and compares it with the manually
    calculated value.
    """

    data = pd.read_csv("datasets/b1.csv")
    x_data = data["x8"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    x0 = 2000

    prediction_ll, prediction_ul = model.prediction_confidence_interval(x0, alpha = 0.05)

    expected_ll = 2.724248393
    expected_ul = 12.75185227

    assert abs(prediction_ll - expected_ll) < 0.001
    assert abs(prediction_ul - expected_ul) < 0.001






# --------------------------------------------------------------------------------------------------------
# DATASET #2: Solar energy project (Georgia Tech)
# --------------------------------------------------------------------------------------------------------


def test_model_parameter_estimates_02():
    """
    Builds the model on the second dataset, and compares the estimated model parameters with the manually calculated
    values.
    """

    data = pd.read_csv("datasets/b2.csv")

    x_data = data["x4"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_intercept = 607.1032654
    expected_slope = -21.40245829

    assert abs(model._intercept - expected_intercept) < 0.001
    assert abs(model._slope - expected_slope) < 0.001


def test_analysis_of_variance_02():
    """
    Checks that model has correctly calculated the following values: SST, SSR, SSRes, MSRes, Sxx.
    Values are tested against manually calculated values.
    """

    data = pd.read_csv("datasets/b2.csv")

    x_data = data["x4"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SST = 14681.92828
    expected_SSR = 10578.68457
    expected_SSRes = 4103.243703
    expected_MSRes = 151.971989
    expected_Sxx = 23.09427586

    assert abs(model.SST - expected_SST) < 0.001
    assert abs(model.SSR - expected_SSR) < 0.001
    assert abs(model.SSRes - expected_SSRes) < 0.001
    assert abs(model.MSRes - expected_MSRes) < 0.001
    assert abs(model.Sxx - expected_Sxx) < 0.001

def test_r2_value_02():
    """
    Comapres the R2 value from the model with manually calculated value.
    """


    data = pd.read_csv("datasets/b2.csv")

    x_data = data["x4"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_SST = 14681.92828
    expected_SSR = 10578.68457

    expected_r2 = expected_SSR / expected_SST

    r2 = model.coefficient_of_determination()

    assert abs(r2 - expected_r2) < 0.001


def test_confidence_interval_slope_02():
    """
    Builds a 99% confidence interval for the slope, and checks with manually calculated values.
    """

    data = pd.read_csv("datasets/b2.csv")
    x_data = data["x4"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_lower_bound = -28.50995116
    expected_upper_bound = -14.29496542

    lower_bound, upper_bound = model.regressor_parameter_confidence_interval(j = 1, alpha = 0.01)

    assert abs(lower_bound - expected_lower_bound) < 0.001
    assert abs(upper_bound - expected_upper_bound) < 0.001


def test_confidence_interval_intercept_02():
    """
    Builds a 99% confidence interval for the intercept, and checks with manually calculated values.
    """
    
    data = pd.read_csv("datasets/b2.csv")
    x_data = data["x4"].values
    y_data = data["y"].values

    x_mean = sum(x_data) / len(x_data)

    model = lr.SimpleLinearRegression(x_data, y_data)

    expected_lower_bound = 488.2241076
    expected_upper_bound = 725.9824233

    lower_bound, upper_bound = model.regressor_parameter_confidence_interval(j = 0, alpha = 0.01)

    assert abs(lower_bound - expected_lower_bound) < 0.001
    assert abs(upper_bound - expected_upper_bound) < 0.001


def test_confidence_interval_mean_response_02():

    data = pd.read_csv("datasets/b2.csv")
    x_data = data["x4"].values
    y_data = data["y"].values

    x_mean = sum(x_data) / len(x_data)

    model = lr.SimpleLinearRegression(x_data, y_data)

    x0 = 16.5

    expected_ll = 249.146752
    expected_ul = 258.7786553

    ll, ul = model.mean_response_confidence_interval(x0, alpha = 0.05)

    assert abs(ll - expected_ll) < 0.001
    assert abs(ul - expected_ul) < 0.001


def test_confidence_interval_predicted_response_02():

    data = pd.read_csv("datasets/b2.csv")
    x_data = data["x4"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)
    x0 = 16.5

    prediction_ll, prediction_ul = model.prediction_confidence_interval(x0, alpha = 0.05)

    expected_ll = 228.2139804
    expected_ul = 279.7114269

    assert abs(prediction_ll - expected_ll) < 0.001
    assert abs(prediction_ul - expected_ul) < 0.001


def test_model_parameters_significance_02():
    """
    Calculates the t-statistics for tests of significance for the model parameters, and compares them with the manually
    calculated values.
    """

    data = pd.read_csv("datasets/b2.csv")
    x_data = data["x4"].values
    y_data = data["y"].values

    model = lr.SimpleLinearRegression(x_data, y_data)

    t_stat_slope = model._slope / model.SEb1
    t_stat_intercept = model._intercept / model.SEb0

    t_stat_intercept_expected = 14.14958434
    t_stat_slope_expected = -8.343227005

    assert abs(t_stat_slope - t_stat_slope_expected) < 0.001
    assert abs(t_stat_intercept - t_stat_intercept_expected) < 0.001