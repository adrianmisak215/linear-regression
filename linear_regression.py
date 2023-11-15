import numpy as np
from scipy.stats import t, f
import matplotlib.pyplot as plt

class SimpleLinearRegression:

    def __init__(self, x, y):
        """
        Creates a new instance of the SimpleLinearRegression model, for the independent
        variable x, and the dependent variable y. Immediately trains the model to fit the data.
        """

        self.x = x
        self.y = y
        self.n = len(x)

        self.Sxx = np.sum((x - np.mean(x))**2)
        self.Sxy = np.dot(x, y) - self.n * np.mean(x) * np.mean(y)

        self._slope = self.Sxy / self.Sxx
        self._intercept = np.mean(self.y) - self._slope * np.mean(self.x)

        self.SST = np.sum((self.y - np.mean(self.y))**2)
        self.SSRes = self.SST - self._slope * self.Sxy
        self.SSR = self.SST - self.SSRes
        self.MSRes = self.SSRes / (self.n - 2)
        self.SEb1 = np.sqrt(self.MSRes / self.Sxx)
        self.SEb0 = np.sqrt(self.MSRes * (1 / self.n + np.mean(self.x)**2 / self.Sxx))


    def __repr__(self) -> str:

        model_repr = "Simple linear regression model trained on {} observations.".format(self.n)
        model_repr += "\nThe estimated parameters are: \n"
        model_repr += "   - b0: {:.4f}\n".format(self._intercept)
        model_repr += "   - b1: {:.4f}\n".format(self._slope)

        return model_repr
    

    def fit(self) -> dict:

        return {
            "b0": self._intercept,
            "b1": self._slope
        }
    
    def predict(self, x0: float) -> float:
        """
        Returns the predicted value for y at x0.
        """

        return self._intercept + self._slope * x0
    

    def regression_significance(self, alpha: float = 0.05):
        """
        t-test to check if the hypothesis H0: b1 = 0 is true or not
        """
        
        t_stat = self._slope / self.SEb1
        # find the 0.025 and 0.975 quantiles of the t-distribution with n-2 degrees of freedom

        t025 = t.ppf(alpha / 2, self.n - 2)
        t975 = t.ppf(1 - alpha / 2, self.n - 2)
        print("The value of the t-statistic with {} degrees of freedom is: {:.2f}".format(self.n - 2, t_stat))
        print("The 0.025 quantile of the t-distribution with {} degrees of freedom is: {:.2f}".format(self.n - 2, t025))    
        print("The 0.975 quantile of the t-distribution with {} degrees of freedom is: {:.2f}".format(self.n - 2, t975))

        if t_stat < t025 or t_stat > t975:
            print("Reject H0: b1 = 0")
        else:
            print("Fail to reject H0: b1 = 0")


    # -------------------------------------------------------------
    # Confidence intervals
    # -------------------------------------------------------------

    def mean_response_confidence_interval(self, x: float, alpha: float = 0.05) -> tuple:
        """
        Constructs the 100(1-alpha)% confidence interval for the mean response at x.
        Returns the lower and upper bounds of the confidence interval as a tuple.
        """

        SEy = np.sqrt(self.MSRes * (1 / self.n + (x - np.mean(self.x))**2 / self.Sxx))
        t_value = t.ppf(alpha/2, self.n - 2)

        lower_bound = self._intercept + self._slope * x - abs(t_value) * SEy
        upper_bound = self._intercept + self._slope * x + abs(t_value) * SEy

        return (lower_bound, upper_bound)
    

    def prediction_confidence_interval(self, x: float, alpha: float = 0.05) -> tuple:
        """
        Returns the 100(1-alpha)% prediction interval for the response at x.
        """

        SEprediction = np.sqrt(self.MSRes * (1 + 1 / self.n + (x - np.mean(self.x))**2 / self.Sxx))
        t_value = t.ppf(alpha/2, self.n - 2)

        lower_bound = self._intercept + self._slope * x - abs(t_value) * SEprediction
        upper_bound = self._intercept + self._slope * x + abs(t_value) * SEprediction

        return (lower_bound, upper_bound)
    

    def regressor_parameter_confidence_interval(self, j: int, alpha: float) -> tuple:
        """
        Constructs the 100(1-alpha)% confidence interval for b0 (intercept) or b1 (slope), determined by index j.
        Returns the lower and upper bounds of the confidence interval as a tuple.
        """

        ci_estimators = {
            0: self._estimate_b0_confidence_interval,
            1: self._estimate_b1_confidence_interval
        }

        return ci_estimators[j](alpha)

    
    def _estimate_b1_confidence_interval(self, alpha: float = 0.05) -> tuple:
        """
        Creates the 100(1-alpha)% confidence interval for the intercept parameter b1.
        """

        t_value = t.ppf(alpha/2, self.n - 2)

        lower_bound = self._slope - abs(t_value) * self.SEb1
        upper_bound = self._slope + abs(t_value) * self.SEb1

        return (lower_bound, upper_bound)
    

    def _estimate_b0_confidence_interval(self, alpha: float = 0.05) -> tuple:
        """
        Creates the 100(1-alpha)% confidence interval for the intercept parameter b0.
        """

        
        t_value = abs(t.ppf(alpha/2, self.n - 2))

        lower_bound = self._intercept - t_value * self.SEb0
        upper_bound = self._intercept + t_value * self.SEb0

        return (lower_bound, upper_bound)

    # -------------------------------------------------------------
    # Goodness of fit
    # -------------------------------------------------------------

    def analysis_of_variance(self):

        print("Analysis of variance identity breakdown:")
        print("  - total sum of squares (SST): {:.2f},".format(self.SST))
        print("  - regression sum of squares (SSR): {:.2f},".format(self.SSR))
        print("  - residual sum of squares (SSRes): {:.2f},".format(self.SSRes), end="\n\n")

        f_distribution_boundary = f.ppf(1 - 0.05, 1, self.n - 2)

        print("Assuming that the errors are normally distributed, we can use the F-test to check if the model is significant.")
        print("The F-statistic is: {:.2f}".format(self.SSR / self.MSRes))
        print("The 0.95 quantile of the F-distribution with 1 and {} degrees of freedom is: {:.2f}".format(self.n - 2, f_distribution_boundary))

        test_passes = self.SSR / self.MSRes > f_distribution_boundary

        if test_passes:
            print("The model is significant; the hypothesis H0: b1 = 0 is rejected.", end="\n\n")
        else:
            print("The model is not significant; the hypothesis H0: b1 = 0 is not rejected.", end="\n\n")

        print("The figure below shows the PDF for the F(1, n-2) distribution, and the value of the F-statistic, along with the 95% quantile.")
        plt.figure(figsize=(10, 6))
        x = np.linspace(0, 10, 100)
        plt.plot(x, f.pdf(x, 1, self.n - 2), color="black", label="F(1, n-2)")
        plt.axvline(x=self.SSR / self.MSRes, color="orange", label="F-statistic")
        plt.axvline(x=f_distribution_boundary, color="red", label="0.95 quantile")
        plt.legend()
        plt.show()


    def coefficient_of_determination(self) -> float:
        """
        Calculates the coefficient of determination R^2.
        """

        return self.SSR / self.SST
    

    def individual_parameter_significance(self, alpha: float = 0.05):
        """
        Uses the t-test to check hypothesis H0: b1 = 0, and hypothesis that b0 = 0. 
        """

        SEb0 = np.sqrt(self.MSRes * (1 / self.n + np.mean(self.x)**2 / self.Sxx))
        SEb1 = np.sqrt(self.MSRes / self.Sxx)

        threshold = t.ppf(alpha / 2, self.n - 2)

        t_statistc_b0 = self._intercept / SEb0
        t_statistc_b1 = self._slope / SEb1

        hypothesis_slope = abs(t_statistc_b1) > abs(threshold)
        hypothesis_intercept = abs(t_statistc_b0) > abs(threshold)

        print("#------------------------------------------------------------------")
        print("Testing hypothesis H0: b0 = 0...")
        print("The value of the t-statistic with {} degrees of freedom is: {:.2f}".format(self.n - 2, t_statistc_b0))
        print("The 0.025 quantile of the t-distribution with {} degrees of freedom is: {:.2f}".format(self.n - 2, threshold))
        print("The 0.975 quantile of the t-distribution with {} degrees of freedom is: {:.2f}".format(self.n - 2, -threshold))
        print("The H0 hypothesis b0 = 0 is: {}".format("REJECTED" if hypothesis_intercept else "NOT REJECTED"), end="\n\n")

        print("#------------------------------------------------------------------")
        print("Testing hypothesis H0: b1 = 0...")
        print("The value of the t-statistic with {} degrees of freedom is: {:.2f}".format(self.n - 2, t_statistc_b1))
        print("The 0.025 quantile of the t-distribution with {} degrees of freedom is: {:.2f}".format(self.n - 2, threshold))
        print("The 0.975 quantile of the t-distribution with {} degrees of freedom is: {:.2f}".format(self.n - 2, -threshold))
        print("The H0 hypothesis b1 = 0 is: {}".format("REJECTED" if hypothesis_slope else "NOT REJECTED"))

    
    # -------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------

    def plot_linear_model_line(self, x_axis_name: str = "Independent variable", y_axis_name: str = "Dependent variable", title: str="Linear model"):
        """
        Plots the x, and y data in a scatter plot, and adds the line of the linear model, to visualize
        the goodness of fit.
        """

        plt.figure(figsize=(10, 6))
        plt.scatter(self.x, self.y, color="black", label="Data")

        parameters = self.fit()

        x_range = np.linspace(np.min(self.x), np.max(self.x), 100)
        y_range = parameters["b0"] + parameters["b1"] * x_range

        plt.plot(x_range, y_range, color="orange", label="Linear model")
        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)
        plt.title(title)
        plt.legend()
        plt.show()

    
    def plot_mean_response_confidence_area(self, alpha: float = 0.05, x_axis_name: str = "Independent variable", y_axis_name: str = "Dependent variable", title: str="Mean response confidence interval"):
        """
        Plots the x, and y data in a scatter plot, and adds the confidence interval for the given alpha, over the entire range of x.
        """

        min_x = np.min(self.x)
        max_x = np.max(self.x)
        x_range = np.linspace(min_x, max_x, 100)

        lower_limits = []
        upper_limits = []

        for x in x_range:
            ll, ul = self.mean_response_confidence_interval(x, alpha)
            lower_limits.append(ll)
            upper_limits.append(ul)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.x, self.y, color="black", label="Data")

        plt.plot(x_range, lower_limits, linestyle="--", color="green", label="Mean response interval boundary (alpha = {})".format(alpha))
        plt.plot(x_range, upper_limits, linestyle="--", color="green")

        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)
        plt.title(title)
        plt.legend()
        plt.show()


    def plot_prediction_confidence_area(self, alpha: float = 0.05, x_axis_name: str = "Independent variable", y_axis_name = "Dependent variable", title: str = "Prediction intervals"):
        """
        Plots the x, and y data in a scatter plot, and adds the prediction interval (for given alpha), over the entire range of x.
        """
        
        min_x = np.min(self.x)
        max_x = np.max(self.x)
        x_range = np.linspace(min_x, max_x, 100)

        lower_limits = []
        upper_limits = []

        for x in x_range:
            ll, ul = self.prediction_confidence_interval(x, alpha)
            lower_limits.append(ll)
            upper_limits.append(ul)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.x, self.y, color="black", label="Data")

        plt.plot(x_range, lower_limits, linestyle="--", color="purple", label="Prediction interval boundary (alpha = {})".format(alpha))
        plt.plot(x_range, upper_limits, linestyle="--", color="purple")

        plt.xlabel(x_axis_name)
        plt.ylabel(y_axis_name)
        plt.title(title)
        plt.legend()
        plt.show()





class MultipleLinearRegression:

    def __init__(self, X: np.array, y: np.array):
        """
        Initializes the MultipleLinearRegression class, and automatically trains to fit the data.
        """

        # add a column of ones as the first column of X
        self.X = np.hstack((np.ones((X.shape[0], 1)), X))
        self.y = y
        self.n = len(y)
        self.p = self.X.shape[1]

        self._parameters = self._fit()

        self.C = np.linalg.inv(np.matmul(self.X.T, self.X))

        self.SSRes = np.sum((self.y - np.matmul(self.X, self._parameters))**2)
        self.MSRes = self.SSRes / (self.n - self.p)

        self.SST = np.dot(self.y, self.y) - (np.sum(self.y)**2) / self.n
        self.SSR = np.dot(np.matmul(self.X, self._parameters), self.y) - (np.sum(self.y)**2) / self.n
        self.SSRes = self.SST - self.SSR
        self.MSRes = self.SSRes / (self.n - self.p)


    def _fit(self) -> np.array:
        """
        Estimates the model parameters, and returns them as a numpy array.
        """

        temp1 = np.linalg.inv(np.matmul(self.X.T, self.X))
        temp2 = np.matmul(self.X.T, self.y)
        

        estimated_model_parameters = np.matmul(temp1, temp2)
        

        return estimated_model_parameters
    
    def _standard_errors(self) -> np.array:
        """
        Calculates the standard errors for the estimated parameters, and returns them as a numpy array.
        """

        return np.sqrt(self.MSRes * np.diag(self.C))
    



    def predict(self, x: np.array) -> float:
        """
        Returns the predicted value for y at x.
        Appends a 1 to the beginning of x, to account for the intercept.
        """

        x_with_intercept = np.array([1] + list(x))

        return np.dot(x_with_intercept, self._parameters)
    

    def __repr__(self) -> str:
        """
        Prints the dimensions of the input data for the model, and the estimated parameters.
        """

        model_repr = "Multiple linear regression model trained on {} observations with {} parameters.".format(self.n, self.p)
        model_repr += "\nThe estimated parameters are: \n"
        for i in range(len(self._parameters)):
            model_repr += "   - b{}: {:.4f}\n".format(i, self._parameters[i])
    
        return model_repr

    # -------------------------------------------------------------
    # Goodness of fit
    # -------------------------------------------------------------

    def analysis_of_variance(self) -> None:
        """
        Performs the analysis of variance on input data, and tests the H0 hypothesis: b1 = b2 = ... = bp = 0.
        """


        print("Analysis of variance identity breakdown:")
        print("  - total sum of squares (SST): {:.2f},".format(self.SST))
        print("  - regression sum of squares (SSR): {:.2f},".format(self.SSR))
        print("  - residual sum of squares (SSRes): {:.2f},".format(self.SSRes), end="\n\n")

        f_statistic = (self.SSR / (len(self._parameters) - 1)) / self.MSRes

        f_distribution_boundary = f.ppf(1 - 0.05, self.p - 1, self.n - self.p)

        print("Assuming that the errors are normally distributed, we can use the F-test to check if the model is significant.")
        print("The H0 hypothesis being tested is: b1 = b2 = ... = bp = 0.")
        print("The F-statistic is: {:.2f}".format(f_statistic))
        print("The 0.95 quantile of the F-distribution with {} and {} degrees of freedom is: {:.2f}".format(self.p - 1, self.n - self.p, f_distribution_boundary))

        test_passes = f_statistic > f_distribution_boundary

        if test_passes:
            print("The model is significant; the hypothesis H0: b1 = b2 = ... = bp = 0 is rejected.", end="\n\n")
        else:
            print("The model is not significant; the hypothesis H0: b1 = b2 = ... = bp = 0 is not rejected.", end="\n\n")

        print("The figure below shows the PDF for the F(p-1, n-p) distribution, and the value of the F-statistic, along with the 95% quantile.")

        plt.figure(figsize=(10, 6))
        x = np.linspace(0, 10, 100)
        plt.plot(x, f.pdf(x, self.p - 1, self.n - self.p), color="black", label="F(p-1, n-p)")

        plt.axvline(x=f_statistic, color="orange", label="F-statistic")
        plt.axvline(x=f_distribution_boundary, color="red", label="0.95 quantile")

        plt.legend()
        plt.show()

    
    def individual_parameter_significance(self, alpha: float = 0.05):
        """
        Performs a t-test on the coefficient of the model at the given index.
        """

        SE_parameters = np.sqrt([self.MSRes * self.C[i][i] for i in range(len(self._parameters))])
        t_values = [self._parameters[i] / SE_parameters[i] for i in range(len(self._parameters))]

        t_value_boundary = abs(t.ppf(alpha / 2, self.n - self.p))
        t_test_results = [abs(t_values[i]) > t_value_boundary for i in range(len(t_values))]

        for ind in range(len(self._parameters)):
            print("#------------------------------------------------------------------")
            print("Running the t-test for significance of parameter b{}...".format(ind))
            print("   - value of t statistic: ", t_values[ind])
            print("   - value of t statistic boundary: ", t_value_boundary)
            result = "REJECTED" if t_test_results[ind] else "NOT REJECTED"
            print("   - H0 hypothesis b{} = 0 is: {}".format(ind, result), end="\n\n")

     

    def coefficient_of_determination_analysis(self, verbose: bool = False):
        """
        Calculates both the R^2 and the adjusted value, and returns as dict. If print_report = True,
        then also prints the values.
        """

        r2 = self._coefficient_of_determination_standard()
        r2_adj = self._coefficient_of_determination_adjusted()

        if verbose:
            print("The coefficient of determination R^2 is: {:.2f}".format(r2))
            print("The adjusted coefficient of determination R^2 is: {:.2f}".format(r2_adj))


        return {
            "r2": r2,
            "r2_adj": r2_adj
        }
    

    def _coefficient_of_determination_standard(self) -> float:
        """
        Calculates the coefficient of determination R^2.
        """

        return self.SSR / self.SST
    

    def _coefficient_of_determination_adjusted(self) -> float:
        """
        Calculates the adjusted coefficient of determination R^2.
        """

        temp = (self.SSRes) / (self.n - self.p)
        temp /= (self.SST) / (self.n - 1)

        return 1 - temp
    

    # -------------------------------------------------------------
    # Confidence intervals
    # -------------------------------------------------------------

    def mean_response_confidence_interval(self, x: np.array, alpha: float = 0.05) -> tuple:
        """
        Constructs the 100(1-alpha)% confidence interval for the mean response at x.
        """

        # calculate mean response as the dot product of x and the estimated parameters
        mean_response = self.predict(x)
        t_boundary = abs(t.ppf(alpha / 2, self.n - self.p))

        x = np.array([1] + list(x))

        temp = self.MSRes * np.dot(np.matmul(x, self.C), x.T)

        lower_bound = mean_response - t_boundary * np.sqrt(temp)
        upper_bound = mean_response + t_boundary * np.sqrt(temp)

        return (lower_bound, upper_bound)


    def regressor_parameter_confidence_interval(self, j: int, alpha: float = 0.05) -> tuple:
        """
        Constructs the 100(1-alpha)% confidence interval for the parameter = parameter, which can be one of the 
        following values: b0, b1, sigma2. 
        Returns the lower and upper bounds of the confidence interval as a tuple.
        """

        if j < 0 or j > self.p:
            raise ValueError("Invalid parameter value. Valid values are: b0, b1, sigma2.")
        
        SE_coefficients = np.array([np.sqrt(self.MSRes * self.C[i, i]) for i in range(self.p)])[j]

        t_value = t.ppf(alpha/2, self.n - self.p)

        lower_bound = self._parameters[j] - abs(t_value) * SE_coefficients
        upper_bound = self._parameters[j] + abs(t_value) * SE_coefficients

        return (lower_bound, upper_bound)
    

    def prediction_confidence_interval(self, x: np.array, alpha: float = 0.05) -> tuple:
        """
        Returns the 100(1-alpha)% prediction interval for the response at x.
        """

        # calculate mean response as the dot product of x and the estimated parameters
        x = np.array([1] + list(x))
        mean_response = np.dot(x, self._parameters)
        t_boundary = abs(t.ppf(alpha / 2, self.n - self.p))

        temp = self.MSRes * (1 + np.dot(np.matmul(x, self.C), x.T))

        lower_bound = mean_response - t_boundary * np.sqrt(temp)
        upper_bound = mean_response + t_boundary * np.sqrt(temp)

        return (lower_bound, upper_bound)

        

    

if __name__ == "__main__":

    delivery_times = [16.68, 11.50, 12.03, 14.88, 13.75, 18.11, 8.0, 17.83, 79.24, 21.5, 40.33, 21.0, 13.50, 19.75, 24.0, 29.0, 15.35, 19.0, 9.5, 35.1, 17.9, 52.32, 18.75, 19.83, 10.75]
    number_cases = [7, 3, 3, 4, 6, 7, 2, 7, 30, 5, 16, 10, 4, 6, 9, 10, 6, 7, 3, 17, 10, 26, 9, 8, 4]
    distances = [560, 220, 340, 80, 150, 330, 110, 210, 1460, 605, 688, 215, 255, 462, 448, 776, 200, 132, 36, 770, 140, 810, 450, 635, 150]

    X = np.array([number_cases, distances]).T
    y = np.array(delivery_times)

    model = MultipleLinearRegression(X, y)
    print(model._standard_errors())