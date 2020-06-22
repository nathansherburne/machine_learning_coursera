# Linear Regression
import pandas as pd

class GradientDescent():
    def __init__(self, input_data, alpha=0.1, seed=(0,0)):
        """
        data: training data set (x,y)
        alpha: learning rate
        """
        self.input_data = input_data
        self.data = self.normalize(input_data)
        self.m = len(input_data.index)
        self.alpha = alpha
        self.seed = seed

    def run(self):
        theta0_norm, theta1_norm = self.descend(*self.seed)
        theta0 = self.unnormalize_theta0(theta0_norm, theta1_norm, self.input_data)
        theta1 = self.unnormalize_theta1(theta1_norm, self.input_data)
        return theta0, theta1

    def descend(self, theta0, theta1):
        while(True):
            h = self.generate_h(theta0, theta1)

            deriv_wrt_theta0 = self.derivative_wrt_theta0(theta0, h)
            deriv_wrt_theta1 = self.derivative_wrt_theta1(theta1, h)

            if (abs(deriv_wrt_theta0) < 0.001 and abs(deriv_wrt_theta1) < 0.001):
                return theta0, theta1

            theta0 = self.step_theta(theta0, self.alpha, deriv_wrt_theta0)
            theta1 = self.step_theta(theta1, self.alpha, deriv_wrt_theta1)

    def generate_h(self, theta0, theta1):
        return lambda x: theta0 + theta1*x

    def derivative_wrt_theta0(self, theta0, h):
        return (1/self.m) * sum(self.data.apply(lambda x: h(x[0]) - x[1], axis=1))

    def derivative_wrt_theta1(self, theta1, h):
        return (1/self.m) * sum(self.data.apply(lambda x: (h(x[0]) - x[1]) * x[0], axis=1))

    def step_theta(self, theta, alpha, deriv):
        return theta - alpha * deriv

    def squared_error(self, h, x, y):
        return (h(x) - y)**2

    def squared_errors(self, h):
        return self.data.apply(lambda x: self.squared_error(h, x[0], x[1]), axis=1)

    def cost(self, h):
        return (1/(2*self.m)) * sum(self.squared_errors(h))

    def normalize(self, data):
        return (data-data.min())/(data.max()-data.min())

    def unnormalize_theta1(self, theta1, data):
        return theta1 * (data.max()-data.min())[1]/(data.max()-data.min())[0]

    def unnormalize_theta0(self, theta0, theta1, data):
        x = 0 * (data.max()[0] - data.min()[0]) + data.min()[0]
        y = theta0 * (data.max()[1] - data.min()[1]) + data.min()[1]
        return y - x*self.unnormalize_theta1(theta1, data)

