import numpy as np
from sklearn import datasets, linear_model


class LinearRegressor:
    def __init__(self, number_of_coefs):
        self.w = np.ones(number_of_coefs)
        self.b = 1.0

    def predict(self, x):
        return np.dot(x, self.w) + self.b

    def MSE(self, X, y):
        sum_of_squares = 0
        for i in range(len(X)):
            sum_of_squares += (self.predict(X[i]) - y[i])**2
        return 1/len(X) * sum_of_squares

    def calc_gradient(self, X, y):
        b = 0
        w = np.zeros(len(X[0]))
        for i in range(len(X)):
            b += (self.predict(X[i]) - y[i])
            # print(len(self.w), len(X[i]))
            w += (self.predict(X[i]) - y[i]) * (X[i])
            # for j in range(len(w)):
            #     w[j] += (self.predict(X[i]) - y[i]) * (X[i][j])
        b /= len(X)
        w /= len(X)
        return w, b

    def fit(self, X, y, epsilon, learning_rate):
        mse = self.MSE(X, y)
        while mse > epsilon:
            mse = self.MSE(X, y)
            gradient = self.calc_gradient(X, y)
            self.b = self.b - learning_rate * gradient[1]
            self.w = self.w - learning_rate * gradient[0]
        # print(mse, self.b, self.w)


X, y = datasets.make_regression(n_samples=100, n_features=2, noise=8, shuffle=True)
regressor = LinearRegressor(2)
regressor.fit(X, y, 100, 0.5)
regr = linear_model.LinearRegression()
regr.fit(X, y)
print(regressor.b, regressor.w)
print(regr.intercept_, regr.coef_)
