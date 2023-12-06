import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Building Linear Regression Phase

class LinearRegression():

    def __init__(self, lr=0.001, epoch=1000):
        self.lr = lr
        self.epoch = epoch
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epoch):
            prediction = np.dot(X, self.weights) + self.bias

            dw = (-2/n_samples) * np.dot(X.T, (y - prediction))
            db = (-2/n_samples) * np.sum(y - prediction)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias - self.lr*db
    
    def predict(self, X):
        prediction = np.dot(X, self.weights) + self.bias
        return prediction
    
def Mean_Squared_Error(y_pred, y_test):
    return np.mean((y_pred-y_test)**2)

# Training Phase

X , y = datasets.make_regression(n_samples=100, n_features=1, noise=30, random_state=123)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)

lr = LinearRegression(lr=0.1)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
print('y_pred: ',y_pred)

print('MSE: ', Mean_Squared_Error(y_pred, y_test))


fig = plt.figure(figsize=(12,6))
plt.scatter(X[:, 0], y, color="black", marker="o")
# plt.show()

# plot regression line
reg_line = lr.predict(X)
plt.plot(X, reg_line, color="red")
plt.show()