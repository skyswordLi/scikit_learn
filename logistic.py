from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel
import math


data = loadtxt('data1.txt', delimiter=' ')
X = data[:, 0:2]
y = data[:, 2]
pos = where(y == 1)
neg = where(y == 0)
scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
xlabel('Feature1/Exam 1 score')
ylabel('Feature2/Exam 2 score')
legend(['Fail', 'Pass'])
show()


def sigmoid(X):
    '''Compute sigmoid function '''
    den = 1.0 + math.e ** (-1.0 * X)
    gz = 1.0 / den

    return gz


def compute_cost(theta, X, y):
    '''computes cost given predicted and actual values'''
    m = X.shape[0]
    theta = reshape(theta, (len(theta), 1))
    J = (1. / m) * (-transpose(y).dot(math.log(sigmoid(X.dot(theta)))) -
        transpose(1 - y).dot(math.log(1 - sigmoid(X.dot(theta)))))
    grad = transpose((1. / m) * transpose(sigmoid(X.dot(theta)) - y).dot(X))

    return J[0][0]


def compute_grad(theta, X, y):
    '''compute gradient'''
    theta.shape = (1, 3)
    grad = zeros(3)
    h = sigmoid(X.dot(theta.T))
    delta = h - y
    l = grad.size
    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = -1 * (1.0 / m) * sumdelta
        theta.shape = (3,)

    return grad
