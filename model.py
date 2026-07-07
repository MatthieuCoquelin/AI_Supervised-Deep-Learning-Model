import numpy as np
from sklearn.metrics import log_loss

##########################################################
#                         MODEL                          #
##########################################################

def Initialisation(dimensions):
    parameters = {}
    C = len(dimensions)

    np.random.seed(1)

    for c in range(1, C):
        parameters['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parameters['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return parameters

def ForwardPropagation(X, parameters):
    activations = {'A0': X}

    C = len(parameters) // 2

    for c in range(1, C + 1):
        Z = parameters['W' + str(c)].dot(activations['A' + str(c - 1)]) + parameters['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

    return activations

def L(y, A):
    epsilon = 1e-15
    return 1/y.shape[1] * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))


def BackPropagation(activations, y, X, parameters):
    m = y.shape[1]
    C = len(parameters) // 2

    dZ = activations['A' + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
        if c > 1:
            dZ = np.dot(parameters['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

    return gradients

def Update(gradients, parameters, learning_rate):
    size = int(len(gradients) / 2) + 1

    for i in range(1, size):
        dWi = gradients['dW' + str(i)]
        dbi = gradients['db' + str(i)]
        Wi = parameters['W' + str(i)]
        bi = parameters['b' + str(i)]
        Wi -= learning_rate * dWi
        bi -= learning_rate * dbi
        parameters['W' + str(i)] = Wi
        parameters['b' + str(i)] = bi
    
    return parameters


def Predict(X, parameters):
    activations = ForwardPropagation(X, parameters)
    C = len(parameters) // 2
    A_Last = activations['A' + str(C)]
    return A_Last >= 0.5