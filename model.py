import numpy as np
import tensorflow as tf
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


##########################################################
#                     ADAM OPTIMIZER                     #
##########################################################

def Optimized_Initialisation(dimensions):
    parameters = {}
    adamParameters = {}
    C = len(dimensions)

    np.random.seed(1)

    for c in range(1, C):
        parameters['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        parameters['b' + str(c)] = np.random.randn(dimensions[c], 1)
        adamParameters['V_w' + str(c)] = np.zeros((dimensions[c], dimensions[c - 1]))
        adamParameters['M_w' + str(c)] = np.zeros((dimensions[c], dimensions[c - 1]))
        adamParameters['V_b' + str(c)] = np.zeros((dimensions[c], 1))
        adamParameters['M_b' + str(c)] = np.zeros((dimensions[c], 1))
   
    return parameters, adamParameters


def Optimized_Update(gradients, parameters, learning_rate, adamParameters, t):
    size = int(len(gradients) / 2) + 1
    beta1 = 0.9
    beta2 = 0.999
    epsilon=1e-8

    for i in range(1, size):
        dWi = gradients['dW' + str(i)]
        dbi = gradients['db' + str(i)]
        Wi = parameters['W' + str(i)]
        bi = parameters['b' + str(i)]
        V_Wi = adamParameters['V_w' + str(i)]
        M_Wi = adamParameters['M_w' + str(i)]
        V_bi = adamParameters['V_b' + str(i)]
        M_bi = adamParameters['M_b' + str(i)]
        
        M_Wi = beta1 * M_Wi + (1 - beta1) * dWi
        V_Wi = beta2 * V_Wi + (1 - beta2) * np.square(dWi)
        M_W_hat = M_Wi / (1 - beta1 ** t + 1)
        V_W_hat = V_Wi / (1 - beta2 ** t + 1)
        Wi -= learning_rate * M_W_hat / (np.sqrt(V_W_hat) + epsilon)
        
        M_bi = beta1 * M_bi + (1 - beta1) * dbi
        V_bi = beta2 * V_bi + (1 - beta2) * np.square(dbi)
        M_b_hat = M_bi / (1 - beta1 ** t + 1)
        V_b_hat = V_bi / (1 - beta2 ** t + 1)
        bi -= learning_rate * M_b_hat / (np.sqrt(V_b_hat) + epsilon)

        parameters['W' + str(i)] = Wi
        parameters['b' + str(i)] = bi

    return parameters


##########################################################
#                    ADAM_W OPTIMIZER                    #
##########################################################


def W_Optimized_Update(gradients, parameters, learning_rate, adamParameters, t):
    size = int(len(gradients) / 2) + 1
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.01
    epsilon=1e-8

    for i in range(1, size):
        dWi = gradients['dW' + str(i)]
        dbi = gradients['db' + str(i)]
        Wi = parameters['W' + str(i)]
        bi = parameters['b' + str(i)]
        V_Wi = adamParameters['V_w' + str(i)]
        M_Wi = adamParameters['M_w' + str(i)]
        V_bi = adamParameters['V_b' + str(i)]
        M_bi = adamParameters['M_b' + str(i)]
        
        M_Wi = beta1 * M_Wi + (1 - beta1) * dWi
        V_Wi = beta2 * V_Wi + (1 - beta2) * np.square(dWi)
        M_W_hat = M_Wi / (1 - beta1 ** t + 1)
        V_W_hat = V_Wi / (1 - beta2 ** t + 1)
        Wi -= learning_rate * (M_W_hat / (np.sqrt(V_W_hat) + epsilon) + weight_decay * Wi)
        
        M_bi = beta1 * M_bi + (1 - beta1) * dbi
        V_bi = beta2 * V_bi + (1 - beta2) * np.square(dbi)
        M_b_hat = M_bi / (1 - beta1 ** t + 1)
        V_b_hat = V_bi / (1 - beta2 ** t + 1)
        bi -= learning_rate * (M_b_hat / (np.sqrt(V_b_hat) + epsilon) + weight_decay * bi)

        parameters['W' + str(i)] = Wi
        parameters['b' + str(i)] = bi

    return parameters

