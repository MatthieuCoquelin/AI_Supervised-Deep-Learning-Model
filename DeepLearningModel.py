from utilities import *
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
import plotly.graph_objects as go
from tqdm import tqdm

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


def Normalize(A):
    return (A - np.mean(A, axis=0)) / np.std(A, axis=0)


##########################################################
#                      TRAINNING                         #
##########################################################


# def ArtificialNeuron(X_train, y_train, X_test, y_test,learning_rate = 0.1, num_iterations = 1000) :
def NeuralNetwork(X_train, y_train, hidden_layers, learning_rate = 0.1, num_iterations = 1000) :
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    np.random.seed(1)
    parameters = Initialisation(dimensions)

    TrainLoss = []
    TrainAcc = []
    # TestLoss = []
    # TestAcc = []

    for i in tqdm(range(num_iterations)):
        activations = ForwardPropagation(X_train, parameters)
        gradients = BackPropagation(activations, y_train, X_train, parameters)
        parameters = Update(gradients, parameters, learning_rate)

        # A_test = Model(X_test, W, b)
        
        if i % 10 == 0:
            #train
            #log_loss()
            C = len(parameters) // 2
            TrainLoss.append(L(y_train, activations['A' + str(C)])) 
            y_pred = Predict(X_train, parameters)
            TrainAcc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))

            #test
            # TestLoss.append(L(y_test, A_test)) 
            # y_pred = Predict(X_test, W, b)
            # TestAcc.append(accuracy_score(y_test, y_pred))

        # W, b = Update(X_train, y_train, W, b, learning_rate)

    # DisplayIA(TrainLoss, TrainAcc, TestLoss, TestAcc)
    DisplayIA(TrainLoss, TrainAcc)
    return parameters

##########################################################
#                    VISUALISATION                       #
##########################################################

#def DisplayIA(TrainLoss, TrainAcc, TestLoss, TestAcc):
def DisplayIA(TrainLoss, TrainAcc):
    plt.figure(figsize = (14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(TrainLoss, label='train loss')
    # plt.plot(TestLoss, label='test loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(TrainAcc, label='train acc')
    # plt.plot(TestAcc, label='test acc')
    plt.legend()
    plt.show()  



# def DisplayPictures():
#     plt.figure(figsize=(16, 8))
#     for i in range(1, 10):
#         plt.subplot(4, 5, i)
#         plt.imshow(X_train[i], cmap='gray')
#         plt.title(y_train[i])
#         plt.tight_layout()
#         plt.show()


if __name__ == "__main__":

    X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X = X.T
    y = y.reshape((1, y.shape[0]))
    
    
    # X_train, y_train, X_test, y_test = load_data()
    # reshaped_X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]) / X_train.max()
    # reshaped_X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]) / X_train.max()
    # print(reshaped_X_train.shape)
    # ArtificialNeuron(reshaped_X_test, y_test, 0.01, 200)
    # DisplayPictures()

    N = [32, 32, 32]
    NeuralNetwork(X, y, N, learning_rate=0.1, num_iterations=1000)



