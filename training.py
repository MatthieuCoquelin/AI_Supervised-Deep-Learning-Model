from visualisation import DisplayIA_Train, DisplayIA_TrainTest
from model import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm


##########################################################
#                      TRAINNING                         #
##########################################################


def NeuralNetwork_Train(X_train, y_train, hidden_layers, learning_rate = 0.1, num_iterations = 1000) :
    dimensions = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    np.random.seed(1)
    parameters = Initialisation(dimensions)

    TrainLoss = []
    TrainAcc = []


    for i in tqdm(range(num_iterations)):
        activations = ForwardPropagation(X_train, parameters)
        gradients = BackPropagation(activations, y_train, X_train, parameters)
        parameters = Update(gradients, parameters, learning_rate)

        if i % 10 == 0:
            #train
            C = len(parameters) // 2
            TrainLoss.append(L(y_train, activations['A' + str(C)]))
            y_pred = Predict(X_train, parameters)
            TrainAcc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))

    DisplayIA_Train(TrainLoss, TrainAcc)
    return parameters



def NeuralNetwork_TrainTest(X_train, y_train, X_test, y_test, hidden_layers, learning_rate = 0.1, num_iterations = 1000) :
    dimensions = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    np.random.seed(1)
    parameters = Initialisation(dimensions)

    TrainLoss = []
    TrainAcc = []
    TestLoss = []
    TestAcc = []

    for i in tqdm(range(num_iterations)):
        activations_train = ForwardPropagation(X_train, parameters)
        gradients = BackPropagation(activations_train, y_train, X_train, parameters)
        parameters = Update(gradients, parameters, learning_rate)

        activations_test = ForwardPropagation(X_test, parameters)


        if i % 10 == 0:
            #train
            C = len(parameters) // 2
            TrainLoss.append(L(y_train, activations_train['A' + str(C)]))
            y_pred = Predict(X_train, parameters)
            TrainAcc.append(accuracy_score(y_train.flatten(), y_pred.flatten()))
            
            #test
            TestLoss.append(L(y_test, activations_test['A' + str(C)]))
            y_pred = Predict(X_test, parameters)
            TestAcc.append(accuracy_score(y_test.flatten(), y_pred.flatten()))


    DisplayIA_TrainTest(TrainLoss, TrainAcc, TestLoss, TestAcc)
    return parameters



