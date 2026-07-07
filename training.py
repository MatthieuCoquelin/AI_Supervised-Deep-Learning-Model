from visualisation import DisplayIA
from model import *
from sklearn.metrics import accuracy_score
from tqdm import tqdm


##########################################################
#                      TRAINNING                         #
##########################################################


# def ArtificialNeuron(X_train, y_train, X_test, y_test,learning_rate = 0.1, num_iterations = 1000) :
def NeuralNetwork(X_train, y_train, hidden_layers, learning_rate = 0.1, num_iterations = 1000) :
    dimensions = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
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



