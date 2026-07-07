from utilities import load_data
from sklearn.datasets import make_circles
from training import NeuralNetwork
# from visualisation import DisplayPictures


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



