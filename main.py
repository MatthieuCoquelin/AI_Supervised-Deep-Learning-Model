from utilities import load_data
from sklearn.datasets import make_circles
from training import NeuralNetwork_Train, NeuralNetwork_TrainTest
# from visualisation import DisplayPictures


if __name__ == "__main__":

    N = (32, 32, 32)

    launcherList = ["Circles", "Images"]
    selection = launcherList[1]

    match selection:
        case "Circles":

            X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
            X = X.T
            y = y.reshape((1, y.shape[0]))

            NeuralNetwork_Train(X, y, N, learning_rate=0.1, num_iterations=1000)
        case "Images":

            X_train, y_train, X_test, y_test = load_data()
            reshaped_X_train2 = X_train.reshape( X_train.shape[1] * X_train.shape[2], X_train.shape[0]) / X_train.max()
            reshaped_X_test2 = X_test.reshape(X_test.shape[1] * X_test.shape[2], X_test.shape[0]) / X_train.max()
            y_train = y_train.T
            y_test = y_test.T
        
            NeuralNetwork_TrainTest(reshaped_X_train2, y_train, reshaped_X_test2, y_test, N, learning_rate=0.01, num_iterations=10000)
        case _:
            print("Invalid input")
    # DisplayPictures()





