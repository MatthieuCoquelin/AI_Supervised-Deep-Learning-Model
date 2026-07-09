from utilities import load_data, load_tensorFlow_dataset, Dataset_To_GrayScale, Resize_Dataset, Convert_TensorFlow_To_Numpy
from sklearn.datasets import make_circles
from training import NeuralNetwork_Train, NeuralNetwork_TrainTest
from visualisation import DisplayPictures

if __name__ == "__main__":

    N = (32, 32, 32)

    launcherList = ["Circles", "Local_Dataset", "TensorFlow_Dataset"]
    selection = launcherList[2]

    match selection:
        case "Circles":
            X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
            X = X.T
            y = y.reshape((1, y.shape[0]))

            NeuralNetwork_Train(X, y, N, learning_rate=0.1, num_iterations=1000)
        case "Local_Dataset":
            X_train, y_train, X_test, y_test = load_data()
            
            reshaped_X_train = X_train.reshape( X_train.shape[1] * X_train.shape[2], X_train.shape[0]) / X_train.max()
            reshaped_X_test = X_test.reshape(X_test.shape[1] * X_test.shape[2], X_test.shape[0]) / X_train.max()
            y_train = y_train.T
            y_test = y_test.T
        
            NeuralNetwork_TrainTest(reshaped_X_train, y_train, reshaped_X_test, y_test, N, learning_rate=0.01, num_iterations=10000)
        case "TensorFlow_Dataset":
            X_train, y_train, X_test, y_test = load_tensorFlow_dataset("horses_or_humans")

            X_train, X_test = Dataset_To_GrayScale(X_train, X_test)
            X_train, X_test = Resize_Dataset(X_train, X_test, int(X_train.shape[1] / 4), int(X_test.shape[1] / 4))

            X_train, y_train, X_test, y_test = Convert_TensorFlow_To_Numpy(X_train, y_train, X_test, y_test)

            X_train = X_train.reshape(X_train.shape[1] * X_train.shape[2], X_train.shape[0]) / X_train.max()
            y_train = y_train.reshape(1, y_train.shape[0])
            X_test = X_test.reshape(X_test.shape[1] * X_test.shape[2], X_test.shape[0]) / 255
            y_test = y_test.reshape(1, y_test.shape[0])

            NeuralNetwork_TrainTest(X_train, y_train, X_test, y_test, N, learning_rate=0.01, num_iterations=10000)
        case _:
            print("Invalid input")
    





