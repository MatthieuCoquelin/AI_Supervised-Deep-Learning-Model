import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow_datasets as tfds
import numpy as np


##########################################################
#                    VISUALISATION                       #
##########################################################


def DisplayIA_Train(TrainLoss, TrainAcc):
    plt.figure(figsize = (14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(TrainLoss, label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(TrainAcc, label='train acc')
    plt.legend()
    plt.show()  


def DisplayIA_TrainTest(TrainLoss, TrainAcc, TestLoss, TestAcc):
    plt.figure(figsize = (14, 4))
    plt.subplot(1, 2, 1)
    plt.plot(TrainLoss, label='train loss')
    plt.plot(TestLoss, label='test loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(TrainAcc, label='train acc')
    plt.plot(TestAcc, label='test acc')
    plt.legend()
    plt.show()  



def DisplayPictures(X, y):
    plt.figure(figsize=(16, 8))
    for i in range(1, 21):
        plt.subplot(4, 5, i)
        plt.imshow(X[i])
        plt.title(y[i])
        plt.tight_layout()
    plt.show()


def Display_TensorFlow_Pictures(dataset, ds_info):
    tfds.show_examples(dataset, ds_info, rows=4, cols=4)

def DisplayCircles(X, y, y_pred):
    print(X.shape)
    print(y.shape)
    x_min, x_max = X[:, 0].min() - 0.1, X[:,0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min,x_max, 25),
    np.linspace(y_min, y_max, 8))
    print (xx.shape)
    print (yy.shape)
    print (y_pred.shape)
    y_pred = np.round(y_pred).reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7 )
    plt.scatter(X[:,0], X[:, 1], c=y, s=40, cmap="summer")
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    return plt

def DrawPlot(plt):
    plt.show()