import matplotlib.pyplot as plt
import plotly.graph_objects as go
import tensorflow_datasets as tfds


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