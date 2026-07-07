import matplotlib.pyplot as plt
import plotly.graph_objects as go


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