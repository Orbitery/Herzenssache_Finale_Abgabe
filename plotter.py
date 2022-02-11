import matplotlib.pyplot as plt
def plot_creater(history,bin, modelname):
    """[For the training progress, a chart about the accuracy / loss is created for the deep learning approaches and stored accordingly]

    Args:
        history (keras.callbacks.History object): [Contains values accuracy, validation-accuracy, validation-loss and loss values during the training of the model]
        bin (String): [shows if binary ("True") or multilabel ("False") classification is active]
        modelname (String): [Name of Model]
    """
    if (modelname=="CNN" or modelname=="LSTM"):
        if (bin=="True"):
        
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig('./CNN_bin/acc_val_bin.png')
            plt.savefig('./CNN_bin/acc_val_bin.pdf')
            plt.close()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')

            plt.savefig('./CNN_bin/loss_val_bin.png')
            plt.savefig('./CNN_bin/loss_val_bin.pdf')
            plt.close()


        elif (bin=="False"):
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig('./CNN_multi/acc_val_bin.png')
            plt.savefig('./CNN_multi/acc_val_bin.pdf')
            plt.close()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')

            plt.savefig('./CNN_multi/loss_val_bin.png')
            plt.savefig('./CNN_multi/loss_val_bin.pdf')
            plt.close()
    elif (modelname == "Resnet"):
        if (bin == "True"):

            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig('./resnet_bin/acc_val_bin.png')
            plt.savefig('./resnet_bin/acc_val_bin.pdf')
            plt.close()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')

            plt.savefig('./resnet_bin/loss_val_bin.png')
            plt.savefig('./resnet_bin/loss_val_bin.pdf')
            plt.close()


        elif (bin == "False"):
            plt.plot(history.history['accuracy'])
            plt.plot(history.history['val_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig('./resnet_multi/acc_val_multi.png')
            plt.savefig('./resnet_multi/acc_val_multi.pdf')
            plt.close()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')

            plt.savefig('./resnet_multi/loss_val_multi.png')
            plt.savefig('./resnet_multi/loss_val_multi.pdf')
            plt.close()

    else:
        print("No Plot available")

