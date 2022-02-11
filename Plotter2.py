import matplotlib.pyplot as plt
def plot_creater(history,bin, modelname):
    if (modelname=="CNN" or modelname=="LSTM"):
        if (bin==True):
        
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


        elif (bin==False):
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
    elif (modelname == "resnet"):
        if (bin == True):

            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig('./resnet_Model/acc_val_bin.png')
            plt.savefig('./resnet_Model/acc_val_bin.pdf')
            plt.close()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')

            plt.savefig('./resnet_Model/loss_val_bin.png')
            plt.savefig('./resnet_Model/loss_val_bin.pdf')
            plt.close()


        elif (bin == False):
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.savefig('./resnet_Model/acc_val_multi.png')
            plt.savefig('./resnet_Model/acc_val_multi.pdf')
            plt.close()

            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')

            plt.savefig('./resnet_Model/loss_val_multi.png')
            plt.savefig('./resnet_Model/loss_val_multi.pdf')
            plt.close()

    else:
        print("Kein Plot verfügbar")

