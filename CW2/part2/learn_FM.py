import numpy as np

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.constraints import max_norm
from keras.optimizers import SGD

from sklearn.model_selection import train_test_split

NUM_CLASSES = 3

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################


    ############################
    ##### 1 - LOAD DATASET #####
    ############################
    rate = 0.5

    [data, labels] = np.split(dataset, 2, axis = 1)


    x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(data, labels, test_size = 0.15 )
    x_train = np.array(x_train_temp)
    x_test = np.array(x_test_temp)
    y_train = np.array(y_train_temp)
    y_test = np.array(y_test_temp)

    ##############################
    ##### 2 - DEFINE NETWORK #####
    ##############################

    network = keras.models.Sequential()
    # Layers
    network.add( Dense(200, input_shape=(3,), activation="relu", kernel_constraint=max_norm(3) ) )
    network.add( Dropout(rate) )
    network.add( Dense(20, input_shape=(100,), activation="relu", kernel_constraint=max_norm(3)) )
    network.add( Dense(NUM_CLASSES, activation="linear") )

    # Define training parameters
    network.compile(    loss = 'mean_squared_error',
                        optimizer = "adam",
                        metrics = ["accuracy"])

    print(network.summary()) # DEBUG

    #############################
    ##### 3 - TRAIN NETWORK #####
    #############################
    print("\n====== 3. Train network ======\n")

    batch_size = 50
    epochs = 1000
    network.fit(x_train, y_train, batch_size, epochs, verbose=0)



    ############################
    ##### 4 - TEST NETWORK #####
    ############################
    print("\n====== 4. Test network ======\n")
    score = network.evaluate(x_test, y_test, verbose=0)
    print("Test loss    :", score[0])
    print("Test accuracy:", score[1]*100, "%")

    # Get predictions on test data
    predictions = network.predict(x_test)
    predictions.tolist()
    y_true = y_test.tolist()

    # # Turn predictions into a 1-hot encoded array
    # for i in range(len(predictions)):
    #     maxVal = max(predictions[i])
    #     for j in range(len(predictions[i])):
    #         if (predictions[i][j] < maxVal):
    #             predictions[i][j] = 0
    #         else:
    #             predictions[i][j] = 1
    #
    # # Get CM data
    # cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    # for i in range(len(y_true)):
    #     classPredicted = np.where(predictions[i] == 1)[0][0]
    #     classTrue = np.where(y_test[i] == 1)[0][0]
    #     assert(classPredicted in range(NUM_CLASSES))
    #     assert(classTrue in range(NUM_CLASSES))
    #     cm[classPredicted][classTrue] += int(1)
    #
    # # Plot CM
    # print("\n\n")
    # print(cm)


    # classes = ["ROI-1", "ROI-2", "ROI-3", "None"]
    # plot_confusion_matrix(cm, classes, normalize=True) --> TODO: include function


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


if __name__ == "__main__":
    main()
