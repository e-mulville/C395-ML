import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split



# from ../illustrate import illustrate_results_ROI
NUM_CLASSES = 4

def main():
    ############################
    ##### 1 - LOAD DATASET #####
    ############################
    dataset = np.loadtxt("ROI_dataset.dat")
    y_shape = range(NUM_CLASSES)
    x_train, x_test, y_train, y_test = train_test_split(dataset, y_shape,  test_size = 0.25 )
    print(dataset[1230])




    ##############################
    ##### 2 - DEFINE NETWORK #####
    ##############################
    # Init
    network = keras.models.Sequential()
    # Layers
    network.add( Dense(30, input_shape=(3,), activation="sigmoid") )
    network.add( Dense(NUM_CLASSES, activation="softmax") )

    # Define training parameters
    network.compile(    loss = "categorical_crossentropy",
                        optimizer = "sgd",
                        metrics = ["accuracy"])

    print(network.summary())

    #############################
    ##### 3 - TRAIN NETWORK #####
    #############################
    batch_size = 300
    epochs = 100
    network.fit(x_train, y_train, batch_size, epochs)




    ############################
    ##### 4 - TEST NETWORK #####
    ############################
    score = network.evaluate(x_test, y_test)
    print("Test loss    :", score[0])
    print("Test accuracy:", score[1]*100, "%")

    # Get predictions on test data
    predictions = newtwork.predict(x_test)
    predictions.tolist()
    y_true = y_test.tolist()

    # Turn predictions into a 1-hot encoded array
    for i in range(len(predictions)):
        maxVal = max(predictions[i])
        for j in range(len(predictions[i])):
            if (predictions[i][j] < maxVal):
                predictions[i][j] = 0
            else:
                predictions[i][j] = 1

    # Get CM data
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(len(y_true)):
        classPredicted = np.where(predictions[i] == 1)[0][0]
        classTrue = np.where(y_test[i] == 1)[0][0]
        assert(classPredicted in range(NUM_CLASSES))
        assert(classTrue in range(NUM_CLASSES))
        cm[classPredicted][classTrue] += 1

    # Plot CM
    classes = ["ROI-1", "ROI-2", "ROI-3", "None"]
    # plot_confusion_matrix(cm, classes, normalize=True) --> TODO: include function

    # illustrate_results_ROI(network, prep)


if __name__ == "__main__":
    main()
