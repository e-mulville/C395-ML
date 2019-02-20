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

    print("\n====== 1. Load and split dataset ======\n")

    dataset = np.loadtxt("ROI_dataset.dat")
    np.random.shuffle(dataset)
    labels = []
    data = []
    for entry in dataset:
        data.append(entry[0:3])
        labels.append(entry[3:7])
        
    # Split dataset
    x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(data, labels, test_size = 0.15 )
    x_train = np.array(x_train_temp)
    x_test = np.array(x_test_temp)
    y_train = np.array(y_train_temp)
    y_test = np.array(y_test_temp)


    ##############################
    ##### 2 - DEFINE NETWORK #####
    ##############################

    print("\n====== 2. Initialize network ======\n")
    # Init
    network = keras.models.Sequential()
    # Layers
    network.add( Dense(100, input_shape=(3,), activation="relu") )
    network.add( Dense(NUM_CLASSES, activation="softmax") )

    # Define training parameters
    network.compile(    loss = "categorical_crossentropy",
                        optimizer = "adam",
                        metrics = ["accuracy"])

    print(network.summary()) # DEBUG
    

    #############################
    ##### 3 - TRAIN NETWORK #####
    #############################
    print("\n====== 3. Train network ======\n")

    batch_size = 50
    epochs = 100
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

    # Turn predictions into a 1-hot encoded array
    for i in range(len(predictions)):
        maxVal = max(predictions[i])
        for j in range(len(predictions[i])):
            if (predictions[i][j] < maxVal):
                predictions[i][j] = 0
            else:
                predictions[i][j] = 1

    # Get CM data
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for i in range(len(y_true)):
        classPredicted = np.where(predictions[i] == 1)[0][0]
        classTrue = np.where(y_test[i] == 1)[0][0]
        assert(classPredicted in range(NUM_CLASSES))
        assert(classTrue in range(NUM_CLASSES))
        cm[classPredicted][classTrue] += int(1)

    # Plot CM
    print("\n\n")
    print(cm)
    # classes = ["ROI-1", "ROI-2", "ROI-3", "None"]
    # plot_confusion_matrix(cm, classes, normalize=True) --> TODO: include function

    # illustrate_results_ROI(network, prep)


if __name__ == "__main__":
    main()
