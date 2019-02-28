import numpy as np
import tensorflow as tf
import keras
import os, sys
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

curDir = os.getcwd()
temp = curDir.split("/")
temp = temp[:len(temp)-1]
parentDir = "/".join(temp)
sys.path.append(parentDir)

from illustrate import illustrate_results_ROI


NUM_CLASSES = 4

def main():

    ############################
    ##### 1 - LOAD DATASET #####
    ############################

    print("\n====== 1. Load and split dataset ======\n")

    (x_train, y_train, x_test, y_test) = load_dataset(test_size = 0.2)

    
    ##############################
    ##### 2 - DEFINE NETWORK #####
    ##############################

    print("\n====== 2. Initialize network ======\n")

    network = init_network( hiddenNeurons = 100,
                            hiddenActivation = "relu",
                            dropout = 0.25,
                            optimizer = "adam",
                            verbose = 1)
    

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
    evaluate_network(network, x_test, y_test)



##############################################################

def load_dataset(test_size):
        dataset = np.loadtxt("ROI_dataset.dat")
        np.random.shuffle(dataset)
        labels = []
        data = []
        for entry in dataset:
            data.append(entry[0:3])
            labels.append(entry[3:7])
            
        # Split dataset
        x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(data, labels, test_size = test_size)
        x_train = np.array(x_train_temp)
        x_test = np.array(x_test_temp)
        y_train = np.array(y_train_temp)
        y_test = np.array(y_test_temp) 

        return(x_train, y_train, x_test, y_test)



def init_network(hiddenNeurons, hiddenActivation, dropout, optimizer, verbose):
        # Init
        network = keras.models.Sequential()
        # Layers
        network.add( Dense(hiddenNeurons, input_shape=(3,), activation=hiddenActivation) )
        network.add( Dropout(dropout) )
        network.add( Dense(NUM_CLASSES, activation="softmax") )

        # Define training parameters
        network.compile(    loss = "categorical_crossentropy",
                            optimizer = optimizer,
                            metrics = ["accuracy"])

        if (verbose):
            print(network.summary()) # DEBUG
        
        return network



def evaluate_network(network, x_test, y_test, verbose=0):

    score = network.evaluate(x_test, y_test, verbose=verbose)
    print("Test loss       :", round(score[0], 4))
    print("Test accuracy   :",round(score[1]*100, 3), "%")


    # Get predictions on test data
    predictions = network.predict(x_test)
    predictions.tolist()
    y_true = y_test.tolist()

    # Turn predfrom keras.models import load_modelictions into a 1-hot encoded array
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
    print("\n")
    print("Confusion matrix:")
    print(cm)
    print("\n")



def predict_hidden(dataset): # --> TODO: implement
    # Preprocess dataset
    # Load best neural network
    network = load_model('best_model.h5')
    # Return predictions on dataset


if __name__ == "__main__":
    main()





