import numpy as np

import tensorflow as tf
import keras
from keras.models import Sequential, model_from_json
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


    [data, labels] = np.split(dataset, 2, axis = 1)


    x_train_temp, x_test_temp, y_train_temp, y_test_temp = train_test_split(data, labels, test_size = 0.15 )
    x_train = np.array(x_train_temp)
    x_test = np.array(x_test_temp)
    y_train = np.array(y_train_temp)
    y_test = np.array(y_test_temp)

    ##############################
    ##### 2 - DEFINE NETWORK #####
    ##############################

    rate = 0.25


    def first_network():
        network = keras.models.Sequential()
        # Layers
        network.add( Dense(700, input_shape=(3,), activation="relu", kernel_constraint=max_norm(3) ) )
        network.add( Dropout(rate) )
        network.add( Dense(100, input_shape=(3,), activation="relu", kernel_constraint=max_norm(3) ) )
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
        print("Test accuracy:", score[1], "%")

        # Get predictions on test data
        predictions = network.predict(x_test)
        predictions.tolist()
        y_true = y_test.tolist()

        loss = [0.0, 0.0, 0.0]

        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                loss[j] += ((predictions[i][j] - y_true[i][j])**2)

        for i in range(3):
             loss[i] = loss[i]/len(predictions)
             loss[i] = loss[i]**0.5
        print("Loss:", loss)

    def hyperparameter():

        # Layers


        for j in range(1,7):
            for i in range(0,3):
                network = keras.models.Sequential()

                network.add( Dense(j*100, input_shape=(3,), activation="relu", kernel_constraint=max_norm(3) ) )

                 #Layers
                for x in range(i):
                    network.add( Dense(j*100, activation="relu", kernel_constraint=max_norm(3) ) )

                network.add( Dense(NUM_CLASSES, activation="linear") )

                network.compile(    loss = 'mean_squared_error',
                                    optimizer = "adam",
                                    metrics = ["accuracy"])
                print(network.summary())


        network_json = network.to_json()
        with open("network.json", "w") as json_file:
            json_file.write(network_json)
        # serialize weights to HDF5
        model.save_weights("network.h5")
        print("Saved model to disk")

    def predict_hidden(dataset):
        json_file = open('network.json', 'r')
        loaded_network_json = json_file.read()
        json_file.close()
        loaded_network = model_from_json(loaded_network_json)
        # load weights into new model
        loaded_network.load_weights("model.h5")
        print("Loaded model from disk")

    first_network()
    hyperparameter()


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


if __name__ == "__main__":
    main()
