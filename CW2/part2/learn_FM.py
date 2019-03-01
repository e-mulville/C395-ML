import numpy as np

import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.constraints import max_norm
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

from sklearn.model_selection import train_test_split

NUM_CLASSES = 3

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #np.random.shuffle(dataset)
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




    def evaluate(network):
        predictions = network.predict(x_test)
        predictions.tolist()
        y_true = y_test.tolist()

        loss = [0.0, 0.0, 0.0]

        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                loss[j] += ((predictions[i][j] - y_true[i][j])**2)

        for i in range(3):
             loss[i] = loss[i]/len(predictions)
             loss[i] = loss[i]

        return ((loss[0]+loss[1]+loss[2])/3)

    def first_network():
        rate = 0.25

        network = keras.models.Sequential()
        # Layers
        network.add( Dense(700, input_shape=(3,), activation="relu", kernel_constraint=max_norm(3) ) )
        network.add( Dropout(rate) )
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
        print(evaluate(network))

    def hyperparameter():

        # Layers
        best_performance = 100.0
        best_network = keras.models.Sequential()

        for neurons in range(8,9):
            for num_epochs in range (2,3):
            #for dropout_rate in range(0,4):

                    #INITAL NETWORK

                network = keras.models.Sequential()

                network.add( Dense(neurons*100, input_shape=(3,), activation="relu") )
                #network.add( Dropout(dropout_rate*0.1))

                network.add( Dense(NUM_CLASSES, activation="linear") )

                network.compile(    loss = 'mean_squared_error',
                                    optimizer = "adam",
                                    metrics = ["accuracy"])

                print(network.summary()) # DEBUG

                #EVALUATION

                print("################ Training ##################")
                print("########## epochs:", 250*num_epochs ,"##############")

                batch_size = 50
                epochs = 250*num_epochs
                network.fit(x_train, y_train, batch_size, epochs, verbose=0)

                score = network.evaluate(x_test, y_test, verbose=0)
                print("Test loss    :", score[0])
                print("Test accuracy:", score[1], "%")

                print(evaluate(network))

                if evaluate(network) < best_performance:
                    best_performance = evaluate(network)
                    best_network = network

                # print("Best network so far:")
                # print(best_network.summary())




        network = best_network

        print("Best network is:")
        print(best_network.summary())
        network.save("network.h5")
        print("Saved model to disk")

    def predict_hidden(dataset):

        network = load_model('network.h5')

        print("Loaded model from disk")

        print(network.summary())

        data = dataset
        for entry in dataset:
            data.append(entry[0:3])

        input_data = np.array(data)
        predictions = network.predict(input_data)

        return predictions


    #first_network()
    hyperparameter()
    #predict_hidden(dataset)


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


if __name__ == "__main__":
    main()
