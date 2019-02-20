import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    [input, label] = np.split(dataset, 2, axis = 1)


    print (np.mean(input, axis = 0))

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################


if __name__ == "__main__":
    main()
