import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    print(dataset.shape)
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
