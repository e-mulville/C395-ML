import numpy as np

from ../nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network
)

from ../illustrate import illustrate_results_ROI


def main():
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    illustrate_results_ROI(network, prep)


if __name__ == "__main__":
    main()
