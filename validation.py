import numpy as np
import math
import time

def crossValidate(data_set){

    split_size = data_set.size()/10;

    #split out the testing data
    for i in range(0,10):

        split_set = np.split(data_set,[i*split_size, (i+1)*split_size])
        test_set = split_set[1]
        training_set = np.concatinate(split_set[0],split_set[1])

        #split out the validation
        for j in range(0,9):

            split_training_set = np.split(training_set,[j*split_size, (j+1)*split_size])
            validation_set = split_set[1]
            training_set = np.concatinate(split_set[0],split_set[1])

            produced_tree  = train_and_prune(validation_set, training_set)

            test_results[i][j] = test(produced_tree, test_set)


    #average the test reults
}

def train_and_prune(validation_set, training_set){

    #build tree from training_set

    #prune it by comparing to validation_set

    #return tree for testing

}


def test(tree, test_set){
    #test the tree

    #return the metrics
}
