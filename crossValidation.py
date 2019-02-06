import numpy as np
import math
import time
from decisiontree import build_decision_tree as getTree
from pruning import prune
from validation import evaluate

def crossValidate(data_set):
    #80 because size gives total data points not no. of rows
    split_size = int(data_set.size/80)

    total_improvment = [0.0]
    #split out the testing data
    for i in range(0,10):

        split_set = np.split(data_set,[i*split_size, (i+1)*split_size])
        test_set = split_set[1]
        set_without_test = np.concatenate((split_set[0],split_set[2]),axis = 0)

        #split out the validation
        for j in range(0,9):

            split_training_set = np.split(set_without_test,[j*split_size, (j+1)*split_size])
            validation_set = split_training_set[1]
            training_set = np.concatenate((split_training_set[0],split_training_set[2]), axis = 0)

            pruned_tree  = train_and_prune(validation_set, training_set,test_set,total_improvment)

            #
            # test_results[i][j] = evaluate(test_set, produced_tree)


    #average the test reults


#########################################################
def train_and_prune(validation_set, training_set,test_set, total_improvment):

    tree = getTree(training_set, 0)[0]
    before = evaluate(test_set, tree)[4]
    pruned_tree = prune(tree, validation_set)
    after =  evaluate(test_set, pruned_tree)[4]
    total_improvment[0] += after - before
    print (total_improvment)
    return pruned_tree

    #prune it by comparing to validation_set

    #return tree for testing

set = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')
np.random.shuffle(set)
crossValidate(set)
