import numpy as np
import math
import time
from decisiontree import build_decision_tree as getTree
from pruning import prune
from validation import evaluate

def crossValidate(data_set):
    #80 because size gives total data points not no. of rows
    test_results = np.zeros((10,9))
    test_results_after = np.zeros((10,9))

    split_size = int(data_set.size/80)

    startTime = time.time()
    total_improvment = [0.0, 0.0, 0.0, 0.0]
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

            tree = getTree(training_set, 0)[0]
            test_results[i][j] = evaluate(test_set, tree)[4]
            pruned_tree = prune(tree, validation_set)
            test_results_after[i][j] = evaluate(test_set, pruned_tree)[4]
            percent = (float(i*9)+float(j+1))/0.9

            timeElapsed = time.time() - startTime
            timeLeft = timeElapsed/percent * (100-percent)
            print ("\t", round(percent,2), "%\t Time elapsed: ", int(timeElapsed/3600),":",int((timeElapsed/60)%60),":",int(timeElapsed%60), "\t Time left: ", int(timeLeft/3600),":",int((timeLeft/60)%60),":",int(timeLeft%60), end="\t\t\r", sep="")

    #average the test results
    print()
    print ("\tAverage test results:       ", round(np.average(test_results)*100,3), "%")
    print ("\tAverage test results after: ", round(np.average(test_results_after)*100,3), "%")



#########################################################
