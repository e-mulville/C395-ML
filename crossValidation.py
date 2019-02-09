import numpy as np
import math
import time
from decisiontree import build_decision_tree as getTree
from pruning import prune
from validation import evaluate, print_metrics

def crossValidate(data_set):
    #80 because size gives total data points not no. of rows
    empty_confusion_matrix = np.zeros((4,4))
    empty_metrics = (empty_confusion_matrix, 0.0, 0.0, 0.0, 0.0)
    test_results = [[empty_metrics for a in range(9)] for b in range(10)]
    test_results_after = [[empty_metrics for a in range(9)] for b in range(10)]

    split_size = int(data_set.size/80)

    startTime = time.time()
    #split out the testing data
    for i in range(10):

        split_set = np.split(data_set,[i*split_size, (i+1)*split_size])
        test_set = split_set[1]
        set_without_test = np.concatenate((split_set[0],split_set[2]),axis = 0)

        #split out the validation
        for j in range(1):

            split_training_set = np.split(set_without_test,[j*split_size, (j+1)*split_size])
            validation_set = split_training_set[1]
            training_set = np.concatenate((split_training_set[0],split_training_set[2]), axis = 0)

            tree = getTree(training_set, 0)[0]
            test_results[i][j] = evaluate(test_set, tree)
            pruned_tree = prune(tree, validation_set)
            test_results_after[i][j] = evaluate(test_set, pruned_tree)
            percent = (float(i*9)+float(j+1))/0.9

            timeElapsed = time.time() - startTime
            timeLeft = timeElapsed/percent * (100-percent)
            print ("\r\t", round(percent,2), "%\t Time elapsed: ", int(timeElapsed/3600),":",int((timeElapsed/60)%60),":",int(timeElapsed%60), "\t Time left: ", int(timeLeft/3600),":",int((timeLeft/60)%60),":",int(timeLeft%60), end="\t\t", sep="")

    #average the test results
    recall_matrix = np.zeros((10,9))
    precision_matrix = np.zeros((10,9))
    F_matrix = np.zeros((10,9))
    CR_matrix = np.zeros((10,9))

    for i in range(10):
        for j in range(9):
            recall_matrix[i][j] = test_results[i][j][1]
            precision_matrix[i][j] = test_results[i][j][2]
            F_matrix[i][j] = test_results[i][j][3]
            CR_matrix[i][j] = test_results[i][j][4]

    average_test_results = (np.average(test_results[0]),np.average(recall_matrix),np.average(precision_matrix),np.average(F_matrix),np.average(CR_matrix))

    for i in range(10):
        for j in range(9):
            recall_matrix[i][j] = test_results_after[i][j][1]
            precision_matrix[i][j] = test_results_after[i][j][2]
            F_matrix[i][j] = test_results_after[i][j][3]
            CR_matrix[i][j] = test_results_after[i][j][4]
    average_test_results_after = (np.average(test_results_after[0]),np.average(recall_matrix),np.average(precision_matrix),np.average(F_matrix),np.average(CR_matrix))

    print("Done:")
    print()
    print("Results before pruning:")
    print_metrics(average_test_results)
    print("Results after pruning:")
    print_metrics(average_test_results_after)
    # print ("\tAverage test results:       ", round(np.average(test_results)*100,3), "%")
    # print ("\tAverage test results after: ", round(np.average(test_results_after)*100,3), "%")



#########################################################
