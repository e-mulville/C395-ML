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
    unpruned_results = [[empty_metrics for a in range(9)] for b in range(10)]
    pruned_results = [[empty_metrics for a in range(9)] for b in range(10)]

    split_size = int(data_set.size/80)

    startTime = time.time()
    #split out the testing data
    for i in range(10):

        split_set = np.split(data_set,[i*split_size, (i+1)*split_size])
        test_set = split_set[1]
        set_without_test = np.concatenate((split_set[0],split_set[2]),axis = 0)

        #split out the validation
        for j in range(9):

            split_training_set = np.split(set_without_test,[j*split_size, (j+1)*split_size])
            validation_set = split_training_set[1]
            training_set = np.concatenate((split_training_set[0],split_training_set[2]), axis = 0)

            tree = getTree(training_set, 0)[0]
            unpruned_results[i][j] = evaluate(test_set, tree)
            pruned_tree = prune(tree, validation_set)
            pruned_results[i][j] = evaluate(test_set, pruned_tree)

            #stuff for printing nicely
            percent = (float(i*9)+float(j+1))/0.9
            timeElapsed = time.time() - startTime
            timeLeft = timeElapsed/percent * (100-percent)
            print ("\r\t", round(percent,2), "%\t Time elapsed: ", int(timeElapsed/3600),":",int((timeElapsed/60)%60),":",int(timeElapsed%60), "\t Time left: ", int(timeLeft/3600),":",int((timeLeft/60)%60),":",int(timeLeft%60), end="\t\t", sep="")



    average_unpruned_results = average_metrics(unpruned_results)
    average_pruned_results = average_metrics(pruned_results)

    print("Done:")
    print()
    print("\nResults before pruning:\n")
    print_metrics(average_unpruned_results)
    print("\nResults after pruning:\n")
    print_metrics(average_pruned_results)
    # print ("\tAverage test results:       ", round(np.average(unpruned_results)*100,3), "%")
    # print ("\tAverage test results after: ", round(np.average(pruned_results)*100,3), "%")



#########################################################

def average_metrics(metrics_matrix):
        #average the test results
        Confusion_matrix = np.zeros((10,9), dtype = object)
        recall_matrix = np.zeros((10,9))
        precision_matrix = np.zeros((10,9))
        F_matrix = np.zeros((10,9))
        CR_matrix = np.zeros((10,9))

        for i in range(10):
            for j in range(9):
                Confusion_matrix[i][j] = metrics_matrix[i][j][0]
                recall_matrix[i][j] = metrics_matrix[i][j][1]
                precision_matrix[i][j] = metrics_matrix[i][j][2]
                F_matrix[i][j] = metrics_matrix[i][j][3]
                CR_matrix[i][j] = metrics_matrix[i][j][4]
        average_metrics = (np.average(Confusion_matrix),np.average(recall_matrix),np.average(precision_matrix),np.average(F_matrix),np.average(CR_matrix))
        return average_metrics
