# Import libraries
import numpy as np
import math
import time
#from pruning import *
from decisiontree import decision_tree_learning as getTree

# Define constants
NUM_CLASSES = 4


def crossValidate(data_set):

    split_size = data_set.size()/10;

    #split out the testing data
    for i in range(0,10):

        split_set = np.split(data_set,[i*split_size, (i+1)*split_size])
        test_set = split_set[1]
        training_set = np.concatenate(split_set[0],split_set[1])

        #split out the validation
        for j in range(0,9):

            split_training_set = np.split(training_set,[j*split_size, (j+1)*split_size])
            validation_set = split_set[1]
            training_set = np.concatinate(split_set[0],split_set[1])

            produced_tree  = train_and_prune(validation_set, training_set)

            test_results[i][j] = test(produced_tree, test_set)


    #average the test reults


#########################################################

def train_and_prune(validation_set, training_set):
    tree = getTree(training_set)

    pruned_tree = prune(tree[0], validation_set)

    return pruned_tree

    #prune it by comparing to validation_set

    #return tree for testing

#########################################################
# Given a tree and a single measurement (7 values)
# return the predicted class
def predict(tree, value):
    # If tree is leaf, return its value
    if (tree["leaf"] != 0):
        return int(tree["leaf"])
    # Else traverse 1 level of the tree depending on the measurement and iterate recursively
    else:
        attr = tree["attribute"]
        split_val = tree["value"]
        if (value[attr] < split_val):
            return predict(tree["left"], value)
        else:
            return predict(tree["right"], value)


#########################################################
# Tests tree based on test set and return 5-tuple
# (Confusion Matrix | Recall | Precision | F score | Classification rate)
def test(tree, test_set):
    # Initialize Confusion Matrix and other metrics variables
    CM = np.zeros((NUM_CLASSES, NUM_CLASSES))
    recall = 0
    precision = 0
    Fscore = 0
    classificationRate = 0
    # Test the tree and fill CM
    for test_val in test_set:
        # for each value in the test set, evaluate the prediction
        classPredicted = predict(tree, test_val)
        # check whether it matches the label and update confusion matrix accordingly
        trueLabel = int(test_val[7])
        CM[trueLabel-1][classPredicted-1] += 1
    # From CM calculate other metrics
        # 1 - Recall
    recallSum = 0
    for room in range(NUM_CLASSES):
        tp = CM[room][room]
        fn = 0
        for x in range(NUM_CLASSES):
            if (x != room): 
                fn += CM[room][x]
            recallTemp = tp / (tp + fn)
        recallSum += recallTemp
    recall = recallSum / NUM_CLASSES
        # 2 - Precision
    precisionSum = 0
    for room in range(NUM_CLASSES):
        tp = CM[room][room]
        fp = 0
        for x in range(NUM_CLASSES):
            if (x != room): 
                fp += CM[x][room]
        precisionTemp = tp / (tp + fp)
        precisionSum += precisionTemp
    precision = precisionSum / NUM_CLASSES
        # 3 - F score
    Fscore = 2 * precision * recall / (precision + recall)
    # 4 - Classification rate
    correctClass = 0
    totalData = 0
    for room in range(NUM_CLASSES):
        correctClass += CM[room][room]
        for x in range(NUM_CLASSES):
            totalData += CM[room][x]
    classificationRate = correctClass / totalData
    
    # Return the metrics
    return (CM, recall, precision, Fscore, classificationRate)


print("Reading datafiles...")
dataSet = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')
print("Producing tree...")
tree = getTree(dataSet, 0)[0]
print("Testing tree...")
# select TESTSET
testSet = dataSet
metrics = test(tree, testSet)
print("Confusion matrix")
print(metrics[0])
print("Recall\t", metrics[1])
print("Precision\t", metrics[2])
print("F Score\t", metrics[3])
print("Accuracy\t", metrics[4])

