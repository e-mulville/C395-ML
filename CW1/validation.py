# Import libraries
import numpy as np
import math
import warnings
warnings.simplefilter("error")


# Define constants
NUM_CLASSES = 4

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
def evaluate(test_set, tree):
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
        try:
            recallTemp = np.float64(tp) / (tp + fn)
        except:
            pass
        recallSum += recallTemp
    recall = np.float64(recallSum) / NUM_CLASSES
        # 2 - Precision
    precisionSum = 0
    for room in range(NUM_CLASSES):
        tp = CM[room][room]
        fp = 0
        for x in range(NUM_CLASSES):
            if (x != room):
                fp += CM[x][room]
        try:
            precisionTemp = np.float64(tp) / (tp + fp)
        except:
            pass
        precisionSum += precisionTemp
    precision = np.float64(precisionSum) / NUM_CLASSES
        # 3 - F score
    Fscore = np.float64(2 * precision * recall) / (precision + recall)
    # 4 - Classification rate
    correctClass = 0
    totalData = 0
    for room in range(NUM_CLASSES):
        correctClass += CM[room][room]
        for x in range(NUM_CLASSES):
            totalData += CM[room][x]
    classificationRate = np.float64(correctClass) / totalData

    # Return the metrics
    return (CM, recall, precision, Fscore, classificationRate)


# Function that takes the metrics 5-tuple and prints it in a nice format
# (Confusion Matrix | Recall | Precision | F score | Classification rate)
def print_metrics(metrics):
    def pretty(n):
        return (str(round(n*100,4)) + " %")
    print("\tConfusion Matrix: ")
    rounded_matrix = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            rounded_matrix[i][j] = round(metrics[0][i][j], 4)
    print(rounded_matrix)
    print("\tRecall:              ", pretty(metrics[1]))
    print("\tPrecision:           ", pretty(metrics[2]))
    print("\tF Score:             ", pretty(metrics[3]))
    print("\tClassification Rate: ", pretty(metrics[4]))

