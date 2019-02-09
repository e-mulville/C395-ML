# Import libraries
import numpy as np
import math
# import matplotlib.pyplot as plt
# import itertools


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
            recallTemp = np.float64(tp) / (tp + fn)
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
        precisionTemp = np.float64(tp) / (tp + fp)
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




# # This function was given to the students for a CW in CO316 Computer Vision to display the confusion matrix.
# def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
#     """
#         This function prints and plots the confusion matrix.

#         cm: confusion matrix, default to be np.int32 data type
#         classes: a list of the class labels or class names
#         normalize: normalize the matrix so that each row amounts to one
#         cmap: color map
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#     print(cm)

#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)

#     fmt = '.2f' if normalize else '.0f'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], fmt),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
