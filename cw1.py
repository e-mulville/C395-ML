import numpy as np
from decisiontree import build_decision_tree
# from pruning import *
from validation import evaluate, print_metrics
from crossValidation import crossValidate




# STEP 1 - LOADING DATA
print("\n------- 1 - LOADING DATA -------")
print("\tReading clean dataset...")
cleanSet = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')
np.random.shuffle(cleanSet)
print("\tReading noisy dataset...")
noisySet = np.loadtxt("co395-cbc-dt/wifi_db/noisy_dataset.txt")
print("\tSplitting data into training and test sets...")
cleanTest = cleanSet[:200]
cleanTreeSet = cleanSet[200:]
noisyTest = noisySet[:200]
noisyTreeSet = noisySet[200:]
#
# # STEP 2 - CREATING DECISION TREES
# print("\n------- 2 - CREATING DECISION TREES -------")
# print("\tCreating tree from clean dataset...")
# cleanTree = build_decision_tree(cleanTreeSet, 0)[0]
# print("\tCreating tree from noisy dataset...")
# noisyTree = build_decision_tree(noisyTreeSet, 0)[0]
#     # TODO - function to visualize the tree

# # STEP 3 - EVALUATION
# print("\n------- 3 - EVALUATION -------")
# print("\tEvaluating cleanTree on cleanTest...")
# metrics = evaluate(cleanTest, cleanTree)
# print_metrics(metrics)
# print("\t------------------------------------")
# print("\tEvaluating cleanTree on noisyTest...")
# metrics = evaluate(noisyTest, cleanTree)
# print_metrics(metrics)
# print("\t------------------------------------")
# print("\tEvaluating noisyTree on cleanTest...")
# metrics = evaluate(cleanTest, noisyTree)
# print_metrics(metrics)
# print("\t------------------------------------")
# print("\tEvaluating noisyTree on noisyTest...")
# metrics = evaluate(noisyTest, noisyTree)
# print_metrics(metrics)

# STEP 4 - PRUNING (AND EVALUATION AGAIN)
print("\n------- 4 - PRUNING -------")
print("\tPruning and evaluating cleanTree...")
crossValidate(cleanSet)
print("\t-----------------------------------")
print("\tPruning and evaluating noisyTree...")
crossValidate(noisySet)
