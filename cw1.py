import numpy as np
from decisiontree import build_decision_tree
from pruning import prune
from validation import evaluate, print_metrics
from crossValidation import crossValidate
from visualizer import visualizeTree as visualize
from sys import argv
import os


# Parse command line input
flag = ""
fileout = ""
if len(argv) > 1:
    flag = argv[1]
    if flag == "--visualize" and len(argv)==3:
        fileout = argv[2]
    elif flag == "-h" or flag == "--help":
        os.system("cat README.txt")
        exit(-1)
    else:
        print("USAGE:   python3  cw1.py                             --> Clean run")
        print("         python3  cw1.py  --visualize  <filename>    --> Write to the specified file the visualized")
        print("                                                         version of the decision trees created.")
        print("         python3  cw1.py  -h | --help                --> Print README.txt")

        print("\n**** Error: incorrect flag detected, input ignored. ****\n")



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


# STEP 2 - CREATING DECISION TREES
print("\n------- 2 - CREATING DECISION TREES -------")
print("\tCreating tree from clean dataset...")
cleanTree = build_decision_tree(cleanTreeSet, 0)
print("\tDepth of cleanTree:\t", cleanTree[1])
cleanTree = cleanTree[0]
visualize(cleanTree)
pruneClean = prune(cleanTree, noisyTest)
print("\t------------------------------------")
visualize(pruneClean)
print("\tCreating tree from noisy dataset...")
noisyTree = build_decision_tree(noisyTreeSet, 0)
print("\tDepth of noisyTree:\t", noisyTree[1])
noisyTree = noisyTree[0]
    # If requested, print visualized trees to file
if fileout != "":
    visualClean = visualize(cleanTree)
    visualNoisy = visualize(noisyTree)
    try:
        f = open(fileout, "w+") 
        f.write("**** CLEAN TREE ****\n")
        f.write(visualClean)
        f.write("\n\n\n")
        f.write("**** NOISY TREE ****\n")
        f.write(visualNoisy)
        print("\t------------------------------------")
        print("\tTrees correctly printed to file.")
    except IOError:
        print("\t------------------------------------")
        print("\t*** Error: Could not open file! Trees visualization NOT printed! ***")



# STEP 3 - EVALUATION
print("\n------- 3 - EVALUATION -------")
print("\tEvaluating cleanTree on cleanTest...")
metrics = evaluate(cleanTest, cleanTree)
print_metrics(metrics)
print("\t------------------------------------")
print("\tEvaluating cleanTree on noisyTest...")
metrics = evaluate(noisyTest, cleanTree)
print_metrics(metrics)
print("\t------------------------------------")
print("\tEvaluating noisyTree on cleanTest...")
metrics = evaluate(cleanTest, noisyTree)
print_metrics(metrics)
print("\t------------------------------------")
print("\tEvaluating noisyTree on noisyTest...")
metrics = evaluate(noisyTest, noisyTree)
print_metrics(metrics)

#STEP 4 - PRUNING (AND EVALUATION AGAIN)
print("\n------- 4 - PRUNING -------")
print("\tPruning and evaluating cleanTree...")
crossValidate(cleanSet)
print("\t-----------------------------------")
print("\tPruning and evaluating noisyTree...")
crossValidate(noisySet)
