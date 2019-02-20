import numpy as np
from decisiontree import build_decision_tree
from pruning import prune
from validation import evaluate, print_metrics
from crossValidation import crossValidate
from visualizer import visualizeTree as visualize
from sys import argv
import os


# Initialize environmental variables
cleanSet_filePath = 'co395-cbc-dt/wifi_db/clean_dataset.txt'
noisySet_filePath = "co395-cbc-dt/wifi_db/noisy_dataset.txt"


def getSafe(ls, index):
    try:
        return ls[index]
    except IndexError:
        return None



# Parse command line input
flag = ""
fileout = ""
inParams = argv
if len(inParams) > 1:
    for i in range(1,len(argv)):
        if inParams[i] != "SEEN":
            curParam = inParams[i]
            nextParam = getSafe(inParams, i+1)

            if curParam == "--visualize" and nextParam != None:
                fileout = nextParam
                inParams[i+1] = "SEEN"   
            elif inParams[i] == "--inClean" and nextParam != None:
                cleanSet_filePath = nextParam
                inParams[i+1] = "SEEN"
            elif inParams[i] == "--inNoisy" and nextParam != None:
                noisySet_filePath = nextParam
                inParams[i+1] = "SEEN"
            elif inParams[i] in ["-h", "--help"]:
                os.system("cat README.txt")
                exit(-1)                
            else:
                print("USAGE:   python3  cw1.py  [flags]")
                print()
                print("FLAGS:   --visualize  <filename>             --> Write to the specified file the visualized")
                print("                                                 version of the decision trees created.")
                print("                  --inClean    <filename>    --> Use the specified file as Clean dataset")
                print("                  --inNoisy    <filename>    --> Use the specified file as Noisy dataset")
                print("                  --help                     --> Print README.txt")
                print("                  -h                         --> Print README.txt")

                print("\n**** Error: incorrect flag detected. ****\n")
                exit(-1)



# STEP 1 - LOADING DATA
print("\n------- 1 - LOADING DATA -------")
print("\tReading clean dataset...")
cleanSet = np.loadtxt(cleanSet_filePath)
np.random.shuffle(cleanSet)
print("\tReading noisy dataset...")
noisySet = np.loadtxt(noisySet_filePath)
print("\tSplitting data into training and test sets...")
cleanTest = cleanSet[:200]
cleanTreeSet = cleanSet[200:]
noisyTest = noisySet[:200]
noisyTreeSet = noisySet[200:]


# STEP 2 - CREATING DECISION TREES
print("\n------- 2 - CREATING DECISION TREES -------")
print("\tCreating tree from clean dataset...")
cleanTree, cleanDepth = build_decision_tree(cleanTreeSet, 0)
print("\tDepth of cleanTree:\t", cleanDepth)

print("\t------------------------------------")
print("\tCreating tree from noisy dataset...")
noisyTree , noisyDepth= build_decision_tree(noisyTreeSet, 0)
print("\tDepth of noisyTree:\t", noisyDepth)

    # If requested, print visualized trees to file
    # (original and pruned version)
if fileout != "":
    visualClean = visualize(cleanTree)
    visualNoisy = visualize(noisyTree)

    prunedClean = prune(cleanTree, cleanTest)
    prunedNoisy = prune(noisyTree, noisyTest)
    
    visualPrunedClean = visualize(prunedClean)
    visualPrunedNoisy = visualize(prunedNoisy)
    try:
        f = open(fileout, "w+") 
        f.write("**** CLEAN TREE -> NOT pruned ****\n")
        f.write(visualClean)
        f.write("\n\n\n")
        f.write("**** CLEAN TREE -> pruned ****\n")
        f.write(visualPrunedClean)
        f.write("\n\n\n")
        f.write("**** NOISY TREE -> NOT pruned) ****\n")
        f.write(visualNoisy)
        f.write("\n\n\n")
        f.write("**** NOISY TREE -> pruned) ****\n")
        f.write(visualPrunedNoisy)
        f.close()

        print("\t------------------------------------")
        print("\tTrees correctly printed to file.")
    except IOError:
        print("\t------------------------------------")
        print("\t*** IOError: Could not write to file! Trees visualization NOT printed! ***")



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
