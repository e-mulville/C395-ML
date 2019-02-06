import numpy as np
from decisiontree import decision_tree_learning
# from pruning import *
from validation import evaluate, crossValidate




# STEP 1 - LOADING DATA
print("\n----- 1 - LOADING DATA -----")
print("\tReading clean dataset...")
cleanSet = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')
np.random.shuffle(cleanSet)
print("\tReading noisy dataset...")
noisySet = np.loadtxt("co395-cbc-dt/wifi_db/noisy_dataset.txt")

# STEP 2 - CREATING DECISION TREES
print("\n----- 2 - CREATING DECISION TREES -----")
print("\tCreating tree from clean dataset...")
cleanTree = decision_tree_learning(cleanSet, 0)
print("\tCreating tree from clean dataset...")
noisyTree = decision_tree_learning(noisySet, 0)
    # TODO - function to visualize the tree

# STEP 3 - EVALUATION
print("\n----- 3 - EVALUATION -----")
print("\tEvaluating cleanTree on cleanSet...")
evaluate(cleanSet, cleanTree)
print("\tEvaluating cleanTree on noisySet...")
evaluate(noisySet, cleanTree)
print("\tEvaluating noisyTree on cleanSet...")
evaluate(cleanSet, noisyTree)
print("\tEvaluating noisyTree on noisySet...")
evaluate(noisySet, noisyTree)

# STEP 4 - PRUNING (AND EVALUATION AGAIN)
print("\n----- 4 - PRUNING -----")
print("\tPruning  and evaluating cleanTree...")
crossValidate(cleanSet)
print("\tPruning  and evaluating noisyTree...")
crossValidate(noisySet)

