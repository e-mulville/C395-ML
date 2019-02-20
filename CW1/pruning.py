import numpy as np
import math
import time
from decisiontree import build_decision_tree
from validation import evaluate

#made this so we can easily change how we evaluate for pruning
def get_evaluation_metric(set,tree):
    metrics = evaluate(set, tree)

    return metrics[4]

def prune(tree, validation_set):

    def recursive_prune(tree_root, current_node):

        left_node = current_node["left"]
        right_node = current_node["right"]

        #recurse through tree TODO
        if left_node["leaf"] == 0:
            recursive_prune(tree_root, left_node)
        if right_node["leaf"] == 0:
            recursive_prune(tree_root, right_node)

        #if both children are leaves
        if (left_node["leaf"] != 0) & (right_node["leaf"] != 0):


            no_change_performance = get_evaluation_metric(validation_set, tree_root)
            current_node["left"] = ""
            current_node["right"] = ""

            #try setting the current node to classify as the left
            current_node["leaf"] = left_node["leaf"]
            taking_left_performance = get_evaluation_metric(validation_set, tree_root)

            #try setting curretn node as right
            current_node["leaf"] = right_node["leaf"]
            taking_right_performance = get_evaluation_metric(validation_set, tree_root)

            if (taking_left_performance > no_change_performance) & (taking_left_performance >= taking_right_performance):
                current_node["leaf"] = left_node["leaf"]
                current_node["value"] = ""
                current_node["attribute"] = ""
            elif (taking_right_performance >= taking_left_performance) & (taking_right_performance > no_change_performance):
                current_node["leaf"] = right_node["leaf"]
                current_node["value"] = ""
                current_node["attribute"] = ""
            else:
                current_node["leaf"] = 0
                current_node["left"] = left_node
                current_node["right"] = right_node

    recursive_prune(tree, tree)
    #print("after pruning:")
    #print(pruning_depth(tree, 0))
    return tree

def pruning_depth(this_node,depth):
    if this_node["leaf"] > 0:
        return(depth)
    else:
        left_depth = pruning_depth(this_node["left"], depth + 1)
        right_depth = pruning_depth(this_node["right"], depth + 1)
        return max(left_depth, right_depth)
