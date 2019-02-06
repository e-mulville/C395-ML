import numpy as np
import math
import time
from decisiontree import build_decision_tree
from validation import evaluate

def get_evaluation_metric(set,tree):
    metrics = evaluate(set, tree)

    return metrics[4]

def prune(tree, validation_set):

    def recursive_prune(tree_root, current_node, best_performance):

        left_node = current_node["left"]
        right_node = current_node["right"]

        #recurse through tree TODO
        if left_node["leaf"] == 0:
            recursive_prune(tree_root, left_node,best_performance)
        if right_node["leaf"] == 0:
            recursive_prune(tree_root, right_node,best_performance)

        #if both children are leaves
        if (left_node["leaf"] != 0) & (right_node["leaf"] != 0):
            current_node["left"] = ""
            current_node["right"] = ""

            #try setting the current node to classify as the left
            current_node["leaf"] = left_node["leaf"]
            if get_evaluation_metric(validation_set, tree_root) > best_performance:
                best_performance = test(tree_root, validation_set)

                #then try right
                current_node["leaf"] = right_node["leaf"]
                if get_evaluation_metric(validation_set, tree_root) > best_performance:
                    best_performance = test(tree_root, validation_set)
                else:
                    current_node["leaf"] = right_node["leaf"]

            else:
                #try right if the left wasnt better
                current_node["leaf"] = right_node["leaf"]
                if get_evaluation_metric(validation_set, tree_root) > best_performance:
                    best_performance = get_evaluation_metric(validation_set, tree_root)
                else:
                    #this is the case when the prune doesnt improve.
                    current_node["leaf"] = 0
                    current_node["left"] = left_node
                    current_node["right"] = right_node


    #test against validation_set

    best_performance = get_evaluation_metric(validation_set, tree)
    print(best_performance)

    recursive_prune(tree, tree, best_performance)

    return tree
