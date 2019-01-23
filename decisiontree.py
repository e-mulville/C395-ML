import numpy as np
import math
import time

#checks if all of the data in a node has the same label
def all_same_label(set):

    first_label = set[0][7]

    for i in set:
        if i[7] != first_label:
            return 0

    return first_label



#information gain for a split
def calculate_gain(all_set,left_set,right_set):


    def calculate_h(set):

        label_count = [0.0,0.0,0.0,0.0]

        for j in set:
            label_count[int(j[7]) - 1] += 1

        h = 0

        for i in range(0,4):
            #pk * log2pk
            if (label_count[i] != 0) & (len(set) != 0):
                h -= (label_count[i]/len(set))*math.log(label_count[i]/len(set),2)
        return h

    remainder = (float( len(left_set) ) / float( len(all_set) ) * calculate_h(left_set))
    remainder += (float( len(right_set) ) / float( len(all_set) ) * calculate_h(right_set))

    return calculate_h(all_set) - remainder




def count_number_of_labels(set):


    labels_present = []

    for i in set:
        labels_present.append(i[7])

    return len(set(labels_present))

#splits the data set
def find_split(training_set):

    best_split = {
        "attribute": 0,
        "value": 0,
        "gain": 0
    }
    #for each attribute
    for attribute in range(0,7):

        set = training_set[training_set[:,attribute].argsort()]

        previous_label = set[0][7]





        for i in set:

            #if the label changes
            if i[7] != previous_label:
                value = i[attribute]
                left_set = []
                right_set = []
                #split there
                for j in set:

                    if j[attribute] < value:
                        left_set.append(j)
                    else:
                        right_set.append(j)

                split_gain = calculate_gain(training_set,left_set,right_set)

                if split_gain > best_split["gain"]:
                    best_split["attribute"] = attribute
                    best_split["value"] = value
                    best_split["gain"] = split_gain

            previous_label = i[7]





    return (best_split["attribute"],best_split["value"])

#builds the tree of nodes
def decision_tree_learning(training_set, depth):
    print depth
    this_node = {
        "attribute": "",
        "value": "",
        "left": "",
        "right": "",
        "leaf" : 0
    }

    x = all_same_label(training_set)
    if x > 0:
        this_node["leaf"] = x
        return (this_node,depth)
    else:
        (attribute, value) = find_split(training_set)

        this_node["attribute"] = attribute
        this_node["value"] = value

        left_set = []
        right_set = []

        for i in training_set:
            if i[attribute] < value:
                left_set.append(i)

            else:
                right_set.append(i)

        left_set =  np.array(left_set)
        right_set = np.array(right_set)

        (this_node["left"],left_depth) = decision_tree_learning(left_set, depth+1)
        (this_node["right"],right_depth) = decision_tree_learning(right_set,depth+1)

        return (this_node,max(left_depth, right_depth))





set = np.loadtxt('co395-cbc-dt/wifi_db/clean_dataset.txt')


print decision_tree_learning(set, 0)
