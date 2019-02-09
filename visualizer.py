# Generate a structured string representing the tree and print it to std:out
def visualizeTree(tree):
    treeLines = treeToLines(tree, 0, False, '-')[0]
    print('\n' + '\n'.join((line.rstrip() for line in treeLines)))


# Recursively generate a list of lines representing the tree
def treeToLines(rootNode, curr_index, index=False, delimiter='-'):
    
    ##################################
    ######## HELPER FUNCTIONS ########
    def value(root): # Get value of current node
        param = root["attribute"]
        val = root["value"]
        if ((param == "") and (val == "")): # If leaf node
            return str(int(root["leaf"]))
        else:
            return (str(param) + "<" + "(" + str(val) + ")") # If decision node

    def left(root): # Get left subtree
        if root["left"] != "":
            return root["left"]
        else:
            return None

    def right(root): # Get right subtree
        if root["right"] != "":
            return root["right"]
        else:
            return None
    ##################################
    
    # Recursion base case --> no tree
    if rootNode is None:
        return [], 0, 0, 0

    # Initialize strings to hold printable data
    line1 = []
    line2 = []
    # Get representation of rootNode
    if index:
        node_repr = '{}{}{}'.format(curr_index, delimiter, value(rootNode))
    else:
        node_repr = str(value(rootNode))

    new_root_width = gap_size = len(node_repr)

    # Call recursively function on left and right subtrees (boxes) to get their string representations
    (leftSubBox, leftbox_width, leftroot_start, leftroot_end) = treeToLines(left(rootNode), 2 * curr_index + 1, index, delimiter)
    (rightSubBox, rightbox_width, rightroot_start, rightroot_end) = treeToLines(right(rootNode), 2 * curr_index + 2, index, delimiter)

    # Add representation of left subtree to strings
    if leftbox_width > 0:
        leftroot = (leftroot_start + leftroot_end) // 2 + 1
        line1.append(' ' * (leftroot + 1))
        line1.append('_' * (leftbox_width - leftroot))
        line2.append(' ' * leftroot + '/')
        line2.append(' ' * (leftbox_width - leftroot))
        new_root_start = leftbox_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Add representation of rootNode to strings
    line1.append(node_repr)
    line2.append(' ' * new_root_width)

    # Add representation of right subtree to strings
    if rightbox_width > 0:
        rightroot = (rightroot_start + rightroot_end) // 2
        line1.append('_' * rightroot)
        line1.append(' ' * (rightbox_width - rightroot + 1))
        line2.append(' ' * rightroot + '\\')
        line2.append(' ' * (rightbox_width - rightroot))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-representations to rootNode structure
    gap = ' ' * gap_size
    new_box = [''.join(line1), ''.join(line2)]
    for i in range(max(len(leftSubBox), len(rightSubBox))):
        if i < len(leftSubBox):
            leftline = leftSubBox[i]  
        else:
            leftline = ' ' * leftbox_width
        if i < len(rightSubBox):
            rightline = rightSubBox[i]
        else:
            rightline = ' ' * rightbox_width
        new_box.append(leftline + gap + rightline)

    # Return 4-tuple containing
    # (the new box | its width | its root repr positions (start | end))
    return (new_box, len(new_box[0]), new_root_start, new_root_end)
    