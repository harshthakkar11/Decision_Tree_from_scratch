# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import math


# clean the data
def clean_data(line):
    return line.replace('(', '').replace(')', '').replace(' ', '').strip().split(',')


# fetching the data textfile.
def fetch_data(filename):
    with open(filename, 'r') as f:
        input_data = f.readlines()
        clean_input = list(map(clean_data, input_data))
        f.close()
    return clean_input


# reading the data from the fi
def readFile(dataset_path):
    input_data = fetch_data(dataset_path)
    # input_np = np.array(input_data)
    input_np = list(input_data)
    return input_np


# converting string to float
def str_column_to_float(dataset, column):
    # Calling each row in list
    for row in dataset:
        row[column] = float(row[column].strip())


# Reading Train file
Q1_B_train = r'Q1_train.txt'

# Taking into list
train_np = readFile(Q1_B_train)

# Reading Test file
Q1_B_test = r'Q1_test.txt'

# Taking into list
test_np = readFile(Q1_B_test)

# creating class to make a tree
class Node:
    # Storing left node
    Left_subtree = None
    # storing right node
    Right_subtree = None

    def __init__(self, feature=None, SplitValue=None, Information_gain=None):
        self.feature = feature
        self.SplitValue = SplitValue

        self.Information_gain = Information_gain

# Calculating Entropy
def CalculateEntropy(Dataset):
    Mcount = 0
    Wcount = 0
    # Man and Woman count
    for i in range(len(Dataset)):
        if Dataset[i][3] == 'M':
            Mcount = Mcount + 1
        else:
            Wcount = Wcount + 1

    # Total count
    total = Mcount + Wcount
    entropy = 0


    if Mcount != 0 and Wcount != 0:
        entropy = float(
            ((-Mcount * math.log2(Mcount / total)) / total) + ((-Wcount * math.log2(Wcount / total)) / total))
    elif Mcount == 0 and Wcount != 0:
        entropy = float((-Wcount * math.log2(Wcount / total)) / total)
    elif Mcount != 0 and Wcount == 0:
        entropy = float((-Mcount * math.log2(Mcount / total)) / total)

    return entropy

# Creating each node in tree

class DecisionTree:

    def __init__(self, max_depth):
        self.max_depth = 5
        self.root = None

    # Calculating Information Gain
    def CalculateInformationGain(self, parent, leftSide, rightSide):

        # Parent node Entropy
        ParentEntropy = CalculateEntropy(parent)
        # Left Node Entropy
        leftEntropy = CalculateEntropy(leftSide)
        leftcount = len(leftSide)
        # Right Node Entropy
        rightEntropy = CalculateEntropy(rightSide)
        rightcount = len(rightSide)
        totalcount = leftcount + rightcount
        # Calculating Information Gain
        InformationGain = ParentEntropy - (
                ((leftcount / totalcount) * (leftEntropy)) + ((rightcount / totalcount) * (rightEntropy)))

        return InformationGain

    # Finding data to get a split in Decision Tree
    def MakingSplitEachPoint(self, dataset):
        DecisionSplit = {}
        HighInformationGain = -1
        for j in range(len(dataset[0]) - 1):
            # Sorting data based on each column
            dataset.sort(key=lambda x: x[j])
            for k in range(len(dataset) - 1):
                # Taking EachPoint here
                EachPoint = float(float(dataset[k][j]) + float(dataset[k + 1][j])) / 2
                left = []
                right = []
                for p in range(len(dataset)):
                    if dataset[p][j] < EachPoint:
                        left.append(dataset[p])
                    else:
                        right.append(dataset[p])

                # Finding Information Gain
                InformationGain = self.CalculateInformationGain(dataset, left, right)

                # Finding The Highest Information Gain
                if InformationGain > HighInformationGain:
                    # Storing best split value here
                    DecisionSplit = {'feature': j,
                                     'Eachpoint': EachPoint,
                                     'Left_subtree': left,
                                     'Right_subtree': right,
                                     'Information_gain': InformationGain
                                     }
                    HighInformationGain = InformationGain

        return DecisionSplit

    # Calling Split recursively to find split for child node until we reach Maximum Depth
    def CreateTreeWithDepthRecursive(self, currentDepth, Dataset, maxDepth, current):

        # checking it does not go over Maximum Depth
        if currentDepth < maxDepth:
            split = self.MakingSplitEachPoint(dataset=Dataset)
            if len(split) > 0:
                if current == None:
                    self.root = Node(split.get('feature'), split.get('Eachpoint'), split.get('Information_gain'))
                    current = self.root
                # Getting split for left node
                if len(split.get('Left_subtree')) > 0:
                    if len(split.get('Left_subtree')) == 1:
                        current.Left_subtree = Node(2, split.get('Left_subtree')[0][2],
                                                    1)
                    else:
                        splitLeft = self.MakingSplitEachPoint(split.get('Left_subtree'))
                        current.Left_subtree = Node(splitLeft.get('feature'), splitLeft.get('Eachpoint'),
                                                    splitLeft.get('Information_gain'))
                        self.CreateTreeWithDepthRecursive(currentDepth + 1, splitLeft.get('Left_subtree'), maxDepth,
                                                          current.Left_subtree)
                # Getting split for right node
                if len(split.get('Right_subtree')) > 0:
                    if len(split.get('Right_subtree')) == 1:
                        current.Right_subtree = Node(2, split.get('Right_subtree')[0][2], 1)
                    else:
                        splitRight = self.MakingSplitEachPoint(split.get('Right_subtree'))
                        current.Right_subtree = Node(splitRight.get('feature'), splitRight.get('Eachpoint'),
                                                     splitRight.get('Information_gain'))
                        self.CreateTreeWithDepthRecursive(currentDepth + 1, splitRight.get('Right_subtree'), maxDepth,
                                                          current.Right_subtree)

            # If list is empty returning none for that node
            else:
                if current != None:
                    if current.Left_subtree == None:
                        current.Left_subtree = Node(2, Dataset[0][2], 1)
                    elif current.Right_subtree == None:
                        current.Right_subtree = Node(2, Dataset[0][2], 1)

    # To call split recursively as per our Maximum depth
    def SplitNode(self, Dataset, Max_depth):
        self.CreateTreeWithDepthRecursive(0, Dataset, Max_depth, self.root)
        return self.root

def main(dataset, Maximum_depth):
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    for i in range(1, Maximum_depth + 1):
        print("Depth:", i)
        Tree = DecisionTree(max_depth=i)
        root = Tree.SplitNode(dataset, i)
        rightPrediction = 0
        for x in train_np:
            predictionFromAlgo = makePrediction(root, dataset, x)
            if (predictionFromAlgo == "M" and x[3] == "M") or (predictionFromAlgo == "W" and x[3] == "W"):
                rightPrediction = rightPrediction + 1
        rightPredictionTest = 0
        for x in test_np:
            predictionFromAlgo = makePrediction(root, dataset, x)
            if (predictionFromAlgo == "M" and x[3] == "M") or (predictionFromAlgo == "W" and x[3] == "W"):
                rightPredictionTest = rightPredictionTest + 1
        print(f"Accuracy Train={rightPrediction / len(train_np)} | Test={rightPredictionTest / len(test_np)}")


def makePrediction(root, dataset, dataToBeTested):
    stack = list()
    stack.append(root)
    while len(stack) > 0:
        node = stack.pop(0)
        left = list()
        right = list()
        for i in range(len(dataset)):
            if float(dataset[i][getattr(node, 'feature')]) < node.SplitValue:
                left.append(dataset[i])
            else:
                right.append(dataset[i])
        if float(dataToBeTested[getattr(node, 'feature')]) < node.SplitValue:
            dataset = left
            if node.Left_subtree != None:
                stack.append(node.Left_subtree)
        else:
            dataset = right
            if node.Right_subtree != None:
                stack.append(node.Right_subtree)
    mCount = 0
    wCount = 0
    for x in dataset:
        if x[3] == "M":
            mCount = mCount + 1
        else:
            wCount = wCount + 1

    if mCount > wCount:
        return "M"
    else:
        return "W"

main(train_np, 5)



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
