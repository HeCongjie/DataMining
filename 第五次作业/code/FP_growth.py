#!/usr/bin/env python

import pandas as pd
import numpy as np
from collections import Counter
import warnings

warnings.filterwarnings('ignore')
class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode      #needs to be updated
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind=1):
        print('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

def createTree(dataSet, minSup=1): #create FP-tree from dataset but don't mine
    #numCounter = {}
    headerTable = {}
    #go over dataSet twice
    for trans in dataSet:#first pass counts frequency of occurance
        headerTable = dict(Counter(headerTable) + Counter(trans))
    #headerTableCopy = headerTable.copy()

    freqItemCounter = {k: v for k, v in headerTable.items() if v >= minSup}

    freqItemSet = set(freqItemCounter.keys())

    #print 'freqItemSet: ',freqItemSet

    if len(freqItemSet) == 0: return None, None  #if no items meet min support -->get out

    for k in headerTable:
        headerTable[k] = [headerTable[k], None] #reformat headerTable to use Node link
    #print 'headerTable: ',headerTable
    retTree = treeNode('Null Set', 1, None) #create tree
    for tranSet, count in dataSet.items():  #go through dataset 2nd time
        localD = {}
        for item in tranSet:  #put transaction items in order
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)#populate tree with ordered freq itemset
    return retTree, headerTable #return tree and header table

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:#check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count) #incrament count
    else:   #add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: #update header table
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:#call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):   #this version does not use recursion
    while (nodeToTest.nodeLink != None):    #Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def ascendTree(leafNode, prefixPath): #ascends from leaf node to root
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode): #treeNode comes from header table
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def loadData():
    data = pd.read_csv('../FP_data/retail.csv',sep=',',error_bad_lines=False)
    data = data.values
    new_data = []
    for i in range(np.shape(data)[0]):
        new_data.append(data[i][~np.isnan(data[i])])
    return new_data

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        if frozenset(trans) in retDict.keys():
            retDict[frozenset(trans)] = retDict[frozenset(trans)] + 1
        else:
            retDict[frozenset(trans)] = 1
    return retDict

if __name__ == '__main__':
    minSup = 3
    data = loadData()
    initSet = createInitSet(data)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    myFPtree.disp()
    print('Well done')