#!/usr/bin/env python

import pandas as pd
import numpy as np

#测试用数据集
def loadExData():
    return np.mat([[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]])

def loadData():
    attribute = pd.read_csv('./attribute.data', header=None, names=['A', 'B', 'C', 'D', 'E']).reset_index(drop=True)
    xKey = attribute['B'].tolist()
    #xKey.sort()
    data = pd.read_csv('./anonymous-msweb.data',header=None,names=['A','B','C']).reset_index(drop=True)
    caseLineList = data[data['A']=='C'].index.tolist()
    n = max(xKey)-min(xKey)
    m = len(caseLineList)
    dataMatrix = np.zeros((m,n),dtype=int)
    for i in range(1, m):
        #caesNum = caseLineList[i]
        cStartPoint = caseLineList[i-1]
        cStopPoint = caseLineList[i]
        vPointList = data.iloc[cStartPoint+1 : cStopPoint]['B'].tolist()
        matrixRow = data.iloc[cStartPoint]['B'] - 10001
        for j in vPointList:
            dataMatrix[matrixRow][j - 1000] = 1
    return dataMatrix

def compuDiagMatrix(dataMatrix):
    tempMatrix = np.dot(dataMatrix.T, dataMatrix)
    eigenvalues, featureVector = np.linalg.eig(tempMatrix)
    totalSum = sum( eigenvalues )
    partSum = 0
    i = 0
    diagList = []
    while(partSum < totalSum*0.9):
        diagList.append(np.sqrt(eigenvalues[i]))
        partSum += eigenvalues[i]
        i = i + 1
    return np.diag(diagList), i



if __name__ == '__main__':
    #dataMatrix = loadData()
    dataMatrix = loadExData()
    resMatrix, num = compuDiagMatrix(dataMatrix)
    print('Well done!')