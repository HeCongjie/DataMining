#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

def loadData():
    data_mat = pd.read_csv('../第4次作业/Adaboost数据集更新/student-mat.csv',sep=';')
    data_por = pd.read_csv('../第4次作业/Adaboost数据集更新/student-por.csv',sep=';')
    data_all = pd.concat([data_mat,data_por], axis=0).reset_index(drop=True)
    labelEncoderColumns = ['school','sex','address','famsize','Pstatus','Mjob','Fjob','reason','guardian','schoolsup','famsup','paid','activities','nursery','higher','internet','romantic']
    le = LabelEncoder()
    for key in labelEncoderColumns:
        data_all[key] = le.fit_transform(data_all[key])
    data_pre = data_all.sample(n=100).reset_index(drop=True)

    Y = pd.DataFrame(data_all['G3'].apply(lambda x: (-1.0, 1.0)[x>=10]))
    Y.columns = ['label']
    X = data_all.drop('G3', axis=1)

    X_pre = data_pre.drop('G3', axis=1)
    Y_true = pd.DataFrame(data_pre['G3'].apply(lambda x: (-1.0, 1.0)[x>=10]))
    Y_true.columns = ['G3_true']

    return X,Y,X_pre,Y_true


def buildStump(X,Y,weight):
    m,n = X.shape
    columnsList = X.columns.tolist()
    minError = float('inf')
    numSteps = 10.0
    bestStump = {}
    for columnsKey in columnsList:
        rangeMin = X[columnsKey].min()
        rangeMax = X[columnsKey].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            threshVal = (rangeMin + float(j) * stepSize)
            for tryValue in ['lt', 'gt']:
                errArr = np.mat(np.ones((m,1)))
                if tryValue == 'lt':
                    predictedVals = X[columnsKey].apply(lambda x: (-1.0, 1.0)[x<=threshVal])
                else:
                    predictedVals = X[columnsKey].apply(lambda x: (1.0, -1.0)[x<=threshVal])
                errArr[predictedVals == Y['label']] = 0
                weightedError = (weight.T * errArr).tolist()[0][0]
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = columnsKey
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = tryValue
    return bestStump, minError, np.mat(bestClasEst)

def adaBoostTrainDS(X, Y, numIt = 40):
    weakClassArr = []
    m = X.shape[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    D = np.mat(np.ones((m, 1)) / m)
    for i in range(numIt):
        bestStump, error, classEst = buildStump(X, Y, D)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expon = np.multiply(-1 * alpha * np.mat(Y).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += (alpha * classEst).T
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(Y).T, np.ones((m,1)))
        errorRate = aggErrors.sum() / m
        if errorRate == 0.0: break
    return weakClassArr, aggClassEst

def adaClassify(X_pre,classifierArr):
    #dataMatrix = np.mat(X_pre)
    m = X_pre.shape[0]
    aggClassEst = np.mat(np.zeros((m,1)))
    for i in range(len(classifierArr)):
        if classifierArr[i]['ineq'] == 'lt':
            classEst = X_pre[classifierArr[i]['dim']].apply(lambda x: (-1.0, 1.0)[x<=classifierArr[i]['thresh']])
        else:
            classEst = X_pre[classifierArr[i]['dim']].apply(lambda x: (1.0, -1.0)[x<=classifierArr[i]['thresh']])
        aggClassEst += np.mat(classifierArr[i]['alpha'] * classEst).T
    return np.sign(aggClassEst)

if __name__ == '__main__':
    X,Y,X_pre,Y_true = loadData()
    weakClassArr, aggClassEst = adaBoostTrainDS(X, Y)
    Y_pre = pd.DataFrame(adaClassify(X_pre, weakClassArr))
    Y_pre.columns=['G3_pre']
    print("分类准确率： ", accuracy_score(Y_true, Y_pre))
    print(pd.concat([Y_pre,Y_true],axis=1))

