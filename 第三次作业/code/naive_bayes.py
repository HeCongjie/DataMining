#!/usr/bin/env python

import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score

def naiveBayesTrain(X,Y,data):
    lambd = 1
    dictY = Y.groupby(['label']).size().reset_index(name='Size')
    labelCount = dictY.shape[0]
    labelSize = dictY.sum().values[1]
    dictY['probability'] = dictY['Size'].apply(lambda x: (x + lambd) / (labelSize + labelCount * lambd))
    featureDict = {}
    for key in X.columns.tolist():
        temp = data.groupby([key,'label']).size().reset_index(name='Size').pivot(key,'label','Size').fillna(0)
        for label_key in temp.columns.tolist():
            temp[label_key] = temp[label_key].apply(lambda x: (x + lambd) / (temp[label_key].sum() + temp.shape[0] * lambd))
        featureDict[key] = temp
    return featureDict,dictY

def naiveBayesPredict(X_pre,featureDict,labelList,dictY):
    Y_pre = []
    columnsList = X_pre.columns.tolist()
    for i in range(X_pre.shape[0]):
        resultList = []
        feaList = X_pre.iloc[i].reset_index(name='feaValues').rename(columns={'index':'feaName'})
        for labelKey in labelList:
            temProbability = 0
            for columnkey in columnsList:
                feaValue = feaList[feaList['feaName'] == columnkey].values[0][1]
                temProbability += math.log(featureDict[columnkey].loc[feaValue][labelKey])
            resultList.append(temProbability + math.log(dictY[dictY['label'] == labelKey]['probability'].values[0]))
        Y_pre.append(labelList[resultList.index(max(resultList))])
    return pd.DataFrame(Y_pre,columns=['label'])

if __name__ == '__main__':
    data = pd.read_csv('/Users/world/课程资料/数据分析工具实践/第三次作业/第3次作业/贝叶斯/数据/nursery.data',header=None)
    data.columns=['parents','has_nurs','form','children','housing','finance','social','health','label']
    dataPre = data.sample(n=100).reset_index(drop=True)
    X = data.drop(['label'],axis=1)
    Y = pd.DataFrame(data['label'])
    labelList = Y.groupby(['label']).count().index.tolist()
    featureDict, dictY = naiveBayesTrain(X,Y,data)
    X_pre = dataPre.drop(['label'],axis=1)
    Y_pre= naiveBayesPredict(X_pre,featureDict,labelList,dictY)
    Y_true = pd.DataFrame(dataPre['label'])
    print("分类准确率： ", accuracy_score(Y_true, Y_pre))