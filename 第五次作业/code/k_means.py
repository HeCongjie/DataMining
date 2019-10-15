#!/usr/bin/env python
import pandas as pd
import numpy as np


def loadData():
    data = pd.read_csv('../Kmeans数据集/Sales_Transactions_Dataset_Weekly.csv')
    newData = data[data.columns[55:]]
    labelData = pd.DataFrame(data['Product_Code'])
    #new_data = pd.concat([data[data.columns[55:]],data['Product_Code']],axis=1)
    return newData, labelData

def distEclud(vecA, vecB):
    return np.sqrt(sum(np.power(list(map(lambda x: x[0]-x[1], zip(vecA, vecB))), 2))) #la.norm(vecA-vecB)

def randCent(dataSet, k):
    m = dataSet.shape[1]
    centroids = np.mat(np.random.random((k,m)))#create centroid mat
    return centroids

def kMeans(dataSet, labelData, k, distMeas=distEclud, createCent=randCent):
    m = dataSet.shape[0] #样本数
    clusterAssment = np.mat(np.zeros((m,2))) #m*2的矩阵
    clusterAssment = pd.DataFrame(clusterAssment)
    clusterAssment.columns = ['point','dist']
    centroids = createCent(dataSet, k) #初始化k个中心
    clusterChanged = True
    while clusterChanged:      #当聚类不再变化
        clusterChanged = False
        for i in range(m):
            minDist = np.inf; minLabel = -1
            for j in range(k): #找到最近的质心
                distJI = distMeas(centroids[j,:].tolist()[0],dataSet.loc[i].tolist())
                if distJI < minDist:
                    minDist = distJI
                    minLabel = j
            if clusterAssment.iloc[i,0] != minLabel:
                clusterChanged = True
            # 第1列为所属质心，第2列为距离
            clusterAssment.loc[i,:] = minLabel,minDist**2
        print(centroids)
        # 更改质心位置
        for cent in range(k):
            ptsInClust = dataSet[clusterAssment['point']==cent]
            centroids[cent,:] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment



if __name__ == '__main__':
    k = 10
    dataSet, labelData = loadData()
    centroids, clusterAssment = kMeans(dataSet, labelData, k, distMeas=distEclud, createCent=randCent)
    print(clusterAssment)
