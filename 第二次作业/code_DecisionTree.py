#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 21:12:21 2018

@author: world
"""
import pandas as pd


from sklearn.metrics import accuracy_score

from math import log

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

'''
data_pre = data.sample(n=100)  #随机选取100个实例检测预测准确性
Y = data.pop('label')
X = data  #生成训练数据集
Y_true = data_pre.pop('label')
Y_true = Y_true.reset_index(drop=True)  #生成检测数据集
X_pre = data_pre
'''
def splitDataSet(data, key, fea_value):
    retDataSet = data[data[key]==fea_value]
    return retDataSet

def calcShannonEnt(data):
    label_size = data.groupby('label').size().reset_index(name='Size')
    numEntires = data.shape[0]
    shannonEnt = 0.0
    for key in label_size['label'].tolist():
        prob = float(label_size[label_size['label']==key].values[0][1]) / numEntires
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def chooseBestFeatureToSplit(data):
    data_XY = data
    X_data = data.drop('label',axis=1)
    baseEntropy = calcShannonEnt(data_XY)
    bestInfoGain = 0.0
    bestFeature = -1
    columns_list = X_data.columns.tolist()
    for key in columns_list:
        newEntropy = 0.0
        fea_size = X_data.groupby(key).size().reset_index(name='Size')
        fea_value_list = fea_size[key].tolist()
        for fea_value in fea_value_list:
            subDataSet = splitDataSet(data_XY, key, fea_value)
            prob = subDataSet.shape[0] / float(data_XY.shape[0])
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if (infoGain >= bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = key
    return bestFeature

def majorityCnt(data):
    temp = data.groupby('label').size().reset_index(name='Size')
    maxCountClass = temp[temp['Size']==temp['Size'].max()].values[0][0]
    return maxCountClass

def createTree(data_XY, featLabels):
    data = data_XY
    if data.shape[1]==2 or data.groupby('label').size().reset_index(name='Size').values[0][1] == data.shape[0]:
        return majorityCnt(data)
    bestFeat = chooseBestFeatureToSplit(data)
    featLabels.append(bestFeat)
    myTree = {bestFeat:{}}
    featValues = data.groupby(bestFeat).size().reset_index(name='Size')[bestFeat].tolist()
    for value in featValues:
        myTree[bestFeat][value] = createTree(splitDataSet(data, bestFeat, value).drop(bestFeat,axis=1), featLabels)
    return myTree

def getNumLeafs(myTree):
    numLeafs = 0												#初始化叶子
    firstStr = next(iter(myTree))								#python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]								#获取下一组字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':				#测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs +=1
    return numLeafs

def getTreeDepth(myTree):
    maxDepth = 0												#初始化决策树深度
    firstStr = next(iter(myTree))								#python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    secondDict = myTree[firstStr]								#获取下一个字典
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':				#测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth			#更新层数
    return maxDepth

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	arrow_args = dict(arrowstyle="<-")											#定义箭头格式
	font = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc", size=14)		#设置中文字体
	createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',	#绘制结点
		xytext=centerPt, textcoords='axes fraction',
		va="center", ha="center", bbox=nodeType, arrowprops=arrow_args, FontProperties=font)

def plotMidText(cntrPt, parentPt, txtString):
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]											#计算标注位置
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)

def plotTree(myTree, parentPt, nodeTxt):
	decisionNode = dict(boxstyle="sawtooth", fc="0.8")										#设置结点格式
	leafNode = dict(boxstyle="round4", fc="0.8")											#设置叶结点格式
	numLeafs = getNumLeafs(myTree)  														#获取决策树叶结点数目，决定了树的宽度
	depth = getTreeDepth(myTree)															#获取决策树层数
	firstStr = next(iter(myTree))															#下个字典
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)	#中心位置
	plotMidText(cntrPt, parentPt, nodeTxt)													#标注有向边属性值
	plotNode(firstStr, cntrPt, parentPt, decisionNode)										#绘制结点
	secondDict = myTree[firstStr]															#下一个字典，也就是继续绘制子结点
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD										#y偏移
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':											#测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
			plotTree(secondDict[key],cntrPt,str(key))        								#不是叶结点，递归调用继续绘制
		else:																				#如果是叶结点，绘制叶结点，并标注有向边属性值
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')													#创建fig
    fig.clf()																				#清空fig
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)    							#去掉x、y轴
    plotTree.totalW = float(getNumLeafs(inTree))											#获取决策树叶结点数目
    plotTree.totalD = float(getTreeDepth(inTree))											#获取决策树层数
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;								#x偏移
    plotTree(inTree, (0.5,1.0), '')															#绘制决策树
    plt.show()

def classify(inputTree, featLabels, testVec):
	firstStr = next(iter(inputTree))														#获取决策树结点
	secondDict = inputTree[firstStr]														#下一个字典
	featIndex = featLabels.index(firstStr)
	for key in secondDict.keys():
		if testVec[featIndex] == key:
			if type(secondDict[key]).__name__ == 'dict':
				classLabel = classify(secondDict[key], featLabels, testVec)
			else: classLabel = secondDict[key]
	return classLabel																				#显示绘制结果

if __name__ == '__main__':
    data = pd.read_fwf('./第2次作业/Page_Blocks_Classification_Data_Set/page-blocks.data',header=None)
    data.columns=["height","lenght","area","eccen","p_black","p_and","mean_tr","blackpix","blackand","wb_trans","label"]  #加载数据
    '''
    Y = data.pop('label')
    Y = pd.DataFrame(Y)
    X = data
    '''
    featLabels = []
    myTree = createTree(data, featLabels)
    #testVec = [6,18,108,3.0,0.287,0.741,4.43,31,80,7]
    #result = classify(myTree, featLabels, testVec)
    data_pre = data.sample(n=100)
    data_pre = data_pre.reset_index(drop=True)
    X_pre = data_pre.drop('label',axis=1)
    Y_true = data_pre['label']
    X_list = X_pre.values
    Y_pre = []
    fea_list = ["height","lenght","area","eccen","p_black","p_and","mean_tr","blackpix","blackand","wb_trans"]
    for testVec in X_list:
        Y_pre.append(classify(myTree, fea_list, testVec))
    Y_pre = pd.Series(Y_pre)
    print('准确率： ',accuracy_score(Y_true,Y_pre))
    createPlot(myTree)

