import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

def autoNormal(dataSet):
    X = dataSet.drop('lettr', axis=1)
    Y = pd.DataFrame(dataSet['lettr'])
    columnsList = X.columns.tolist()
    for key in columnsList:
        ranges = X[key].max() - X[key].min()
        X[key] = X[key] / ranges
    dataRes = pd.concat([X,Y], axis=1)
    return dataRes

def distance(dataSet):
    X = dataSet.drop('lettr', axis=1)
    length = X.shape[0]
    disMatrix = np.zeros((length,length))
    for i in range(length):
        for j in range(length):
            disMatrix[i][j] = np.linalg.norm(np.array(X.iloc[i]) - np.array(X.iloc[j]))
    return disMatrix

def findNNearestClassify(data, dataPre, disMatrix):
    indexList = dataPre.index.tolist()
    Y_pre = []
    for key in indexList:
        nums = disMatrix[key]
        temp=[]
        Inf = float('inf')
        for i in range(10):
            temp.append(nums.index(min(nums)))
            nums[nums.index(min(nums))]=Inf
        preValue = data.iloc[temp].groupby('lettr').size().reset_index(name='Size').sort_values(by='Size').values[0][0]
        Y_pre.append(preValue)
    return Y_pre

if __name__ == '__main__':
    data = pd.read_csv('/Users/world/课程资料/数据分析工具实践/第二次作业/第2次作业/letter_Recognition_Datasets/letter-recognition.data',header=None)
    data.columns = ['lettr','x-box','y-box','width','high','onpix','x-bar','y-bar','x2bar','y2bar','xybar','x2ybr','xy2br','x-ege','xegvy','y-ege','yegvx']
    le = LabelEncoder()
    data['lettr'] = le.fit_transform(data['lettr'])
    data = autoNormal(data)
    disMatrix = distance(data)
    dataPre = data.sample(n=100)
    Y_pre = pd.Series(findNNearestClassify(data, dataPre, disMatrix))
    Y_true = dataPre['lettr']
    print('准确率： ', metrics.f1_score(Y_true, Y_pre, average='weighted'))