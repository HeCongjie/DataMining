 #!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn import metrics

def sigmoid(inX):
	return 1.0 / (1 + np.exp(-inX))

def gradAscent(X, Y):
    dataMatrix = np.mat(X)
    labelMat = np.mat(Y).transpose()
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    maxCycles = 400
    weights = np.ones((n,1))
    weights_array = np.array([])
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = labelMat - h
        weights = weights + alpha * dataMatrix.transpose() * error
        weights_array = np.append(weights_array,weights)
    weights_array = weights_array.reshape(maxCycles,n)
    return weights.getA(),weights_array

if __name__ == '__main__':
     data = pd.read_excel("../第3次作业/回归算法/2014 and 2015 CSM dataset.xlsx")
     data = data.dropna(axis=0,how='any').reset_index(drop=True)
     dataPre = data.sample(n=50).reset_index(drop=True)
     X = data.drop(['Movie','Ratings'],axis=1)
     Y = data['Ratings'].apply(lambda x: (0,1)[x>5])
     weights,weights_array = gradAscent(X, Y)

     X_pre = dataPre.drop(['Movie','Ratings'],axis=1)
     Y_true = dataPre['Ratings'].apply(lambda x: (0,1)[x>5])
     Y_pre = sigmoid(np.dot(X_pre,weights)).tolist()
     Y_pre = pd.Series([x[0] for x in Y_pre])
     print('准确率： \n', metrics.f1_score(Y_true, Y_pre, average='weighted'))
     print("Well done!")


