#!/usr/bin/env python

import pandas as pd
import numpy as np

def
if __name__ == '__main__':
    data = pd.read_csv('../第4次作业/SVM数据集/post-operative.data',header=None)
    data.columns = ['L-CORE','L-SURF','L-O2','L-BP','SURF-STBL','CORE-STBL','BP-STBL','COMFORT','label']
    data.loc[data['label']=='I', 'label'] = 'S'
    data = data[data['COMFORT']!='?'].reset_index(drop=True)

    X = data.drop('label',axis=1)
    Y = pd.DataFrame(data['label'])

    data_pre = data.sample(n=10).reset_index(drop=True)
    X_pre = data_pre.drop('label',axis=1)
    Y_true = pd.DataFrame(data_pre['label'])