#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 15:42:04 2018

@author: world
"""

import pandas as pd

data = pd.read_csv('./第10次作业/data/semeion.data', sep=' ', header=None)
dataVali = data.tail(200)
data.drop(data.tail(200).index,inplace=True)
x = data.iloc[:,0:256]
y = data.iloc[:,256:266]