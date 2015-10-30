# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 15:35:28 2015

@author: lichen
"""

import numpy as np
import pandas as pd
import pylab as plt

train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
