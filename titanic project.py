# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 15:57:09 2015

@author: lichen
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


### Import data from file with pandas

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

### Cleaning data by remove the columns having too much missing values 
### and then removes the NaN values from every reminding feature. 

train = train.drop(['Ticket', 'Cabin'], axis = 1)
train = train.dropna()

# specifies the parameters of graphs
fig = plt.figure(figsize = (18, 8), dpi = 2000)

ax1 = plt.subplot2grid((2,2),(0,0))
data1 = train.Survived.value_counts()
data1.plot(kind = "bar", color = "violet")
plt.title("Survival Breakdowm (1 = Survived, 0 = Died)")

ax2 = plt.subplot2grid((2,2), (0,1))
plt.scatter(train.Survived, train.Age, alpha = 0.3, color = "purple" )
plt.ylabel("Age")
plt.grid(True, "Major", axis = 'y')
plt.title("Survived by Age (1 = Survived, 0 = Died)")

ax3 = plt.subplot2grid((2,2), (1,0))
data3 = train.Pclass.value_counts()
data3.plot(kind = "bar", alpha = 0.7, color = "lightblue")
plt.title("Class Distribution")

ax4 = plt.subplot2grid((2,2), (1,1))
train.Age[train.Pclass == 1].plot(kind = "kde")
train.Age[train.Pclass == 2].plot(kind = "kde")
train.Age[train.Pclass == 3].plot(kind = "kde")
plt.xlabel("Age")
plt.title("Age distribution with Classes")
plt.legend(("1st Class", "2nd Class", "3rd Class"))

## Exploratory Visualization

fig = plt.figure(figsize = (21, 6))

ax1 = fig.add_subplot(2,4,1)
train.Survived.value_counts().plot(kind = "bar", color = "violet")
plt.title("Survival Breakdowm (1 = Survived, 0 = Died)")

ax2 = fig.add_subplot(2,4,2)
train.Survived[train.Sex == "male"].value_counts().plot(kind = "bar", color = "lightblue", label = "Male")
train.Survived[train.Sex == "female"].value_counts().plot(kind = "bar", color = "lightpink", label = "Female")
plt.title("Survived within gender")
plt.legend(loc = "best")

