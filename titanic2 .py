# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 14:17:05 2015

@author: lichen
"""

import numpy as np
import pandas as pd
from patsy import dmatrices, dmatrix
import statsmodels.api as sm

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, Imputer
from sklearn.calibration import CalibratedClassifierCV

train = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("test.csv", dtype={"Age": np.float64}, )

train = train.drop(['Ticket', 'Cabin'], axis = 1)
train = train.dropna()

test = test.drop(['Ticket', 'Cabin'], axis = 1)
test = test.dropna()

train.loc[train["Age"].isnull(), "Age"]= np.nanmedian(train["Age"])


y_train, x_train = dmatrices('Survived ~ Age + Sex + Pclass + SibSp + Parch + Embarked', train)

x_test = dmatrix('Age + Sex + Pclass + SibSp + Parch + Embarked', test)

steps1 = [('poly_features', PolynomialFeatures(3, interaction_only=True)),
          ('logistic', LogisticRegression(C=5555., max_iter=16, penalty='l2'))]
pipeline1 = Pipeline(steps=steps1)

steps2 = [('rforest', RandomForestClassifier(min_samples_split=15, n_estimators=73, criterion='entropy'))]
pipeline2 = Pipeline(steps=steps2)

pipeline1.fit(x_train, y_train.ravel())
print('Accuracy (Logistic Regression-Poly Features (cubic)): {:.4f}'.format(pipeline1.score(x_train, y_train.ravel())))
### Accuracy (Logistic Regression-Poly Features (cubic)): 0.8357

pipeline2.fit(x_train[:600], y_train[:600].ravel())
calibratedpipe2 = CalibratedClassifierCV(pipeline2, cv=3, method='sigmoid')
calibratedpipe2.fit(x_train[600:], y_train[600:].ravel())
print('Accuracy (Random Forest - Calibration): {:.4f}'.format(calibratedpipe2.score(x_train, y_train.ravel())))
### Accuracy (Random Forest - Calibration): 0.7879

# Create the output dataframe
output = pd.DataFrame(columns=['PassengerId', 'Survived'])
output['PassengerId'] = test['PassengerId']

# Predict the survivors and output csv
output['Survived'] = pipeline1.predict(x_test).astype(int)
output.to_csv('output.csv', index=False)