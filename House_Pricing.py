# -*- coding: utf-8 -*-
"""
Created on Mon May 07 18:40:32 2018

@author: binni
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset = pd.read_csv('train.csv')

#ploting categrical features
dataset['MSZoning'].value_counts().plot('bar')
dataset['Street'].value_counts().plot('bar')
dataset['Alley'].value_counts().plot('bar')
dataset['LotShape'].value_counts().plot('bar')
dataset['LandContour'].value_counts().plot('bar')
dataset['Utilities'].value_counts().plot('bar')
dataset['LotConfig'].value_counts().plot('bar')
dataset['LandSlope'].value_counts().plot('bar')
dataset['Neighborhood'].value_counts().plot('bar')
dataset['Condition1'].value_counts().plot('bar')
dataset['Condition2'].value_counts().plot('bar')
dataset['BldgType'].value_counts().plot('bar')
dataset['HouseStyle'].value_counts().plot('bar')
dataset['RoofStyle'].value_counts().plot('bar')
dataset['RoofMatl'].value_counts().plot('bar')
dataset['Exterior1st'].value_counts().plot('bar')
dataset['MasVnrType'].value_counts().plot('bar')
dataset['ExterQual'].value_counts().plot('bar')
dataset['ExterCond'].value_counts().plot('bar')
dataset['Foundation'].value_counts().plot('bar')
dataset['BsmtQual'].value_counts().plot('bar')
dataset['BsmtCond'].value_counts().plot('bar')
dataset['BsmtExposure'].value_counts().plot('bar')
dataset['BsmtFinType1'].value_counts().plot('bar')
dataset['Heating'].value_counts().plot('bar')
dataset['HeatingQC'].value_counts().plot('bar')
dataset['CentralAir'].value_counts().plot('bar')
dataset['Electrical'].value_counts().plot('bar')
dataset['KitchenQual'].value_counts().plot('bar')
dataset['Functional'].value_counts().plot('bar')
dataset['FireplaceQu'].value_counts().plot('bar')
dataset['GarageType'].value_counts().plot('bar')
dataset['GarageFinish'].value_counts().plot('bar')
dataset['GarageQual'].value_counts().plot('bar')
dataset['GarageCond'].value_counts().plot('bar')
dataset['PavedDrive'].value_counts().plot('bar')
dataset['PoolQC'].value_counts().plot('bar')
dataset['Fence'].value_counts().plot('bar')
dataset['MiscFeature'].value_counts().plot('bar')
dataset['SaleType'].value_counts().plot('bar')
dataset['SaleCondition'].value_counts().plot('bar')


dataset.describe()
#cleaning null values
dataset = pd.get_dummies(dataset)
dataset = dataset.fillna(dataset.mean())
print(dataset.isnull().sum())

#ploting features

dataset['MSSubClass'].value_counts().plot('bar')
dataset['OverallCond'].value_counts().plot('bar')
dataset['OverallQual'].value_counts().plot('bar')
dataset['BedroomAbvGr'].value_counts().plot('bar')
dataset['KitchenAbvGr'].value_counts().plot('bar')
dataset['TotRmsAbvGrd'].value_counts().plot('bar')
dataset['Fireplaces'].value_counts().plot('bar')
dataset['GarageCars'].value_counts().plot('bar')
dataset['MoSold'].value_counts().plot('bar')
dataset['YrSold'].value_counts().plot('bar')


dataset.plot(kind = 'scatter', x = 'SalePrice', y ='LotFrontage' )
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='LotArea' )
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='YearBuilt' )
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='YearRemodAdd' )
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='MasVnrArea' )
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='BsmtFinSF1' )
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='BsmtFinSF2' )
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='BsmtUnfSF' )
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='TotalBsmtSF' )
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='1stFlrSF' )
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='2ndFlrSF' )
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='LowQualFinSF')
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='GrLivArea')
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='BsmtFullBath')
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='BsmtHalfBath')
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='FullBath')
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='GarageYrBlt')
dataset.plot(kind = 'scatter', x = 'SalePrice', y ='GarageArea')
dataset.plot(kind = 'scatter', x = 'SalePrice', y = 'WoodDeckSF')
dataset.plot(kind = 'scatter', x = 'SalePrice', y = 'OpenPorchSF')
dataset.plot(kind = 'scatter', x = 'SalePrice', y = 'EnclosedPorch')
dataset.plot(kind = 'scatter', x = 'SalePrice', y = '3SsnPorch')
dataset.plot(kind = 'scatter', x = 'SalePrice', y = 'ScreenPorch')
dataset.plot(kind = 'scatter', x = 'SalePrice', y = 'PoolArea')
dataset.plot(kind = 'scatter', x = 'SalePrice', y = 'MiscVal')
dataset.plot(kind = 'scatter', x = 'SalePrice', y = 'YrSold')

correlation = dataset.corr(method='pearson')
#selecting
selected_data = ['MSSubClass', 'LotArea', 'OverallQual','OverallCond','YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars', 'WoodDeckSF', 'MSZoning_C (all)', 'MSZoning_FV','MSZoning_RH','MSZoning_RL', 'MSZoning_RM','Street_Grvl','Street_Pave','LotShape_IR1', 'LotShape_IR2', 'LotShape_IR3' ,'LotShape_Reg', 'LandSlope_Gtl', 'LandSlope_Mod', 'LandSlope_Sev', 'Neighborhood_Blmngtn', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NoRidge', 'Neighborhood_NPkVill', 'Neighborhood_NridgHt', 'Neighborhood_NWAmes', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker','Condition1_Artery', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_RRNn', 'Condition1_RRAn', 'Condition1_PosN', 'Condition1_PosA','Condition1_RRNe', 'Condition1_RRAe','BldgType_1Fam', 'BldgType_TwnhsE', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_2fmCon', 'ExterQual_Ex','ExterQual_Gd','ExterQual_TA','ExterQual_Fa','BsmtQual_Ex','BsmtQual_Gd','BsmtQual_TA','BsmtQual_Fa','BsmtExposure_No','BsmtExposure_Av', 'BsmtExposure_Gd', 'BsmtExposure_Mn','HeatingQC_Ex','HeatingQC_TA', 'HeatingQC_Gd', 'HeatingQC_Fa', 'HeatingQC_Po','KitchenQual_TA', 'KitchenQual_Gd', 'KitchenQual_Ex', 'KitchenQual_Fa', 'Functional_Typ', 'Functional_Min2', 'Functional_Min1', 'Functional_Mod', 'Functional_Maj1', 'Functional_Maj2', 'Functional_Sev', 'GarageFinish_Unf', 'GarageFinish_RFn', 'GarageFinish_Fin', 'PavedDrive_Y', 'PavedDrive_N', 'PavedDrive_P', 'PoolQC_Gd', 'PoolQC_Ex', 'PoolQC_Fa', 'SaleType_WD', 'SaleType_New', 'SaleType_COD', 'SaleType_ConLD', 'SaleType_ConLw', 'SaleType_ConLI', 'SaleType_CWD', 'SaleType_Oth', 'SaleType_Con' , 'SaleCondition_Normal', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Partial']
X = dataset[selected_data]
y = dataset.SalePrice

#spliting into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
'''
# Fitting Simple Linear Regression to the Training set
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression
pipeline = make_pipeline(LinearRegression())
pipeline.fit(X_train, y_train)

# Predicting the Test set results
y_pred = pipeline.predict(X_test)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = pipeline, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()