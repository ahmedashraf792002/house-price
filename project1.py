import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression,Ridge,SGDRegressor,Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
print(train_data.shape)
print(test_data.shape)
col=train_data.shape[1]
print(col)
x_train=train_data.iloc[:,0:col-1]
y_train=train_data.iloc[:,col-1:col]
print(x_train.shape)
print(y_train.shape)
list=['MSZoning','Street','Alley','LotShape','LandContour',
      'Utilities','LotConfig','LandSlope','Neighborhood'
      ,'Condition1','Condition2','BldgType','HouseStyle'
      ,'RoofStyle','RoofMatl','Exterior1st','Exterior2nd'
      ,'MasVnrType','ExterQual','ExterCond','Foundation',
      'BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
      'BsmtFinType2','Heating','HeatingQC','CentralAir'
      ,'Electrical','KitchenQual','Functional','FireplaceQu',
      'GarageType','GarageFinish','GarageQual','GarageCond',
      'PavedDrive','PoolQC','Fence','MiscFeature',
      'SaleType','SaleCondition']
label=LabelEncoder()
for li in list:
           x_train[li]=label.fit_transform(x_train[li])
           test_data[li]=label.fit_transform(test_data[li])
impute=SimpleImputer(missing_values=np.nan,strategy='mean')
x_train=impute.fit_transform(x_train)
test_data=impute.fit_transform(test_data)
x_train2,x_test,y_train2,y_test=train_test_split(x_train,y_train,test_size=.33,shuffle=True,random_state=33)
print(x_train2.shape)
print(x_test.shape)
linear=LinearRegression()
linear.fit(x_train2,y_train2)
print("LinearRegression train score =",linear.score(x_train2,y_train2))
print("LinearRegression train score =",linear.score(x_test,y_test))
pred=linear.predict(test_data)
print(pred[:10])
test=np.array(test_data[:,0],dtype=int)
submission=pd.DataFrame({'id':test})
submission['SalePrice']=pred
submission.set_index('id', inplace=True)
submission.to_csv('submission2.csv')
print(submission)