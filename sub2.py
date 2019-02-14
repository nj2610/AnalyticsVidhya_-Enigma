import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


train = pd.read_csv('train_NIR5Yl1.csv')
test = pd.read_csv('test_8i3B3FC.csv')


train.loc[train['Tag'] == 'a','Tag'] = 0
train.loc[train['Tag'] == 'b','Tag'] = 1
train.loc[train['Tag'] == 'c','Tag'] = 2
train.loc[train['Tag'] == 'd','Tag'] = 3
train.loc[train['Tag'] == 'e','Tag'] = 4
train.loc[train['Tag'] == 'f','Tag'] = 5
train.loc[train['Tag'] == 'g','Tag'] = 6
train.loc[train['Tag'] == 'h','Tag'] = 7
train.loc[train['Tag'] == 'i','Tag'] = 8
train.loc[train['Tag'] == 'j','Tag'] = 9
train.loc[train['Tag'] == 'k','Tag'] = 10
train.loc[train['Tag'] == 'l','Tag'] = 11
train.loc[train['Tag'] == 'm','Tag'] = 12
train.loc[train['Tag'] == 'n','Tag'] = 13
train.loc[train['Tag'] == 'o','Tag'] = 14
train.loc[train['Tag'] == 'p','Tag'] = 15
train.loc[train['Tag'] == 'q','Tag'] = 16
train.loc[train['Tag'] == 'r','Tag'] = 17
train.loc[train['Tag'] == 's','Tag'] = 18
train.loc[train['Tag'] == 't','Tag'] = 19
train.loc[train['Tag'] == 'u','Tag'] = 20
train.loc[train['Tag'] == 'v','Tag'] = 21
train.loc[train['Tag'] == 'w','Tag'] = 22
train.loc[train['Tag'] == 'x','Tag'] = 23
train.loc[train['Tag'] == 'y','Tag'] = 24
train.loc[train['Tag'] == 'z','Tag'] = 25


test.loc[test['Tag'] == 'a','Tag'] = 0
test.loc[test['Tag'] == 'b','Tag'] = 1
test.loc[test['Tag'] == 'c','Tag'] = 2
test.loc[test['Tag'] == 'd','Tag'] = 3
test.loc[test['Tag'] == 'e','Tag'] = 4
test.loc[test['Tag'] == 'f','Tag'] = 5
test.loc[test['Tag'] == 'g','Tag'] = 6
test.loc[test['Tag'] == 'h','Tag'] = 7
test.loc[test['Tag'] == 'i','Tag'] = 8
test.loc[test['Tag'] == 'j','Tag'] = 9
test.loc[test['Tag'] == 'k','Tag'] = 10
test.loc[test['Tag'] == 'l','Tag'] = 11
test.loc[test['Tag'] == 'm','Tag'] = 12
test.loc[test['Tag'] == 'n','Tag'] = 13
test.loc[test['Tag'] == 'o','Tag'] = 14
test.loc[test['Tag'] == 'p','Tag'] = 15
test.loc[test['Tag'] == 'q','Tag'] = 16
test.loc[test['Tag'] == 'r','Tag'] = 17
test.loc[test['Tag'] == 's','Tag'] = 18
test.loc[test['Tag'] == 't','Tag'] = 19
test.loc[test['Tag'] == 'u','Tag'] = 20
test.loc[test['Tag'] == 'v','Tag'] = 21
test.loc[test['Tag'] == 'w','Tag'] = 22
test.loc[test['Tag'] == 'x','Tag'] = 23
test.loc[test['Tag'] == 'y','Tag'] = 24
test.loc[test['Tag'] == 'z','Tag'] = 25

X_train = train.iloc[:,[1,2,3,4,5]]
Y_train = train.iloc[:,6]


X_test = test.iloc[:,[1,2,3,4,5]]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X_test = onehotencoder.fit_transform(X_test).toarray()

from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_test = scaler_X.fit_transform(X_test)


onehotencoder = OneHotEncoder(categorical_features = [0])
X_train = onehotencoder.fit_transform(X_train).toarray()

#Feature scaling
X_train = scaler_X.fit_transform(X_train)
   

from xgboost import XGBRegressor
regressor = XGBRegressor(learning_rate = 0.09, n_estimators = 200,gamma = 2)
regressor.fit(X_train, Y_train)



    
Y_pred = regressor.predict(X_test)
    
Y_pred[Y_pred < 0] = 0





predictions = pd.DataFrame(Y_pred, columns=['Upvotes']).to_csv('prediction.csv')
