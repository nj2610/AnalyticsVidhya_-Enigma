import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


train = pd.read_csv('train_NIR5Yl1.csv')
test = pd.read_csv('test_8i3B3FC.csv')
Y_pred = [0 for _ in range(len(test))]

for i in range(0,len(train),1000):
    i = 0
    X_train = train.iloc[i:i+1000:,[1,2,3,5]]
    Y_train = train.iloc[i:i+1000:,6]
    
    
    X_test = test.iloc[:,[1,2,3,5]]
    
    X_train.loc[X_train['Tag'] == 'a','Tag'] = 0
    X_train.loc[X_train['Tag'] == 'b','Tag'] = 1
    X_train.loc[X_train['Tag'] == 'c','Tag'] = 2
    X_train.loc[X_train['Tag'] == 'd','Tag'] = 3
    X_train.loc[X_train['Tag'] == 'e','Tag'] = 4
    X_train.loc[X_train['Tag'] == 'f','Tag'] = 5
    X_train.loc[X_train['Tag'] == 'g','Tag'] = 6
    X_train.loc[X_train['Tag'] == 'h','Tag'] = 7
    X_train.loc[X_train['Tag'] == 'i','Tag'] = 8
    X_train.loc[X_train['Tag'] == 'j','Tag'] = 9
    X_train.loc[X_train['Tag'] == 'k','Tag'] = 10
    X_train.loc[X_train['Tag'] == 'l','Tag'] = 11
    X_train.loc[X_train['Tag'] == 'm','Tag'] = 12
    X_train.loc[X_train['Tag'] == 'n','Tag'] = 13
    X_train.loc[X_train['Tag'] == 'o','Tag'] = 14
    X_train.loc[X_train['Tag'] == 'p','Tag'] = 15
    X_train.loc[X_train['Tag'] == 'q','Tag'] = 16
    X_train.loc[X_train['Tag'] == 'r','Tag'] = 17
    X_train.loc[X_train['Tag'] == 's','Tag'] = 18
    X_train.loc[X_train['Tag'] == 't','Tag'] = 19
    X_train.loc[X_train['Tag'] == 'u','Tag'] = 20
    X_train.loc[X_train['Tag'] == 'v','Tag'] = 21
    X_train.loc[X_train['Tag'] == 'w','Tag'] = 22
    X_train.loc[X_train['Tag'] == 'x','Tag'] = 23
    X_train.loc[X_train['Tag'] == 'y','Tag'] = 24
    X_train.loc[X_train['Tag'] == 'z','Tag'] = 25
    
    
    X_test.loc[X_test['Tag'] == 'a','Tag'] = 0
    X_test.loc[X_test['Tag'] == 'b','Tag'] = 1
    X_test.loc[X_test['Tag'] == 'c','Tag'] = 2
    X_test.loc[X_test['Tag'] == 'd','Tag'] = 3
    X_test.loc[X_test['Tag'] == 'e','Tag'] = 4
    X_test.loc[X_test['Tag'] == 'f','Tag'] = 5
    X_test.loc[X_test['Tag'] == 'g','Tag'] = 6
    X_test.loc[X_test['Tag'] == 'h','Tag'] = 7
    X_test.loc[X_test['Tag'] == 'i','Tag'] = 8
    X_test.loc[X_test['Tag'] == 'j','Tag'] = 9
    X_test.loc[X_test['Tag'] == 'k','Tag'] = 10
    X_test.loc[X_test['Tag'] == 'l','Tag'] = 11
    X_test.loc[X_test['Tag'] == 'm','Tag'] = 12
    X_test.loc[X_test['Tag'] == 'n','Tag'] = 13
    X_test.loc[X_test['Tag'] == 'o','Tag'] = 14
    X_test.loc[X_test['Tag'] == 'p','Tag'] = 15
    X_test.loc[X_test['Tag'] == 'q','Tag'] = 16
    X_test.loc[X_test['Tag'] == 'r','Tag'] = 17
    X_test.loc[X_test['Tag'] == 's','Tag'] = 18
    X_test.loc[X_test['Tag'] == 't','Tag'] = 19
    X_test.loc[X_test['Tag'] == 'u','Tag'] = 20
    X_test.loc[X_test['Tag'] == 'v','Tag'] = 21
    X_test.loc[X_test['Tag'] == 'w','Tag'] = 22
    X_test.loc[X_test['Tag'] == 'x','Tag'] = 23
    X_test.loc[X_test['Tag'] == 'y','Tag'] = 24
    X_test.loc[X_test['Tag'] == 'z','Tag'] = 25
    
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    
    onehotencoder = OneHotEncoder(categorical_features = [0])
    X_train = onehotencoder.fit_transform(X_train).toarray()
    onehotencoder = OneHotEncoder(categorical_features = [0])
    X_test = onehotencoder.fit_transform(X_test).toarray()
    
    
    #Feature scaling
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.fit_transform(X_test)
    
    #Applying PCA
#    from sklearn.decomposition import PCA
#    pca = PCA(n_components = 2)
#    X_train = pca.fit_transform(X_train)
#    X_test = pca.transform(X_test)
#    explained_variance = pca.explained_variance_ratio_
#    
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'linear')
    regressor.fit(X_train, Y_train)
    
    Y_pred = Y_pred + regressor.predict(X_test)
    
#    from sklearn.model_selection import cross_val_score
#    accuracies = cross_val_score(estimator = regressor, X = X_train, y = Y_train, cv = 10 )
#    accuracies.mean()
#    accuracies.std()
    
    
#    
#    from sklearn.svm import SVC
#    classifier = SVC(kernel = "linear", random_state = 0, gamma = 0.2001)
#    classifier.fit(X_train,Y_train)

