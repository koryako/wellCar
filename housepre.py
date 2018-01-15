# -*- coding: utf-8 -*-
import numpy
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC 
# load dataset
dataframe = pandas.read_csv("pre.csv", header=None)
dataset = dataframe.values
dataset=dataset[1:,:]
dataset = shuffle(dataset) 

#y=(x-MinValue)/(MaxValue-MinValue)



# 定义array

# 对array进行归一化(normalization)
# scale进行的操作是按列减去均值, 除以方差, 因此数据的均值为0, 方差为1
#dataset=preprocessing.scale(dataset)
print(dataframe.idxmax())
print(dataframe.describe())
print(dataset.shape) 

# split into input (X) and output (Y) variables
X = dataset[1:, 0]
Y = dataset[1:, 1]
print(len(X))
print(len(Y))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
X_test = dataset[1:, 0]
Y_test = dataset[1:, 1]
# define base mode
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(11, input_dim=1, init='normal', activation='relu'))
    model.add(Dense(6, init='normal', activation='relu'))
    model.add(Dense(1, init='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    return model
plt.scatter(dataset[:, 0], dataset[:, 1])
#plt.scatter(dataset[:, 0], dataset[:, 1], c = y)
plt.show()
model=baseline_model()



# 生成数据集
#X, y = make_classification(n_samples = 200, n_features = 2, n_redundant = 0, n_informative = 2, 
                           #random_state = 22, n_clusters_per_class = 1, scale = 100)
#model.fit(X, Y, nb_epoch=600, batch_size=30, verbose=2)
# training 

print('Training -----------')  
for step in range(600):  
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:  
        print('train cost: ', cost)  
    if cost[0]<1.0:
        break  
    

"""
# test  
print('\nTesting ------------')  
cost = model.evaluate(X_test, Y_test, batch_size=30)  
print('test cost:', cost)  
W, b = model.layers[0].get_weights()  
print('Weights=', W, '\nbiases=', b)  
"""

X_pre=[1801,1802,1803,1804,1805,1807,1808,1809,1810,1811,1812]

# plotting the prediction  
Y_pred = model.predict(X_pre)  
print(Y_pred)
plt.scatter(dataset[:, 0], dataset[:, 1])
plt.show() 
"""
plt.plot(X_pre, Y_pred)  
plt.show() 
""" 


"""
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=5000, batch_size=5, verbose=0)
# use 10-fold cross validation to evaluate this baseline model
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
"""
"""
# evaluate model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
# use 10-fold cross validation to evaluate this baseline model
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
"""
