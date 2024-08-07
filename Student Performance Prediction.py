#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
data = pd.read_csv(r"student-mat.csv",sep=';')
data = data[["G1","G2","G3","health","age"]]
print(data.head())
predict = "G3"
x = np.array(data.drop([predict],axis= 1))
print(data)
y = np.array(data[predict])
x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split( x, y, test_size = 0.1 )

best = 0
for _ in range(30):

    x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split( x, y, test_size = 0.1 )
    linear = linear_model.LinearRegression()
    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)
    if acc>best :
        acc = best
        print(acc) 
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f)
    
pickle_in = open("studentmodel.pickle","rb")
linear = pickle.load(pickle_in)

print('coef: \n',linear.coef_)
print('Intercept: \n' , linear.intercept_)
predictions = linear.predict(x_test)

 
for i in range(len(predictions)):
    
    print(predictions[x],x_test[x],y_test[x])
    
p = "G2"  
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel("p")
pyplot.ylabel("final Grade")
pyplot.show()



# In[ ]:




