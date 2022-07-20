import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


path = 'hiring.csv'
data = pd.read_csv(path)
data.experience.fillna(0,inplace=True)
data.test_score.fillna(data.test_score.mean(),inplace=True)
print(data)

# X = data.drop('salary',axis=1)
X = data.iloc[:,:3]
y = data.iloc[:,-1]



# labels = [x for x in X.experience if not isinstance(x, int)]
# le = preprocessing.LabelEncoder()
# le.fit(labels)
# le.classes_
# X.experience = le.transform(X.experience)

def convert_to_int(word):
    alpha_numeric_dic = {
        'zero':0,
        'one':1,
        'two':2,
        'three':3,
        'four':4,
        'five':5,
        'six':6,
        'seven':7,
        'eight':8,
        'nine':9,
        'ten':10,
        'eleven':11,
        'twelve':12,
        0:0
    }
    return alpha_numeric_dic[word]



X.experience = X.experience.apply(lambda x: convert_to_int(x))
X.experience


#model
regressor = LinearRegression()
regressor.fit(X,y)

#save model
pickle.dump(regressor,open('model.pkl','wb'))



