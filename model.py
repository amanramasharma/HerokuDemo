import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


df = pd.read_csv('homeprices.csv')


#Splitting independent and dependent data
#Since we have very small data that's why we use all dataset as training data
x =df.iloc[:,:3]
y = df.iloc[:,-1]

#Linear Regression use for train the Model
reg = LinearRegression()

#Fitting model with training data
reg.fit(x,y)

#Saving model to disk
pickle.dump(reg , open('model.pkl','wb'))

#Loding model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))
