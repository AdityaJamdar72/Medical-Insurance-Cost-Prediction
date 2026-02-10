import numpy as np
import pandas as pd
import matplotlib.pyplot
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

data=pd.read_csv(r"C:\Users\Aditya\Downloads\archive (2)\insurance.csv")

#data Encodinng
sex_map={'male':1,'female':0}
smoker_map={'yes':1,'no':0}
data['sex_enc']=data['sex'].map(sex_map)
data['smoker_enc']=data['smoker'].map(smoker_map)
data.drop(columns=['sex','smoker','region'],inplace=True)

#Independent And Dependent Feature 
X=data.drop(columns=['charges'])
y=data['charges']

#Train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Standardization
scale=StandardScaler()

X_train_scale=scale.fit_transform(X_train)
X_test_scale=scale.transform(X_test)

#Model Training 
regression=LinearRegression()
model1=regression.fit(X_train_scale,y_train)
y_pred=regression.predict(X_test_scale)

#Metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mae_model1=mean_absolute_error(y_test,y_pred)
mse_model1=mean_squared_error(y_test,y_pred)
r2_val=r2_score(y_test,y_pred)

st.title("Medical Insurance Cost Prediction")

age=st.number_input('Enter You Age')
bmi=st.number_input("Enter Your Body Mass Index")
child=st.number_input('Enter The number of children ')
gender=st.radio('Select Your Gender',['Male','Female'])
if gender=='Male':
     sex=1
else:
     sex=0
smoke=st.radio("Do You Smoke ",['Yes','No'])
if smoke=='Yes':
     smoker=1
else:
     smoker=0
scaled_val=scale.transform([[age,bmi,child,sex,smoker]])
output=regression.predict(scaled_val)
if st.button("Predict"):
    prediction = model1.predict(scaled_val)[0]
    st.metric("Insurance Cost Prediction", f"₹ {prediction:,.2f}")
