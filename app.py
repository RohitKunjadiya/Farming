import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

df = pickle.load(open('df.pkl','rb'))
xgb = pickle.load(open('xgb.pkl','rb'))

X = df.drop('label',axis=1)
y = df.iloc[:,-1]

le = LabelEncoder()
y_le = le.fit_transform(y)

st.title('Crop Recommendation')

nitrogen = st.number_input(label='Amount of Nitrogen:',step=1,min_value=0,max_value=145)

phosphorous = st.number_input(label='Amount of Phosphorous:',step=1,min_value=0,max_value=150)

pottasium = st.number_input(label='Amount of Pottasium:',step=1,min_value=3,max_value=210)

temp = st.number_input(label='Amount of Temperature:',step=1.,format="%.2f",min_value=7.01,max_value=44.01)

humidity = st.number_input(label='Amount of Humidity:',step=1.,format="%.2f",min_value=13.01,max_value=101.01)

ph = st.number_input(label='Amount of PH :',step=1.,format="%.2f",min_value=2.01,max_value=11.01)

rain = st.number_input(label='Rain : ',step=1.,format="%.2f",min_value=10.01,max_value=300.01)

btn = st.button('Predict')

if btn:
    query = pd.DataFrame(data=np.array([[float(nitrogen), float(phosphorous), float(pottasium), float(temp), float(humidity), float(ph), float(rain)]]), columns=X.columns)
    st.title('Prediction is:'+str(le.inverse_transform(xgb.predict(query))))