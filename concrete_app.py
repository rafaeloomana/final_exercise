import streamlit as st
import pandas as pd
import numpy as np
import joblib

data = pd.read_csv("concrete_data.csv")

daFr = pd.DataFrame(data.head(20))

st.title("Concrete Strength App")
st.write("From the concrete data, we built a machine learning model for concrete strength.")

st.sidebar.title("Concrete Strength App Parameters")
st.sidebar.write("Tweak to change predictions")

option_sidebar = option = st.sidebar.selectbox("Hide / Show Data Frame", ('Hide', 'Show'))

#Cement
cement = st.sidebar.slider("Cement (KG)", 0.0, 500.0, 25.0)

#bfs
bfs = st.sidebar.slider("Blast Furnance Slag (KG)" , 0.0, 500.0, 25.0)

#Water
water = st.sidebar.slider("Water (KG)" , 0.0, 500.0, 25.0)

#Fly Ash
flyash = st.sidebar.slider("Fly Ash (KG)", 0.0, 10.0, 1.0)

#Coarse Aggregate
ca = st.sidebar.slider("Coarse Aggregate (KG)", 0.0, 10.0, 1.0)

#fine Aggregate
fa = st.sidebar.slider("Fine Aggregate (KG)", 0.0, 10.0, 1.0)

#Superplasticizer
sp = st.sidebar.slider("Superplasticizer (KG)", 0.0, 10.0, 1.0)

#Age
age = st.sidebar.slider("Age" , 0, 15, 1)


if option_sidebar == 'Show':
    st.write(daFr)
    
st.subheader("Chart One: Bar Chart")
#bar charts 1
chart_data = pd.DataFrame(
    data.head(30),
    columns=['Cement','Fly Ash','Water','Age']
)

st.bar_chart(chart_data)

#area charts 2
st.subheader("Chart Two: Area Chart")
chart_data = pd.DataFrame(
    data.head(30),
    columns=['Cement','Fly Ash','Water','Age']
)

st.area_chart(chart_data)

st.subheader("Chart Three: Line Chart")
chart_data = pd.DataFrame(
    data.head(30),
    columns=['Cement','Fly Ash','Water','Age']
)

st.line_chart(chart_data)

st.subheader('Concrete Strength')

filename="concrete_model.sav"

loaded_model = joblib.load(filename)

prediction=loaded_model.predict([[cement, bfs, water, flyash, sp, ca, fa, age]])

if prediction < 50:
    st.write(f"The Concrete Strength is: Weak")
elif prediction > 50:
    st.write(f"The Concrete Strength is: Strong")
    
st.write(f"The Concrete Strength Numerical Value is: {prediction} MPa")
