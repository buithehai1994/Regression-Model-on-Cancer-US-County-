import streamlit as st

def introduction():
    st.write("""
    During this project, I will perform linear regression models on Cancer Death Rate of US counties. 
    The dataset has one independent variable **TARGET deathrate** and 34 independent variables. 
    The project is divided into three main parts. 
    The first part is the performance of univariate linear regression on two different variables. 
    The second part will perform multivariate linear regression on various variables. 
    The final one will be the multivariate linear regression with feature engineering on variables.
    """)
