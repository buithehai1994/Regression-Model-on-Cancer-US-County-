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
    
def display_univariate_introduction():

    explanation_text = """
    The goal of this experiment is to examine the relationship between wealth and cancer death rate (TARGET_deathRate). Therefore, I will choose two variables, namely **medianIncome** and povertyPercent as independent variables and train two univariate linear regression models. 
    **medianIncome** variable represents the median income per US county while povertyPercent calculates percent of the populace in poverty. As a result, these two independent variables should demonstrate the relationship between wealth and cancer rate.
    The results of this study may indicate a potential inequality in healthcare treatment between the rich and the poor. The costs of treatment or standard of living may be the reasons for this imparity. Despite the high fee of cancer treatment, the fee for insurance is more affordable. A reasonable price insurance package with an effective mechanism for people with low incomes can be the solution to shorten the gap in cancer diagnosis and treatment. As a result, micro insurance products, which offer coverage for poor people with little savings, should be promoted.
    """
    
    st.markdown(explanation_text)  # Display the explanatory text
