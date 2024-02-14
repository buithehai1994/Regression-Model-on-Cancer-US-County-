import streamlit as st

def display_univariate_analysis():
    explanation_text = """
    I. Experiment on univariate linear regression: medIncome and povertyPercent
    1. Overview:
    
    The goal of this experiment is to examine the relationship between wealth and cancer death rate (TARGET_deathRate). Therefore, I will choose two variables, namely medianIncome and povertyPercent as independent variables and train two univariate linear regression models. 
    medianIncome variable represents the median income per US county while povertyPercent calculates percent of the populace in poverty. As a result, these two independent variables should demonstrate the relationship between wealth and cancer rate.
    The results of this study may indicate a potential inequality in healthcare treatment between the rich and the poor. The costs of treatment or standard of living may be the reasons for this imparity. Despite the high fee of cancer treatment, the fee for insurance is more affordable. A reasonable price insurance package with an effective mechanism for people with low incomes can be the solution to shorten the gap in cancer diagnosis and treatment. As a result, micro insurance products, which offer coverage for poor people with little savings, should be promoted.
    2. Analysis:

    a. Univariate models (TARGET_deathRate and medIncome, TARGET_deathRate and povertyPercent):
    This experience will use Mean Square Error for assessing the performances of linear regression models. The mean of variables serves as a baseline for evaluating. First, MSE of the two modes are smaller than that of baseline. It means that two models achieve improvement compared to the baseline. Second, Model 2 (using medIncome) has a lower testing MSE compared to Model 1 (using povertyPercent).  This suggests that Model 2 might be performing slightly better in terms of prediction accuracy on unseen data.
    Based on the Coefficient, the model shows the exact result as we predicted.  medIncome has a negative impact on TargetDeathRate while povertyPercent and TargetDeathRate show a positive correlation.
    Therefore, it is evident that the wealth of households and cancer death rate are negatively correlated.

    Experimental results prove the relationship between wealth and risk of dying from cancer. As mentioned above, this relationship may have many profound and different meanings in many areas of our society. For example, from the government view point, this experiment emphasizes the bad influence of income parity in US society and they need to find a way to tackle these problems. However, dealing with cancer needs the joint efforts of everyone in our society. The participation of entrepreneurs is a good solution when they can balance social responsibility and profit maximization. For example, they can get an edge on their rivalry in business areas like insurance.
    """
    st.markdown(explanation_text)
