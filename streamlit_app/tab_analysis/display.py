import streamlit as st

def display_univariate_analysis():
    explanation_text = """
    **I. Experiment on univariate linear regression: medIncome and povertyPercent:**
    
    **1. Overview:**
    
    The goal of this experiment is to examine the relationship between wealth and cancer death rate (TARGET_deathRate). Therefore, I will choose two variables, namely medIncome and povertyPercent as independent variables and train two univariate linear regression models. 
    medianIncome variable represents the median income per US county while povertyPercent calculates the percent of the populace in poverty. As a result, these two independent variables should demonstrate the relationship between wealth and cancer rate.
    The results of this study may indicate a potential inequality in healthcare treatment between the rich and the poor. The costs of treatment or standard of living may be the reasons for this imparity. Despite the high fee for cancer treatment, the insurance fee is more affordable. A reasonable price insurance package with an effective mechanism for people with low incomes can be the solution to shorten the gap in cancer diagnosis and treatment. As a result, micro-insurance products, which offer coverage for poor people with little savings, should be promoted.
    
    **2. Analysis:**
    
    This experience will use Mean Square Error for assessing the performances of linear regression models. The mean of variables serves as a baseline for evaluation. First, the MSE scores of the two modes are smaller than that of the baseline. It means that the two models achieve improvement compared to the baseline. Second, Model 2 (using medIncome) has a lower testing MSE compared to Model 1 (using povertyPercent).  This suggests that Model 2 might be performing slightly better in terms of prediction accuracy on unseen data.
    Based on the Coefficient, the model shows the exact result as we predicted.  medIncome has a negative impact on TargetDeathRate while povertyPercent and TargetDeathRate show a positive correlation.
    Therefore, it is evident that the wealth of households and the cancer death rate are negatively correlated.

    Experimental results prove the relationship between wealth and the risk of dying from cancer. As mentioned above, this relationship may have many profound and different meanings in many areas of our society. For example, from the government's viewpoint, this experiment emphasizes the bad influence of income parity in US society and they need to find a way to tackle these problems. However, dealing with cancer needs the joint efforts of everyone in our society. The participation of entrepreneurs is a good solution when they can balance social responsibility and profit maximization. For example, they can get an edge on their rivalry in business areas like insurance.
    
    In my view, this experiment should be carried out globally to obtain a broader picture. When the investigation is conducted in American society, we cannot give an overall conclusion and recommendation from a global perspective.

    However, it is good evidence of a link between wealth and cancer mortality. Many experts suggest that early diagnosis is essential in cancer treatment. As suggested above, this is an excellent opportunity for insurance companies. The cost of routine screening and insurance is much cheaper than cancer treatment. Their next step is to conduct more thorough surveys to obtain a broader picture of demography. Insurance companies should evaluate each specific class of customers and offer appropriate product packages.
    
    **II. Experiment on multivariate linear regression:** 
    
    **1. Overview:**
    
    The objective of this experiment is to illustrate the relationship between the cancer death rate and other socioeconomic factors (as ‘incidenceRate', 'medIncome', 'studyPercap', 'MedianAge', 'Geography', 'AvgHouseholdSize', 'PctNoHS18_24', 'PctHS18_24', 'BirthRate' )
    
    The above variables cover a broad range of factors like health outcomes, demography, education, and geography, which could potentially explain variations in death rates across different populations or geographic regions

    First, the MSE of the mode is smaller than that of the baseline. It means that the model achieves improvement compared to the baseline. The model performance is also improved on the testing set.

    The coefficient of the model shows that areas with higher median incomes tend to have lower cancer death rates.
    
    **2. Analysis:**

    The negative coefficient for median income indicates that there is indeed a negative relationship between median income and cancer death rate, which is aligned with the findings of the first experiment. So, areas with higher median incomes tend to have lower cancer death rates, which aligns with what we would typically expect in terms of access to healthcare, better lifestyle choices, and other socioeconomic factors associated with higher income levels.

    Higher study per capita is associated with lower cancer death rates. This suggests that areas with more educational resources or research tend to have lower cancer death rates.
    
    There appears to be a positive relationship between the incidence rate and cancer death rate. This suggests that areas with higher incidence rates tend to have higher cancer death rates.
    
    Overall, the regression model suggests that socioeconomic factors such as income and education, as well as demographic factors like age and birth rate, may play significant roles in determining cancer death rates within a given area. Additionally, incidence rate and study per capita also seem to be important factors.

    The above results may provide various business impacts. Healthcare providers and policymakers can use this information to better allocate resources for cancer prevention, screening, and treatment. For example, areas with higher incidence rates and lower median incomes may require additional funding for cancer screening programs or access to affordable treatment options.
    
    Moreover, health insurance companies, pharmaceutical companies, and healthcare providers can use this information to tailor their marketing and outreach efforts. For instance, they can focus on promoting cancer prevention and early detection services to communities with higher incidence rates and lower education levels.

    However, the issue of these experiments is that they are tested in US society. We need a bigger dataset to test the influence of education level on cancer death rate. 

    My key learning from the experiment is the complex interplay between socio-economic and demographic factors and cancer death rates, thereby emphasizing the impact of developing strategies to reduce disparities and improve health outcomes for all communities.

    **III. Experiment on multivariate linear regression with feature engineering:**

    **1. Overview:**
    
    The objective of this experiment is to illustrate the relationship between the cancer death rate and using all other socioeconomic factors. In this part, other factors related to races will be include in the independent variables.

    In this part the variables are transformed by logarithm function before the regression test. The reason for transforming the data is that the data is not normally distributed. As a results, the log transformation will make it close to normal distribution.
    
    **2. Analysis:**

    In this part, after transforming the variables, the model is overfitted. As a result, I will perform regularizations using Lasso, Ridge, and Elastic models. 

    It appears that the Multivariate Linear Regression model without regularization (baseline) has the highest MSE and MAE on the training set, indicating overfitting. However, on the testing set, the Ridge model and the baseline model have the lowest MSE and MAE, suggesting better generalization performance. The Elastic Net model has the highest MSE and MAE on both training and testing sets, indicating it might not be the most suitable choice for this dataset.

    Overall, the Ridge model seems to strike a good balance between bias and variance, offering decent performance on both the training and testing sets.

    The above graph demonstrates the coefficients of the Ridge Regression model. median income has the most influence on cancer death rate predictions. This suggests that socio-economic factors, particularly income levels, play a crucial role in determining cancer mortality rates, as higher median incomes are associated with lower cancer death rates.

    AvgHouseholdSize also has a significant influence on cancer death rate predictions. This suggests that socio-economic factors associated with larger households, such as lower income levels or limited access to healthcare, may contribute to poorer cancer outcomes.

    The coefficients for PctNoHS18_24 and PctHS18_24 indicate that education levels among young adults correlate with cancer mortality rates. Higher education levels are associated with lower cancer death rates, emphasizing the importance of education in promoting health literacy, awareness of preventive measures, and adherence to healthcare recommendations.

    The coefficient for studyPerCap suggests that areas with higher educational resources or research activities tend to have slightly lower cancer death rates. This implies that access to healthcare resources, including educational institutions and research facilities, can positively impact cancer outcomes by facilitating early detection, innovative treatments, and dissemination of health information.

    The coefficient for BirthRate suggests that areas with higher birth rates tend to have slightly lower cancer death rates. This may reflect the age distribution of the population, with younger populations potentially having lower cancer mortality rates due to different disease prevalence or screening behaviors.

    The model results have many implications. Higher-income levels or larger household sizes may be associated with certain lifestyle choices, access to healthcare, or exposure to environmental risks that can impact cancer risk or mortality. It may has impacts on treatment of cancer when the patients are white people, especially this study is carried out in the US, the country has many white people.

    My key learning from the experiment is the connection between various socioeconomic factors and the target death rate. Those factors may affect the living conditions, people’s awareness and lifestyles, which may contribute greatly to the cancer risk.

    **IV. Conclusion:**

    The Ridge model's ability to strike a balance between bias and variance often makes it a preferred choice in situations where there's a risk of overfitting, as it introduces a penalty term to the loss function, which helps to reduce the model's sensitivity to changes in the input data and mitigates overfitting.
    In this experiment, the **Ridge model** consistently outperforms other models across metrics like Mean Squared Error (MSE), Mean Absolute Error (MAE), and maintains a good balance between performance on the training and testing sets, making it the model of choice for the deployment phase.
    """
    st.markdown(explanation_text)
