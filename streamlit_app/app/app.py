import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import openpyxl
import base64
import imblearn
import ydata_profiling
from streamlit_pandas_profiling import st_profile_report
import dataprep
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import altair as alt
from sklearn.preprocessing import StandardScaler

# Set Python path
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parents[0])
sys.path.append(parent_dir)

from tab_eda.display import display_summary_statistics,display_info,display_missing_values
from tab_encoding.display import display_tab_df_encoding_explain, display_correlation_encoding_heatmap
from tab_encoding.logics import Encoding
from tab_ml.logics import ML
from tab_ml.display import display_baseline_metrics,display_model_metrics,display_line_chart,display_scatter_chart,display_chart,display_multiple_chart,display_coefficients
from tab_analysis.display import display_univariate_analysis
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from sklearn.linear_model import LinearRegression

from dataprep.eda import plot,create_report
import pickle
import csv
import streamlit.components.v1 as components
import warnings
from sklearn.exceptions import ConvergenceWarning

with warnings.catch_warnings():
    # Suppress warnings related to feature names in Logistic Regression
    warnings.simplefilter("ignore", category=ConvergenceWarning)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CSV Explorer",
    page_icon=None,
    layout="wide",
)

# Load data from "Data" tab
# Get the current directory of the script
current_dir = os.path.dirname(__file__)

training_set_path=Path(__file__).resolve().parent.parent.parent  / "Dataset" / "cancer_us_county-training.csv"
testing_set_path=Path(__file__).resolve().parent.parent.parent  / "Dataset" / "cancer_us_county-testing.csv"

df_train=pd.read_csv(training_set_path)
df_test=pd.read_csv(testing_set_path)

dataset=pd.concat([df_train,df_test],axis=0)

def remove_hyperlinks(html_content):
    # Remove anchor tags (<a>) from the HTML content
    return re.sub(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"[^>]*>', r'<span>\1</span>', html_content)

data_from_tab_df = dataset


def perform_encoding():
    encoding = Encoding(data=data_from_tab_df)
    data_for_ml = encoding.label_encoding()
    return data_for_ml
    
def perform_encoding_and_multivariate():
    encoding = Encoding(data=data_from_tab_df)
    data_for_ml = encoding.label_encoding()
    data_for_ml =encoding.multivariate_process()
    return data_for_ml
    
def perform_feature_engineering():
    encoding = Encoding(data=data_from_tab_df)
    data_for_ml =encoding.feature_engineering_process()
    return data_for_ml
    
data_for_ml_univariate = perform_encoding()
data_for_ml_multivariate= perform_encoding_and_multivariate()
data_for_ml_feature_engineering= perform_feature_engineering()

selected_tab = st.sidebar.radio("Navigation", ["Introduction", "Data", "EDA", "Encoding", "Machine Learning Model", "Analysis", "Deployment", "Ethical Consideration", "References", "GitHub"], key="navigation")

# Display content based on selected sidebar tab

if selected_tab =="Introduction":
    pass
elif selected_tab == "Data":
    st.sidebar.header("Data")
    st.write(dataset.head())
elif selected_tab == "EDA":
    st.sidebar.header("EDA")

    # Create sub-tabs for EDA section
    tab_titles = ["Summary Statistics", "Plots"]
    selected_sub_tab = st.sidebar.radio("Sub-navigation", tab_titles)

    if selected_sub_tab == tab_titles[0]:
        st.header(f"Summary Statistics")
        # Create sub-sub-tabs for Correlation
        sub_tab_titles = ["Summary", "Info", "Missing Values"]
        selected_sub_sub_tab = st.sidebar.radio("Dataset", sub_tab_titles)
        if selected_sub_sub_tab == sub_tab_titles[0]:
            display_summary_statistics(data_from_tab_df)
        elif selected_sub_sub_tab == sub_tab_titles[1]:
            display_info(data_from_tab_df)
        else:
            display_missing_values(data_from_tab_df)
    if selected_sub_tab == tab_titles[1]:
        # https://buithehaiuts.github.io/repurchase-car/report.html
        external_url = "https://htmlpreview.github.io/?https://github.com/buithehai1994/Regression-Model-on-Cancer-US-County-/blob/main/github_page/eda_report.html"
        # Render the external content in an iframe
        st.write(f'<iframe src="{external_url}" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; border: none;"></iframe>', unsafe_allow_html=True)

elif selected_tab == "Machine Learning Model":
    tab_titles = ["Univariate", "Multvariate", "Multivariate with Feature Enginnering"]

    selected_sub_tab = st.sidebar.radio("Dataset", tab_titles)
    
    if selected_sub_tab == tab_titles[0]:
        sub_sub_tab_titles = ["povertyPercent", "medIncome"]
        selected_sub_sub_tab = st.sidebar.radio("Dataset", sub_sub_tab_titles)

        if selected_sub_sub_tab == sub_sub_tab_titles[0]:
            X = data_for_ml_univariate['povertyPercent'].values
            y = data_for_ml_univariate['TARGET_deathRate'].values
            
            ml_instance = ML()
            # Call the split_data method to split your data into training and testing sets
            X_train, X_test, y_train, y_test = ml_instance.split_data(X, y)
            
            ml_instance.train_linear_regression(X_train.reshape(-1, 1), y_train)

            st.write("<h1 style='font-size: 32px; font-weight: bold;'>Baseline Model</h1>", unsafe_allow_html=True)
            
            # calculate baseline
            y_mean = y_train.mean()
            y_base = np.full(y_train.shape, y_mean)
            mse_score = mse(y_train, y_base, squared=True)
            mae_score = mae(y_train, y_base)
            st.write("MSE of Baseline: ", mse_score)
            st.write("MAE of Baseline: ", mae_score)

            st.write("    ")
            st.write("    ")

            y_train_preds = ml_instance.predict(X_train.reshape(-1, 1))
            y_test_preds = ml_instance.predict(X_test.reshape(-1, 1))

            line_chart_train = alt.Chart(pd.DataFrame({'x':X_train, 'y': y_train_preds})).mark_line(opacity=1, color='blue').encode(
                    x='x',
                    y='y'
                  )

            scatter_chart_train = alt.Chart(pd.DataFrame({'x':X_train, 'y': y_train})).mark_circle(opacity=1, color='red').encode(
                x='x',
                y='y'
              )

            final_chart_train=(line_chart_train+scatter_chart_train).properties(
                                        title='Poverty Percent',
                                        width=400,
                                        height=800
                                    ).configure_title(
                                        anchor='middle'
                                    ).configure_legend(
                                        orient='top'
                                    ).configure_axis(
                                        labelFontSize=12,
                                        titleFontSize=14
                                    )


            line_chart_test = alt.Chart(pd.DataFrame({'x':X_test, 'y': y_test_preds})).mark_line(opacity=1, color='blue').encode(
                    x='x',
                    y='y'
                  )

            scatter_chart_test = alt.Chart(pd.DataFrame({'x':X_test, 'y': y_test})).mark_circle(opacity=1, color='red').encode(
                x='x',
                y='y'
              )

            final_chart_test=(line_chart_test+scatter_chart_test).properties(
                                title='Testing Set',
                                width=400,
                                height=800
                            ).configure_title(
                                anchor='middle'
                            ).configure_legend(
                                orient='top'
                            ).configure_axis(
                                labelFontSize=12,
                                titleFontSize=14
                            )
            
            mse_train_score = mse(y_train, y_train_preds, squared=True)
            mae_train_score = mae(y_train, y_train_preds)

            mse_test_score = mse(y_test, y_test_preds, squared=True)
            mae_test_score = mae(y_test, y_test_preds)

            # Display the chart
            st.markdown("<h1 style='font-size: 32px; font-weight: bold;margin-right: 100px;'>Training set</h1>", unsafe_allow_html=True)
              
            st.write("MSE of Training: ", mse_train_score)
            st.write("MAE of Training: ", mae_train_score)
            st.write("    ")
            st.write("    ")   
            st.altair_chart(final_chart_train, use_container_width=True)       
            
            # Display the chart
            st.markdown("<h1 style='font-size: 32px; font-weight: bold;margin-right: 100px;'>Testing set</h1>", unsafe_allow_html=True)
            st.write("MSE of Testing: ", mse_test_score)
            st.write("MAE of Testing: ", mae_test_score)
            st.altair_chart(final_chart_test, use_container_width=True)

            coef=ml_instance.coef_
            intercept=ml_instance.intercept_

            coef=list(coef)
            factors=list(X.columns)
            
            display_coefficients(coefs_list,factors_list,intercept)

        if selected_sub_sub_tab == sub_sub_tab_titles[1]:
            X = data_for_ml_univariate['medIncome'].values
            y = data_for_ml_univariate['TARGET_deathRate'].values
            
            ml_instance = ML()
            # Call the split_data method to split your data into training and testing sets
            X_train, X_test, y_train, y_test = ml_instance.split_data(X, y)
            
            ml_instance.train_linear_regression(X_train.reshape(-1, 1), y_train)
            
            # calculate baseline
            y_mean = y_train.mean()
            y_base = np.full(y_train.shape, y_mean)
            mse_score = mse(y_train, y_base, squared=True)
            mae_score = mae(y_train, y_base)

            st.write("<h1 style='font-size: 32px; font-weight: bold;'>Baseline Model</h1>", unsafe_allow_html=True)
            st.write("MSE of Baseline: ", mse_score)
            st.write("MAE of Baseline: ", mae_score)

            st.write("    ")
            st.write("    ")

            y_train_preds = ml_instance.predict(X_train.reshape(-1, 1))
            y_test_preds = ml_instance.predict(X_test.reshape(-1, 1))

            line_chart_train = alt.Chart(pd.DataFrame({'x':X_train, 'y': y_train_preds})).mark_line(opacity=1, color='blue').encode(
                    x='x',
                    y='y'
                  )

            scatter_chart_train = alt.Chart(pd.DataFrame({'x':X_train, 'y': y_train})).mark_circle(opacity=1, color='red').encode(
                x='x',
                y='y'
              )

            final_chart_train=(line_chart_train+scatter_chart_train).properties(
                                        title='medIncome',
                                        width=400,
                                        height=800
                                    ).configure_title(
                                        anchor='middle'
                                    ).configure_legend(
                                        orient='top'
                                    ).configure_axis(
                                        labelFontSize=12,
                                        titleFontSize=14
                                    )


            line_chart_test = alt.Chart(pd.DataFrame({'x':X_test, 'y': y_test_preds})).mark_line(opacity=1, color='blue').encode(
                    x='x',
                    y='y'
                  )

            scatter_chart_test = alt.Chart(pd.DataFrame({'x':X_test, 'y': y_test})).mark_circle(opacity=1, color='red').encode(
                x='x',
                y='y'
              )

            final_chart_test=(line_chart_test+scatter_chart_test).properties(
                                title='medIncome',
                                width=400,
                                height=800
                            ).configure_title(
                                anchor='middle'
                            ).configure_legend(
                                orient='top'
                            ).configure_axis(
                                labelFontSize=12,
                                titleFontSize=14
                            )
            
            mse_train_score = mse(y_train, y_train_preds, squared=True)
            mae_train_score = mae(y_train, y_train_preds)

            mse_test_score = mse(y_test, y_test_preds, squared=True)
            mae_test_score = mae(y_test, y_test_preds)
            
            # Display the chart
            st.markdown("<h1 style='font-size: 32px; font-weight: bold;margin-right: 50px;'>Training set</h1>", unsafe_allow_html=True)
            st.write("MSE of Training: ", mse_train_score)
            st.write("MAE of Training: ", mae_train_score)
            st.altair_chart(final_chart_train, use_container_width=True)
            
            st.write("    ")
            st.write("    ")            

            # Display the chart
            st.markdown("<h1 style='font-size: 32px; font-weight: bold;margin-right: 100px;'>Testing set</h1>", unsafe_allow_html=True)
            st.write("MSE of Testing: ", mse_test_score)
            st.write("MAE of Testing: ", mae_test_score)
            st.altair_chart(final_chart_test, use_container_width=True)

            coef=ml_instance.coef_
            intercept=ml_instance.intercept_

            coef=list(coef)
            factors=list(X.columns)
            
            display_coefficients(coefs_list,factors_list,intercept)

    if selected_sub_tab == tab_titles[1]:
        X =data_for_ml_multivariate.drop(['TARGET_deathRate','avgDeathsPerYear','avgAnnCount','popEst2015','povertyPercent','MedianAgeMale',
            'MedianAgeFemale','PctPrivateCoverage','PctPrivateCoverageAlone',
            'PctEmpPrivCoverage','PctPublicCoverage','PctPublicCoverageAlone','PctOtherRace','PctWhite','PctHS25_Over','PctEmployed16_Over',
            'PctBachDeg18_24','PctBlack','PctAsian','Id','PctBachDeg25_Over','PctMarriedHouseholds',
           'PctUnemployed16_Over','PercentMarried','binnedInc','Geography'],axis=1)
        
        y = data_for_ml_multivariate['TARGET_deathRate']

        # st.write(X)
        # st.write(y)

        from sklearn.model_selection import train_test_split
        # Split the data into training and testing sets (60% training, 40% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        from sklearn.preprocessing import StandardScaler
        
        # Initialize the scaler
        scaler = StandardScaler()
        
        # Fit the scaler on the training data and transform both the training and testing data
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
                        
        # calculate baseline
        y_mean = y_train.mean()
        y_base = np.full(y_train.shape, y_mean)
        mse_score = mse(y_train, y_base, squared=True)
        mae_score = mae(y_train, y_base)
        st.write("<h1 style='font-size: 32px; font-weight: bold;'>Baseline Model</h1>", unsafe_allow_html=True)
        st.write("MSE of Baseline: ", mse_score)
        st.write("MAE of Baseline: ", mae_score)
        
        st.write("    ")
        st.write("    ")
    
        # Train the linear regression model
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        y_train_preds=reg.predict(X_train)
        mse_train_score = mse(y_train, y_train_preds, squared=True)
        mae_train_score = mae(y_train, y_train_preds)

        # Predict on the testing set
        y_test_preds = reg.predict(X_test)
        mse_test_score = mse(y_test, y_test_preds, squared=True)
        mae_test_score = mae(y_test, y_test_preds)

        # Training chart
        y_train_preds=pd.DataFrame(y_train_preds)
        y_train_preds=y_train_preds.rename(columns={0:"TARGET_deathRate_pred"})
        y_train=pd.DataFrame(y_train)
        y_train = y_train.reset_index(drop=True)
        y_train_preds = y_train_preds.reset_index(drop=True)

        # Combine the actual and predicted values into a single DataFrame
        data = pd.DataFrame({
            'Actual Target': y_train['TARGET_deathRate'],
            'Predicted Values': y_train_preds['TARGET_deathRate_pred']
        })

        # Create a perfect prediction line
        perfect_prediction_line = alt.Chart(data).mark_line(color='green', point=True).encode(
            x='Actual Target',
            y=alt.Y('Actual Target', scale=alt.Scale(domain=[100, data['Actual Target'].max()], nice=True)),
            tooltip=['Actual Target', 'Predicted Values']
        ).properties(
            width=200,
            height=800
        )
        
        # Create scatter plot for predicted values
        scatter_plot = alt.Chart(data).mark_circle(color='red', opacity=0.7).encode(
            x='Actual Target',
            y=alt.Y('Predicted Values', scale=alt.Scale(domain=[100, data['Predicted Values'].max()], nice=True)),
            tooltip=['Actual Target', 'Predicted Values']
        )

        # Combine the charts
        final_chart = (perfect_prediction_line + scatter_plot).properties(
            title='Training set',
            width=200,
            height=800
        ).configure_title(
            anchor='middle'
        ).configure_legend(
            orient='top'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )
        
        st.markdown("<h1 style='font-size: 32px; font-weight: bold;margin-right: 100px;'>Training Set</h1>", unsafe_allow_html=True)
        st.write("MSE of Training: ", mse_train_score)
        st.write("MAE of Training: ", mae_train_score)
        st.altair_chart(final_chart, use_container_width=True)
        st.write("    ")
        st.write("    ")
        
    
        # Test chart
        y_test_preds=pd.DataFrame(y_test_preds)
        y_test_preds=y_test_preds.rename(columns={0:"TARGET_deathRate_pred"})
        y_test=pd.DataFrame(y_test)
        y_test = y_test.reset_index(drop=True)
        y_test_preds = y_test_preds.reset_index(drop=True)

        # Combine the actual and predicted values into a single DataFrame
        data = pd.DataFrame({
            'Actual Target': y_test['TARGET_deathRate'],
            'Predicted Values': y_test_preds['TARGET_deathRate_pred']
        })

        # Create a perfect prediction line
        perfect_prediction_line = alt.Chart(data).mark_line(color='green', point=True).encode(
            x='Actual Target',
            y=alt.Y('Actual Target', scale=alt.Scale(domain=[100, data['Actual Target'].max()], nice=True)),
            tooltip=['Actual Target', 'Predicted Values']
        ).properties(
            width=200,
            height=800
        )
        
        # Create scatter plot for predicted values
        scatter_plot = alt.Chart(data).mark_circle(color='red', opacity=0.7).encode(
            x='Actual Target',
            y=alt.Y('Predicted Values', scale=alt.Scale(domain=[100, data['Predicted Values'].max()], nice=True)),
            tooltip=['Actual Target', 'Predicted Values']
        )

        # Combine the charts
        final_chart = (perfect_prediction_line + scatter_plot).properties(
            title='Testing Set',
            width=200,
            height=800
        ).configure_title(
            anchor='middle'
        ).configure_legend(
            orient='top'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ) 

        st.markdown("<h1 style='font-size: 32px; font-weight: bold;margin-right: 100px;'>Testing set</h1>", unsafe_allow_html=True)
        st.write("MSE of Testing: ", mse_test_score)
        st.write("MAE of Testing: ", mae_test_score)   
        st.altair_chart(final_chart, use_container_width=True)

        coef=ml_instance.coef_
        intercept=ml_instance.intercept_

        coef=list(coef)
        factors=list(X.columns)
            
        display_coefficients(coefs_list,factors_list,intercept)
   
    if selected_sub_tab == tab_titles[2]:
        X =data_for_ml_feature_engineering.drop(['TARGET_deathRate','avgDeathsPerYear','avgAnnCount','popEst2015','povertyPercent','MedianAgeMale',
            'MedianAgeFemale','PctPrivateCoverage','PctPrivateCoverageAlone','PctUnemployed16_Over',
            'PctEmpPrivCoverage','PctPublicCoverage','PctOtherRace','PctWhite','PctHS25_Over','PctEmployed16_Over',
            'PctBachDeg18_24','PctBlack','PctAsian','Id','PctBachDeg25_Over','PctMarriedHouseholds','MedianAge','binnedInc','PctPublicCoverageAlone',
           'PercentMarried'],axis=1)
        
        y = data_for_ml_feature_engineering['TARGET_deathRate']

        from sklearn.model_selection import train_test_split
        # Split the data into training and testing sets (60% training, 40% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        from sklearn.preprocessing import StandardScaler
        
        # Initialize the scaler
        scaler = StandardScaler()
        
        # Fit the scaler on the training data and transform both the training and testing data
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
                        
        # calculate baseline
        y_mean = y_train.mean()
        y_base = np.full(y_train.shape, y_mean)
        mse_score = mse(y_train, y_base, squared=True)
        mae_score = mae(y_train, y_base)
        st.write("<h1 style='font-size: 32px; font-weight: bold;'>Baseline Model</h1>", unsafe_allow_html=True)
        st.write("MSE of Baseline: ", mse_score)
        st.write("MAE of Baseline: ", mae_score)
        
        st.write("    ")
        st.write("    ")
    
        # Train the linear regression model
        reg = LinearRegression()
        reg.fit(X_train, y_train)

        y_train_preds=reg.predict(X_train)
        mse_train_score = mse(y_train, y_train_preds, squared=True)
        mae_train_score = mae(y_train, y_train_preds)

        # Predict on the testing set
        y_test_preds = reg.predict(X_test)
        mse_test_score = mse(y_test, y_test_preds, squared=True)
        mae_test_score = mae(y_test, y_test_preds)

        # Training chart
        y_train_preds=pd.DataFrame(y_train_preds)
        y_train_preds=y_train_preds.rename(columns={0:"TARGET_deathRate_pred"})
        y_train=pd.DataFrame(y_train)
        y_train = y_train.reset_index(drop=True)
        y_train_preds = y_train_preds.reset_index(drop=True)

        # Combine the actual and predicted values into a single DataFrame
        data = pd.DataFrame({
            'Actual Target': y_train['TARGET_deathRate'],
            'Predicted Values': y_train_preds['TARGET_deathRate_pred']
        })

        # Create a perfect prediction line
        perfect_prediction_line = alt.Chart(data).mark_line(color='green', point=True).encode(
            x='Actual Target',
            y=alt.Y('Actual Target', scale=alt.Scale(domain=[100, data['Actual Target'].max()], nice=True)),
            tooltip=['Actual Target', 'Predicted Values']
        ).properties(
            width=200,
            height=800
        )
        
        # Create scatter plot for predicted values
        scatter_plot = alt.Chart(data).mark_circle(color='red', opacity=0.7).encode(
            x='Actual Target',
            y=alt.Y('Predicted Values', scale=alt.Scale(domain=[100, data['Predicted Values'].max()], nice=True)),
            tooltip=['Actual Target', 'Predicted Values']
        )

        # Combine the charts
        final_chart = (perfect_prediction_line + scatter_plot).properties(
            title='Training Set',
            width=200,
            height=800
        ).configure_title(
            anchor='middle'
        ).configure_legend(
            orient='top'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )
        st.markdown("<h1 style='font-size: 32px; font-weight: bold;margin-right: 100px;'>Training Set</h1>", unsafe_allow_html=True)
        st.write("MSE of Training: ", mse_train_score)
        st.write("MAE of Training: ", mae_train_score)
        st.write("    ")
        st.write("    ")
        st.altair_chart(final_chart, use_container_width=True)
        
        # Test chart
        st.markdown("<h1 style='font-size: 32px; font-weight: bold;margin-right: 100px;'>Testing set</h1>", unsafe_allow_html=True)
        y_test_preds=pd.DataFrame(y_test_preds)
        y_test_preds=y_test_preds.rename(columns={0:"TARGET_deathRate_pred"})
        y_test=pd.DataFrame(y_test)
        y_test = y_test.reset_index(drop=True)
        y_test_preds = y_test_preds.reset_index(drop=True)

        # Combine the actual and predicted values into a single DataFrame
        data = pd.DataFrame({
            'Actual Target': y_test['TARGET_deathRate'],
            'Predicted Values': y_test_preds['TARGET_deathRate_pred']
        })

        # Create a perfect prediction line
        perfect_prediction_line = alt.Chart(data).mark_line(color='green', point=True).encode(
            x='Actual Target',
            y=alt.Y('Actual Target', scale=alt.Scale(domain=[100, data['Actual Target'].max()], nice=True)),
            tooltip=['Actual Target', 'Predicted Values']
        ).properties(
            width=200,
            height=800
        )
        
        # Create scatter plot for predicted values
        scatter_plot = alt.Chart(data).mark_circle(color='red', opacity=0.7).encode(
            x='Actual Target',
            y=alt.Y('Predicted Values', scale=alt.Scale(domain=[100, data['Predicted Values'].max()], nice=True)),
            tooltip=['Actual Target', 'Predicted Values']
        )

        # Combine the charts
        final_chart = (perfect_prediction_line + scatter_plot).properties(
            title='Testing Set',
            width=200,
            height=800
        ).configure_title(
            anchor='middle'
        ).configure_legend(
            orient='top'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )
        
        st.write("MSE of Testing: ", mse_test_score)
        st.write("MAE of Testing: ", mae_test_score)       
        st.altair_chart(final_chart, use_container_width=True)

        coef=ml_instance.coef_
        intercept=ml_instance.intercept_

        coef=list(coef)
        factors=list(X.columns)
        
        display_coefficients(coefs_list,factors_list,intercept)

elif selected_tab == "Analysis":
    display_univariate_analysis()
elif selected_tab == "Ethical Consideration":
    pass
elif selected_tab == "References":
    pass
elif selected_tab == "GitHub":
    pass
