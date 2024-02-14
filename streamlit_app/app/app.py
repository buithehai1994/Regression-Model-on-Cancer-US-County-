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

# Set Python path
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parents[0])
sys.path.append(parent_dir)

from tab_eda.display import display_summary_statistics,display_info,display_missing_values
from tab_encoding.display import display_tab_df_encoding_explain, display_correlation_encoding_heatmap
from tab_encoding.logics import Encoding
from tab_ml.logics import ML
from tab_ml.display import display_baseline_metrics,display_model_metrics,display_line_chart,display_scatter_chart,display_chart,display_multiple_chart

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
    data_for_ml =encoding.multivarate_process()
    return data_for_ml
    
data_for_ml_univariate = perform_encoding()
data_for_ml_multivariate= perform_encoding_and_multivariate()

selected_tab = st.sidebar.radio("Navigation", ["Introduction", "Data", "EDA", "Encoding", "Machine Learning Model", "Feature Importance", "Deployment", "Ethical Consideration", "References", "GitHub"], key="navigation")

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

            # Call the split_data method to split your data into training and testing sets
            X_train, X_test, y_train, y_test = ml_instance.split_data(X, y)

            ml_instance = ML()
            ml_instance.train_linear_regression(X_train, y_train)
            
            # calculate baseline
            y_mean = y_train.mean()
            y_base = np.full(y_train.shape, y_mean)
            mse_score = mse(y_train, y_base, squared=True)
            mae_score = mae(y_train, y_base)
            st.write("MSE of Baseline: ", mse_score)
            st.write("MAE of Baseline: ", mae_score)

            st.write("    ")
            st.write("    ")

            # Reshape X_train and X_test
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)
            reg = LinearRegression()
            reg.fit(X_train, y_train)

            y_train_preds = reg.predict(X_train)
            y_test_preds = reg.predict(X_test)

            mse_train_score = mse(y_train, y_train_preds, squared=True)
            mae_train_score = mae(y_train, y_train_preds)

            mse_test_score = mse(y_test, y_test_preds, squared=True)
            mae_test_score = mae(y_test, y_test_preds)

            st.write("Training chart")
            display_chart(X_train,y_train,y_train_preds)

            st.write("MSE of Training: ", mse_train_score)
            st.write("MAE of Training: ", mae_train_score)
            st.write("    ")
            st.write("    ")
            
            st.write("Testing chart")
            display_chart(X_test,y_test,y_test_preds)

            st.write("MSE of Testing: ", mse_test_score)
            st.write("MAE of Testing: ", mae_test_score)

        if selected_sub_sub_tab == sub_sub_tab_titles[1]:
            X = data_for_ml_univariate['medIncome'].values
            y = data_for_ml_univariate['TARGET_deathRate'].values

            ml_instance = ML()
            # Call the split_data method to split your data into training and testing sets
            X_train, X_test, y_train, y_test = ml_instance.split_data(X, y)
            
            # calculate baseline
            y_mean = y_train.mean()
            y_base = np.full(y_train.shape, y_mean)
            mse_score = mse(y_train, y_base, squared=True)
            mae_score = mae(y_train, y_base)
            st.write("MSE of Baseline: ", mse_score)
            st.write("MAE of Baseline: ", mae_score)

            st.write("    ")
            st.write("    ")

            # Train the multilinear regression model
            ml_instance.train_linear_regression(X_train, y_train)

            y_train_preds = ml_instance.predict(X_train)
            mse_train_score = mse(y_train, y_train_preds, squared=True)
            mae_train_score = mae(y_train, y_train_preds)
            
            y_test_preds = ml_instance.predict(X_test)
            mse_test_score = mse(y_test, y_test_preds, squared=True)
            mae_test_score = mae(y_test, y_test_preds)

            y_train_preds=pd.DataFrame(y_train_preds)
            y_train_preds=y_train_preds.rename(columns={0:"TARGET_deathRate_pred"})
            y_train=pd.DataFrame(y_train)
            # Resetting the index of y_train and y_train_preds DataFrames
            y_train = y_train.reset_index(drop=True)
            y_train_preds = y_train_preds.reset_index(drop=True)
            st.write("Training chart")
            display_chart(X_train,y_train,y_train_preds)

            st.write("MSE of Training: ", mse_train_score)
            st.write("MAE of Training: ", mae_train_score)
            st.write("    ")
            st.write("    ")
            y_test_preds=pd.DataFrame(y_test_preds)
            y_test_preds=y_test_preds.rename(columns={0:"TARGET_deathRate_pred"})
            y_test=pd.DataFrame(y_test)
            y_test = y_test.reset_index(drop=True)
            y_test_preds = y_test_preds.reset_index(drop=True)
            st.write("Testing chart")
            display_chart(X_test,y_test,y_test_preds)

            st.write("MSE of Testing: ", mse_test_score)
            st.write("MAE of Testing: ", mae_test_score)

    if selected_sub_tab == tab_titles[1]:
        X =data_for_ml_multivariate.drop(['TARGET_deathRate','avgDeathsPerYear','avgAnnCount','popEst2015','povertyPercent','MedianAgeMale',
            'MedianAgeFemale','PctPrivateCoverage','PctPrivateCoverageAlone',
            'PctEmpPrivCoverage','PctPublicCoverage','PctPublicCoverageAlone','PctOtherRace','PctWhite','PctHS25_Over','PctEmployed16_Over',
            'PctBachDeg18_24','PctBlack','PctAsian','Id','PctBachDeg25_Over','PctMarriedHouseholds',
            'PctUnemployed16_Over','PercentMarried','binnedInc','Geography'],axis=1).values
        y = data_for_ml_multivariate['TARGET_deathRate']
                
        ml_instance = ML()
                
        # Call the split_data method to split your data into training and testing sets
        X_train, X_test, y_train, y_test = ml_instance.split_data(X, y)
        
        from sklearn.preprocessing import StandardScaler
        
        # Initialize the scaler
        scaler = StandardScaler()
                
        # Fit the scaler on the training data and transform both the training and testing data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
                
        # calculate baseline
        y_mean = y_train.mean()
        y_base = np.full(y_train.shape, y_mean)
        mse_score = mse(y_train, y_base, squared=True)
        mae_score = mae(y_train, y_base)
        st.write("MSE of Baseline: ", mse_score)
        st.write("MAE of Baseline: ", mae_score)
        
        st.write("    ")
        st.write("    ")
    
        # Train the multilinear regression model
        ml_instance.train_linear_regression(X_train_scaled, y_train)
        y_train_preds = ml_instance.predict(X_train_scaled)
        
        mse_train_score = mse(y_train, y_train_preds, squared=True)
        mae_train_score = mae(y_train, y_train_preds)
        
        st.write("Training chart")
        display_multiple_chart(X_train_scaled, y_train, y_train_preds)
        
        st.write("MSE of Training: ", mse_train_score)
        st.write("MAE of Training: ", mae_train_score)
        st.write("    ")
        st.write("    ")

        y_test_preds = ml_instance.predict(X_test_scaled)
        
        mse_test_score = mse(y_test, y_test_preds, squared=True)
        mae_test_score = mae(y_test, y_test_preds)
        
        st.write("Testing chart")
        display_multiple_chart(X_test_scaled, y_test, y_test_preds)
        
        st.write("MSE of Testing: ", mse_test_score)
        st.write("MAE of Testing: ", mae_test_score)
        st.write("    ")
        st.write("    ")
        
elif selected_tab == "Ethical Consideration":
    pass
elif selected_tab == "References":
    pass
elif selected_tab == "GitHub":
    pass
