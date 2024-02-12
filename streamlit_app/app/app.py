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

# Set Python path
current_dir = os.path.dirname(__file__)
parent_dir = str(Path(current_dir).resolve().parents[0])
sys.path.append(parent_dir)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from sklearn.linear_model import LinearRegression

from tab_df.logics import Dataset
from tab_eda.logics import EDA
from tab_df.display import display_tab_df_content
# from tab_eda.display import display_tab_eda_report
from tab_eda.display import display_missing_values,display_plots,display_correlation_heatmap,display_analysis,display_info,display_summary_statistics,display_stack_bar_chart, display_plot_distribution, display_generate_visual_eda_report
from tab_encoding.display import display_tab_df_encoding_explain, display_correlation_encoding_heatmap
from tab_encoding.logics import Encoding
from tab_ml.display import display_baseline_metrics,display_model_metrics,display_confusion_matrix,metric, display_roc_curve, display_metrics_and_visualizations,display_model_performance_analysis,display_cross_validation_analysis, feature_importance_explanation
from tab_ml.logics import ML
from tab_intro.introduction import display_introduction
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


# Sidebar navigation for different sections

selected_tab = st.sidebar.radio("Navigation", ["Introduction", "Data", "EDA","Encoding", "Machine Learning Model","Feature Importance","Deployment","Ethical Consideration", "References","GitHub"])

# Load data from "Data" tab
# Get the current directory of the script
current_dir = os.path.dirname(__file__)

training_set_path=Path(__file__).resolve().parent.parent / "Dataset" / "cancer_us_county-training.csv"
testing_set_path=Path(__file__).resolve().parent.parent / "Dataset" / "cancer_us_county-testing.csv"

df_train=pd.read_csv(training_set_path)
df_test=pd.read_csv(testing_set_path)

dataset=pd.concat([df_train,df_test],axis=0)

def remove_hyperlinks(html_content):
    # Remove anchor tags (<a>) from the HTML content
    return re.sub(r'<a\s+(?:[^>]*?\s+)?href="([^"]*)"[^>]*>', r'<span>\1</span>', html_content)

data_from_tab_df = pd.DataFrame(dataset.data)

eda = EDA(data_from_tab_df)

  # Display content based on selected sidebar tab
if selected_tab =="Introduction":
  pass
elif selected_tab == "Data":
  pass
elif selected_tab == "EDA":
  pass
elif selected_tab == "Machine Learning Model":
    pass
elif selected_tab == "Ethical Consideration":
  pass
elif selected_tab == "References":
  pass
elif selected_tab == "GitHub":
  pass



