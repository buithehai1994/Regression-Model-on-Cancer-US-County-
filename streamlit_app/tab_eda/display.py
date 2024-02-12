import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st
from tab_eda.logics import EDA
import pandas as pd
from dataprep.eda import create_report
# import streamlit_pandas_profiling 
# from streamlit_pandas_profiling import st_profile_report

def display_summary_statistics(dataset):
    eda=EDA(dataset)
    statistics_report=eda.summary_statistics()
    st.write(statistics_report)

def display_info(dataset):
    eda=EDA(dataset)
    info_report=eda.info_statistics()
    st.write(info_report)

def display_missing_values(dataset):
    eda=EDA(dataset)
    info_missing_values=eda.missing_values()
    st.write(info_missing_values)

def display_plots(dataset):
    eda = EDA(dataset)
    columns_list = list(dataset.columns)
    
    for idx, column in enumerate(columns_list):
        st.write(f"### Plots for {column}")
         # Check if the column is numeric or non-numeric
        if dataset[column].dtype in ['int64', 'float64']:  # Numeric data
            plot_type = st.radio(f"Select plot type for {column}:",
                                 ("Box Plot", "Bar Plot"),
                                 key=f"{column}_radio_{idx}")  # Unique key for each radio button
            
            if plot_type == "Box Plot":
                eda.box_plot(column)
            elif plot_type == "Bar Plot":
                eda.bar_plot(column)
        else:  # Non-numeric data
            plot_type = st.radio(f"Select plot type for {column}:",
                                 ("Pie Plot", "Count Plot"),
                                 key=f"{column}_radio_{idx}")
            if plot_type == "Pie Plot":
                eda.pie_plot(column)
            elif plot_type == "Count Plot":
                eda.count_plot(column)

def display_generate_eda_report(data):
    eda = EDA(data)
    profile = eda.generate_visual_eda_report()
    
    if profile is not None:
        # Get the HTML content from the profiling report
        html_content = profile.to_html()

        st.markdown('<h1>EDA Report</h1>', unsafe_allow_html=True)
        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown(html_content, unsafe_allow_html=True)
    else:
        st.error("There was an issue generating the EDA report.")

def display_generate_visual_eda_report(data):
    eda = EDA(data)
    profile = eda.generate_visual_eda_report()
    
    # Get the HTML content from the profiling report
    html_content = profile.to_html()

    st.markdown('<h1>EDA Report</h1>', unsafe_allow_html=True)
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown(html_content, unsafe_allow_html=True)
    
# Function to generate and display word cloud for a given text
def display_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot()

            
def display_correlation_heatmap(dataset):
    eda = EDA(dataset)
    correlation_heatmap = eda.get_correlation_heatmap()
    st.altair_chart(correlation_heatmap, use_container_width=True)
    explanation_text = """
    
    The correlation heatmap above illustrates the relationship between variables in the dataset. Notably, it shows that the 'Target' variable has a low correlation with other features.

    For further analysis and modeling, the dataset will be structured with dependent variable 'y' and independent variables 'X' as follows:

    - **y = df_cleaned['Target']**
    - **X = df_cleaned.drop(['Target', 'ID'], axis=1)**

    This code segregates the 'Target' variable into 'y' and retains the remaining variables in 'X'. The 'ID' column is excluded. Subsequently, the dataset will be split into distinct sets with a 90-10 ratio:

    - **X_data, X_test, y_data, y_test = train_test_split(X, y, test_size=0.1, random_state=42)**
    - **X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.1, random_state=42)**

    Due to differing scales between the target variable and other features, scaling becomes essential for effective model training. This process standardizes all 'X' features:

    - **from sklearn.preprocessing import StandardScaler**
    - **sc = StandardScaler()**
    - **X_train = sc.fit_transform(X_train)**
    - **X_test = sc.transform(X_test)**
    - **X_val = sc.transform(X_val)**

    """
    
    st.markdown(explanation_text)  # Display the explanatory text

