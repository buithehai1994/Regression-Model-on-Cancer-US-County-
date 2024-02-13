from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
import streamlit as st
from tab_ml.logics import ML
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import altair as alt

def display_baseline_metrics(y_train):
    ml = ML()
    baseline_model = ml.calculate_baseline_metrics(y_train)
    return baseline_model


def display_model_metrics(model, X_train, X_test, y_train, y_test):
    # Load model
    ml = ML()
    ml.load_model(model)
    
    # Calculate metrics for training, testing, and validation datasets
    metrics_training = ml.calculate_model_metrics(X_train, y_train)
    metrics_testing = ml.calculate_model_metrics(X_test, y_test)

    # Create a DataFrame with all metrics
    metrics_data = {
        'Dataset': ['Training', 'Testing'],
        'MSE': [metrics_training["MSE score"], metrics_testing["MSE score"]],
        'MAE': [metrics_training["MAE score"], metrics_testing["MAE score"]]
    }

    # Display all metrics in one table
    st.write("Metrics for Training, Testing")
    st.table(pd.DataFrame(metrics_data))

def display_line_chart(X,y_preds):
    line_chart = alt.Chart(pd.DataFrame({'x':X, 'y': y_preds})).mark_line(opacity=1, color='blue').encode(
    x='x',
    y='y').properties(
        height=600,  # Adjust the height as desired
        width=400    # Adjust the width as desired
    )

    st.altair_chart(line_chart, use_container_width=True)

def display_scatter_chart(X,y):
    scatter_chart = alt.Chart(pd.DataFrame({'x':X, 'y': y})).mark_circle(opacity=1, color='red').encode(
        x='x',
        y='y'
      ).properties(
        height=600,  # Adjust the height as desired
        width=400    # Adjust the width as desired
    )

    st.altair_chart(scatter_chart, use_container_width=True)

def display_chart(X,y,y_preds):
    line_chart = alt.Chart(pd.DataFrame({'x':X, 'y': y_preds})).mark_line(opacity=1, color='blue').encode(
    x='x',
    y='y').properties(
        height=600,  # Adjust the height as desired
        width=400    # Adjust the width as desired
    )

    scatter_chart = alt.Chart(pd.DataFrame({'x':X, 'y': y})).mark_circle(opacity=1, color='red').encode(
        x='x',
        y='y'
      ).properties(
        height=600,  # Adjust the height as desired
        width=400    # Adjust the width as desired
    )

    chart= line_chart+scatter_chart
    st.altair_chart(chart, use_container_width=True)
