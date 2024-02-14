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


def display_chart(X, y, y_preds):
    X = X.reshape(-1,1)
    y = y.flatten()  # Flatten y to ensure it's 1D
    y_preds = y_preds.flatten()  # Flatten y_preds to ensure it's 1D
    data = pd.DataFrame({'x': X.flatten(), 'y': y, 'y_preds': y_preds})
    
    scatter_chart = alt.Chart(data).mark_circle(opacity=1, color='red').encode(
        x='x',
        y='y'
    )

    line_chart = alt.Chart(data).mark_line(opacity=1, color='blue').encode(
        x='x',
        y='y_preds'
    )
    
    chart = scatter_chart + line_chart
    st.altair_chart(chart, use_container_width=True)

def display_multiple_chart(X, y_true, y_preds):
    # Combine features and true/predicted target values into a DataFrame
    df = pd.DataFrame({'Feature': X.flatten(), 'True': y_true, 'Predicted': y_pred})
    
    # Scatter plot of True vs. Predicted values
    scatter_plot = alt.Chart(df).mark_circle(size=60).encode(
        x='True',
        y='Predicted',
        color=alt.Color('Feature', scale=alt.Scale(scheme='category20')),
        tooltip=['Feature', 'True', 'Predicted']
    ).properties(
        title='True vs. Predicted values'
    )
    
    # Line chart of True vs. Predicted values
    line_chart = alt.Chart(df).mark_line().encode(
        x='Feature',
        y=alt.Y('value', title='Value'),
        color=alt.Color('variable', scale=alt.Scale(range=['blue', 'orange']), legend=alt.Legend(title="Type")),
        tooltip=['Feature', alt.Tooltip('value', title='Value'), 'variable']
    ).transform_fold(
        ['True', 'Predicted'],
        as_=['variable', 'value']
    ).properties(
        title='True vs. Predicted values by Feature'
    )
    chart=scatter_plot + line_chart
    return chart
