from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
import streamlit as st
from tab_ml.logics import ML
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import auc

def display_baseline_metrics(y_train):
    ml = ML()
    baseline_accuracy = ml.calculate_baseline_metrics(y_train)
    st.write(f"Baseline Accuracy: {baseline_accuracy}")

def display_model_metrics(x,y,model,average='weighted'):
    """
        Display the evaluation metrics.

        Parameters:
        - y: tartet
        - x: variable
        - model: model name
        """
    y_pred=model.predict(x)
    accuracy=accuracy_score(y,y_pred)
    
    st.write("Evaluation Metrics:")
    st.write(f"Accuracy: {accuracy:.4f}")
    scores = precision_recall_fscore_support(y, y_pred, average)

    st.write("Scores (precision, recall, F1 score):", scores[:3])


def metric(model,X_train,X_test,X_val,y_train,y_test,y_val):
    # Load model
    ml=ML()
    ml.load_model(model)
    metrics_accuracy_training=ml.calculate_accuracy(X_train,y_train)
    metrics_accuracy_testing=ml.calculate_accuracy(X_test,y_test)
    metrics_accuracy_validation=ml.calculate_accuracy(X_val,y_val)

    # Create a DataFrame with all metrics
    metrics_data = {
        'Dataset': ['Training', 'Testing'],
        'Accuracy': [metrics_accuracy_training, metrics_accuracy_testing]
        }
    # Display all metrics in one table
    st.write("Metrics for Training and Testing")
    st.table(pd.DataFrame(metrics_data))
