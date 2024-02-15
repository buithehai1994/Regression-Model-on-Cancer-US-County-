import streamlit as st
from tab_encoding.logics import Encoding
from tab_eda.logics import EDA

def display_code_explanation():
    explanation_text = """
    # Code Explanation
    
    **Using LabelEncoder for Categorical Encoding:**
    
    The provided code snippet utilizes LabelEncoder instances to encode categorical features, 'Geography' and 'binnedInc', into numeric labels.

    **1. Encoding 'Geography':**
    
    The 'Geography' feature likely represents geographic regions or locations. By encoding it using a LabelEncoder, the categorical values are converted into numeric labels. This enables machine learning algorithms to process geographic information effectively.
    
    **2. Encoding 'binnedInc':**
    
    The 'binnedInc' feature likely represents income levels that have been binned or grouped into categories. Encoding it allows the algorithm to interpret income-related information numerically, facilitating computations and analysis involving income data.
    
    Encoding categorical features is a crucial preprocessing step in machine learning tasks, as it transforms non-numeric data into a format that algorithms can handle.
    """
    st.markdown(explanation_text)

def display_correlation_encoding_heatmap(dataset):
    eda = EDA(dataset)
    correlation_heatmap = eda.get_correlation_heatmap()
    st.altair_chart(correlation_heatmap, use_container_width=True)
    st.write(dataset.head())
    comment="""
        """
    st.map()
