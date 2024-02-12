import streamlit as st
from tab_encoding.logics import Encoding
from tab_eda.logics import EDA

def display_tab_df_encoding_explain(dataset):
    st.subheader("Encoded DataFrame:")
    
    explanation_text = """
    """
    
    st.markdown(explanation_text)
    st.write(dataset.head())  # Display or return the encoded data frame

def display_correlation_encoding_heatmap(dataset):
    eda = EDA(dataset)
    correlation_heatmap = eda.get_correlation_heatmap()
    st.altair_chart(correlation_heatmap, use_container_width=True)
    st.write(dataset.head())
    comment="""
        """
    st.map()
