
from sklearn.preprocessing import LabelEncoder
import numpy as np
import streamlit as st

class Encoding:
    def __init__(self, data):
        # Initialize Dataset attributes
        self.data = data  # Placeholder for dataset

    def label_encoding(self):
        if self.data is None:
            raise ValueError("No data available. Use set_data() to load data first.")
        

        self.data = self.data.copy()
        
        # Initialize the LabelEncoder
        geography_encoder = LabelEncoder()
        binnedInc_encoder = LabelEncoder()
        
        # Fit and transform the data
        self.data['Geography'] = geography_encoder.fit_transform(self.data['Geography'])
        self.data['binnedInc'] = binnedInc_encoder.fit_transform(self.data['binnedInc'])
        return self.data

    @st.cache
    def head_df(self):
        if self.data is not None:
            return self.data.head()
        else:
            return "No data available"