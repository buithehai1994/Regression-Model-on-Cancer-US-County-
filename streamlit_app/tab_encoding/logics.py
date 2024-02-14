
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
        
    def multivariate_process(self):
        if self.data is None:
            raise ValueError("No data available. Use set_data() to load data first.")
        

        self.data = self.data.copy()
        self.data=self.data.drop('PctSomeCol18_24',axis=1)
        self.data['PctEmployed16_Over'] = self.data['PctEmployed16_Over'].fillna(self.data['PctEmployed16_Over'].mean())
        self.data['PctPrivateCoverageAlone'] = self.data['PctPrivateCoverageAlone'].fillna(self.data['PctPrivateCoverageAlone'].mean())
        self.data=self.data[(self.data['TARGET_deathRate']<=240) & (self.data['TARGET_deathRate']>=120.2)]
        return self.data

    def feature_engineering_process(self):
        if self.data is None:
            raise ValueError("No data available. Use set_data() to load data first.")
        
        self.data = self.data.copy()
        self.data=self.data.drop('PctSomeCol18_24',axis=1)
        self.data['PctEmployed16_Over'] = self.data['PctEmployed16_Over'].fillna(self.data['PctEmployed16_Over'].mean())
        self.data['PctPrivateCoverageAlone'] = self.data['PctPrivateCoverageAlone'].fillna(self.data['PctPrivateCoverageAlone'].mean())
        self.data=self.data[(self.data['TARGET_deathRate']<=240) & (self.data['TARGET_deathRate']>=120.2)]

        # Calculate the mean income per capita for each decile
        mean_income_per_decile = self.data.groupby('binnedInc')['medIncome'].mean()
        
        # Replace each decile value with its corresponding mean income
        self.data['binnedInc'] = self.data['binnedInc'].map(mean_income_per_decile)
        self.data=self.data.drop(['Geography'],axis=1)

        self.data['avgAnnCount']=np.log1p(self.data['avgAnnCount'])
        self.data['avgDeathsPerYear']=np.log1p(self.data['avgDeathsPerYear'])
        self.data['incidenceRate']=np.log1p(self.data['incidenceRate'])
        self.data['medIncome']=np.log1p(self.data['medIncome'])
        self.data['popEst2015']=np.log1p(self.data['popEst2015'])
        self.data['povertyPercent']=np.log1p(self.data['povertyPercent'])
        self.data['studyPerCap']=np.log1p(self.data['studyPerCap'])
        self.data['AvgHouseholdSize']=np.log1p(self.data['AvgHouseholdSize'])
        self.data['PercentMarried']=np.log1p(self.data['PercentMarried'])
        self.data['PctNoHS18_24']=np.log1p(self.data['PctNoHS18_24'])
        self.data['PctHS18_24']=np.log1p(self.data['PctHS18_24'])
        self.data['PctBachDeg18_24']=np.log1p(self.data['PctBachDeg18_24'])
        self.data['PctHS25_Over']=np.log1p(self.data['PctHS25_Over'])
        self.data['PctBachDeg25_Over']=np.log1p(self.data['PctBachDeg25_Over'])
        self.data['PctEmployed16_Over']=np.log1p(self.data['PctEmployed16_Over'])
        self.data['PctUnemployed16_Over']=np.log1p(self.data['PctUnemployed16_Over'])
        self.data['PctPrivateCoverage']=np.log1p(self.data['PctPrivateCoverage'])
        self.data['PctPrivateCoverageAlone']=np.log1p(self.data['PctPrivateCoverageAlone'])
        self.data['PctEmpPrivCoverage']=np.log1p(self.data['PctEmpPrivCoverage'])
        self.data['PctPublicCoverage']=np.log1p(self.data['PctPublicCoverage'])
        self.data['PctPublicCoverageAlone']=np.log1p(self.data['PctPublicCoverageAlone'])
        self.data['PctWhite']=np.log1p(self.data['PctWhite'])
        self.data['PctBlack']=np.log1p(self.data['PctBlack'])
        self.data['PctAsian']=np.log1p(self.data['PctAsian'])
        self.data['PctOtherRace']=np.log1p(self.data['PctOtherRace'])
        self.data['PctMarriedHouseholds']=np.log1p(self.data['PctMarriedHouseholds'])
        self.data['BirthRate']=np.log1p(self.data['BirthRate'])
        self.data['MedianAge']=np.log1p(self.data['MedianAge'])
        self.data['MedianAgeMale']=np.log1p(self.data['MedianAgeMale'])
        self.data['MedianAgeFemale']=np.log1p(self.data['MedianAgeFemale'])
        self.data['binnedInc']=np.log1p(self.data['binnedInc'])

        self.data=self.data[(self.data['MedianAge']<4.002)&(self.data['MedianAge']>=3.5)]
        self.data=self.data[(self.data['MedianAgeMale']<=3.965)&(self.data['MedianAgeMale']>=3.453)]
        self.data=self.data[(self.data['MedianAgeFemale']<=3.987)&(self.data['MedianAgeFemale']>=3.535)]
        self.data=self.data[(self.data['avgAnnCount']<=9.042)]
        self.data=self.data[(self.data['popEst2015']<=13.355)&(self.data['popEst2015']>=7.365)]
        self.data=self.data[(self.data['BirthRate']<=2.386) & (self.data['BirthRate']>=1.328)]

        self.data=self.data[(self.data['AvgHouseholdSize']>=1.122)&(self.data['AvgHouseholdSize']<=1.381)]
        self.data=self.data[(self.data['TARGET_deathRate']<=240) & (self.data['TARGET_deathRate']>=120.2)]
        self.data=self.data[(self.data['avgDeathsPerYear']<=7.093)&(self.data['avgDeathsPerYear']>=1.609)]
        self.data=self.data[(self.data['PctBachDeg25_Over']<=3.49)&(self.data['PctBachDeg25_Over']>=1.668)]
        self.data=self.data[(self.data['PctBachDeg18_24']<=3.157)&(self.data['PctBachDeg18_24']>=0.531)]
        
        return self.data
        
    @st.cache
    def head_df(self):
        if self.data is not None:
            return self.data.head()
        else:
            return "No data available"
