import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st
def main():

    st.title('Breast Cancer Prediction')
    data = pd.read_csv('')