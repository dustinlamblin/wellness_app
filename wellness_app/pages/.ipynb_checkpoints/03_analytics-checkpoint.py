import streamlit as st
from wellness_app.loader.loader import *
import pandas as pd
from wellness_app.tuning_model.optimization import *
import shap
import matplotlib.pyplot as plt
import seaborn as sns

def plot_target(input_data):
    return input_data.iloc[:,-1].rolling(window=5).mean().plot()
    

def main():
    st.title("Descriptive analytics")
    FOLDER_PATH = Path.cwd().parent / "input_data"
    input_data=load_input_data(FOLDER_PATH)
    target_name=input_data.columns[-1]
    st.write("Descriptive analysis of your ",target_name)
    st.write("If we follow the nomenclature: 1 is good, 0 is rather neutral and -1 is bad")
    st.write("Here is what we can say about you data, and what we shoudl work on")
    # fig_target=plot_target(input_data)
    st.write(input_data.iloc[:,-1].rolling(window=5).mean().plot())
    
    

if __name__=='__main__':
    main()