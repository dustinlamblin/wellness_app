import streamlit as st
from wellness_app.loader.loader import *
import pandas as pd
from wellness_app.tuning_model.optimization import *
import shap

def main():
    st.title("How to hack your wellness powered by A.I")
    st.markdown("Welcome to the tutorial")
    FOLDER_PATH = Path.cwd().parent / "input_data"
    st.write(FOLDER_PATH)
    input_data=load_input_data(FOLDER_PATH)
    st.write("You need to load a CSV file with the following format:")
    st.dataframe(input_data.head())
    st.markdown(
        """
        Target :  
        - The last column is the target, this the variable you try to explain, it can be happiness, productivity, stress...  
        - Use categorical value to define it, exemple: -1 is bad, 0 neutral, +1 is good  
        
        Variables :  
        - You can add as many variables (columns) you want that you think impact your wellness  
        - You can do the same with your variable:  0 if you didnt meditate that day and 1 if you did.  
        - If you have a social negative interaction, make 1 in the column, 0 when nothing happens.  
        
        Piece of advice: Fill the data at lunch and after-work, you will quickly increase your data set, it is better for machine learning applications, and you dont biased the impact of morning/afternoon. If you do so add 2columns morning_col with 1 when its morning, 0 otherwise, same for afternoon.
        """ 
    )
    top_model=load_model_from_folder(FOLDER_PATH)
    st.write("In this particular case, let's analyse what impact most my hapiness")
    X_train, y_train, X_test, y_test=prepare_datasets(input_data)
    st.markdown(
        """
        How to interpret result from SHAP values
        SHAP values is a concept from Game Theory, that tell us which variable contribute the most to the overall
    """
    )
    shap_values = shap.TreeExplainer(top_model).shap_values(X_train)
    fig_shap_values=shap.summary_plot(shap_values, X_train, plot_type="bar")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig_shap_values)
    st.write("Bottom up approach:")
    st.write("Which variable impact most when you are **happy**, class=2")
    fig_shap_values_happy=shap.summary_plot(shap_values[2],X_train, plot_type="violin")
    st.pyplot(fig_shap_values_happy)
    st.write("Which variable impact most when you are **unhappy**, class=0")
    fig_shap_values_unhappy=shap.summary_plot(shap_values[0],X_train, plot_type="violin")
    st.pyplot(fig_shap_values_unhappy)
        
if __name__=='__main__':
    main()