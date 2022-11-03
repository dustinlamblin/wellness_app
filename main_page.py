import streamlit as st
from datetime import datetime, timedelta
import seaborn as sns
import numpy as np
import pandas as pd
import optuna
import sklearn
import matplotlib
import shap
import pickle
from wellness_app.loader.loader import *
from wellness_app.tuning_model.optimization import fit_simple_random_forest,prepare_datasets
from wellness_app.model_analysis.analysis import get_features_importances,plot_features_importance
from wellness_app.analytics.performance import (
    show_most_activities,
    most_frequent_activities,
    rebased_output
)
from pathlib import Path
FOLDER_PATH = Path(__file__).parent / "input_data"


def main():
    st.title("How to hack your wellness powered by A.I")
    st.markdown(
        """
        What you do, or dont do during your day, influence your well being.  
        There is scientific evidence around certain actions impacting your day positively (meditation, sport, eating healty) but also negatively (scrolling on Instagram, drinking alcool)  
        But everybody is different, and how effective do you think it impacts YOU  
        Based on your daily actions, we will determine what influence most the variable of your choice (hapiness, productivity, anxiety)
        We all have a sense on what influence our mood, but can you actually quantify it? And Which one impact you most ? Do you think you eat healty often? How often do you actually meditate?
        Lets find out. Be ready to face the numbers. 
    
        For additional information on how to use the algorithm, you can refer to the tutorial on the side bar, you will get a complete guide on how to build your data and on how to interpret the different results.  
        If you want a full deep dive into your data and get descriptive analytics, you can go into analytics  
        Please upload ... and be ready to change your life!"
        """
    )
    uploaded_file = st.file_uploader("Upload your file (csv format)")
    if (uploaded_file is None):
        input_data=load_input_data(FOLDER_PATH)
    else:
        input_data=pd.read_csv(uploaded_file)
    st.markdown("If you dont have a file, and just want to see what it looks like, click on **predict**")
    if (st.button("Predict")):
        st.write("This is the uploaded data:")
        st.dataframe(input_data.head())        
        target_name=input_data.columns[-1]
        st.header("I. Overall statistics")
        st.write("Descriptive analysis of your ",target_name)
        st.write("What your ",target_name," look like over time")
        rebased_target=rebased_output(input_data)
        st.line_chart(rebased_target)
        st.write("If we look at it from a trend perspective on a weekly basis")
        st.line_chart(rebased_target.rolling(window=5).mean())
        st.write("Below, the activities or actions you do more often:")
        frq_data=most_frequent_activities(input_data)
        st.write(frq_data)
        show_most_activities(frq_data)
        st.header("II. Machine learning applications")
        X_train,_,_,_=prepare_datasets(input_data)
        top_model=fit_simple_random_forest(input_data)
        features_imp=get_features_importances(top_model,X_train)
        st.pyplot(plot_features_importance(features_imp))
        target_name=input_data.columns[-1]
        st.write(features_imp.index[0]," and ",features_imp.index[1],"are the most important variables to explain your",target_name)
        st.write("On the other hand ", features_imp.index[len(features_imp)-1]," and ",features_imp.index[len(features_imp)-2],"seems to have little or no impact on ",target_name)
        
        st.markdown(
            """
            This tell us about which features impact your target GLOBALY . But it doesnt tell us the direction of the impact and which class it impacts. Do these variables impact happiness? or unhappiness?   
            We need to drill down and decompose it : answering which features impact which classes positively and negatively. So far we just have the big picture.    
            For exemple, meditating can make you happy.  
            But if you dont meditate, will you necessarly be unhappy? Not necessarly. On the other hand, scrolling on Instram can make your really unhappy. But not being on Instagram will not necessarly make a happy day for you.  
            """
        )
        st.write("We can get a better understanding of the impact of variables if you use the SHAP values (**please refer to the tutorial on the left bar for more information**)")
        st.write("We can get a better of impact with decomposition")
        shap_values = shap.TreeExplainer(top_model).shap_values(X_train)
        fig_shap_values=shap.summary_plot(shap_values, X_train, plot_type="bar")
        st.write(fig_shap_values)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig_shap_values)
        st.write("Bottom up approach:")
        st.write("Which variable impact most ",target_name)
        fig_shap_values_happy=shap.summary_plot(shap_values[2],X_train, plot_type="violin")
        st.pyplot(fig_shap_values_happy)
        st.write("Which variable impact most when you are **unhappy**, class=0")
        fig_shap_values_unhappy=shap.summary_plot(shap_values[0],X_train, plot_type="violin")
        st.pyplot(fig_shap_values_unhappy)        
        explainer=shap.Explainer(top_model)
        shap_values_wf=explainer(X_train)
        waterfall_graph_0=shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],shap_values_wf[0].values[:,0],feature_names=X_train.columns)
        waterfall_graph_1=shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],shap_values_wf[1].values[:,0],feature_names=X_train.columns)
        waterfall_graph_2=shap.plots._waterfall.waterfall_legacy(explainer.expected_value[2],shap_values_wf[2].values[:,0],feature_names=X_train.columns)
        st.pyplot(waterfall_graph_0)
        st.write("diff graph")
        st.plotly_chart(waterfall_graph_1)
        st.write("diff graph")
        st.pyplot(waterfall_graph_2)

if __name__=='__main__':
    
    main()