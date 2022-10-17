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

FOLDER_PATH = Path.cwd().parent / "input_data"


def main():
    st.title("How to hack your wellness powered by A.I")
    st.markdown(
        """
        Based on what you do, or dont do, we will determine which actions influence most your wellness.  
        For exemple, based on certain actions (meditation, sport, eating healty, scrolling on Instagram, drinking alcool), we will determine what influence most your hapiness, productivity, or which of these actions triggers anxiety\
        There is scientific evidence around certain actions: meditating, eating healty. But everybody is different, and how effective do you think it impacts YOU\ 
        We all have a sense on what influence our mood, but can you actually quantify it? Which one actuallz impact you most ? Do you think you eat healty often? \ 
        How often actually do you eat healthy. Lets find out. Be ready to face the numbers. Because numbers dont lie.\
        <br>
        For additional information on how to use the algorithm, you can refer to the tutorial on the side bar, you will get a complete guide on how to interpret the different results.  
        If you want a full deep dive into your data and get descriptive analytics, you can go into analytics  
        Please upload ... and be ready to change your life!"
        """
    )
    uploaded_file = st.file_uploader("Upload your file (csv format)")
    if (uploaded_file is None):
        input_data=load_input_data(FOLDER_PATH)
    else:
        input_data=pd.read_csv(uploaded_file)
    if (st.button("Predict")):
        st.write("This is the uploaded data:")
        st.dataframe(input_data.head())
        st.write("... Please wait while we are running the algorithm...")
        X_train,_,_,_=prepare_datasets(input_data)
        top_model=fit_simple_random_forest(input_data)
        features_imp=get_features_importances(top_model,X_train)
        st.pyplot(plot_features_importance(features_imp))
        target_name=input_data.columns[-1]
        st.write(features_imp.index[0]," and ",features_imp.index[1],"are the most important variables to explain your",target_name)
        st.write("On the other hand ", features_imp.index[len(features_imp)-1]," and ",features_imp.index[len(features_imp)-2],"seems to have little or no impact on ",target_name)
        
        st.markdown(
            """
            This tell us about which features impact GLOBALY your target. But it doesnt tell us the direction of the impact and which class it impacts. Do these variables impact happiness? or unhappiness?   
            We need to drill down and decompose it : answering which features impact which classes positively and negatively. So far we just have the big picture.    
            For exemple, meditating can make you happy.  
            But if you dont meditate, will you necessarly be unhappy? Not necessarly. On the other hand, scrolling on Instram can make your really unhappy. But not being on Instagram will not necessarly make a happy day for you.  
            """
        )
        st.write("We can get a better of impact with decomposition")
        
        shap_values = shap.TreeExplainer(top_model).shap_values(X_train)
        fig_shap_values=shap.summary_plot(shap_values, X_train, plot_type="bar")
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot(fig_shap_values)
        st.write("Bottom up approach:")
        st.write("Which variable impact most ",target_name)
        fig_shap_values_happy=shap.summary_plot(shap_values[2],X_train, plot_type="violin")
        st.pyplot(fig_shap_values_happy)
        st.write("Which variable impact most when you are **unhappy**, class=0")
        fig_shap_values_unhappy=shap.summary_plot(shap_values[0],X_train, plot_type="violin")
        st.pyplot(fig_shap_values_unhappy)        

#     # here we define some of the front end elements of the web page like 
#     # the font and background color, the padding and the text to be displayed
#     html_temp = """
#     <div style ="background-color:yellow;padding:13px">
#     <h1 style ="color:black;text-align:center;">Streamlit Iris Flower Classifier ML App </h1>
#     </div>
#     """
      
#     # this line allows us to display the front end aspects we have 
#     # defined in the above code
#     st.markdown(html_temp, unsafe_allow_html = True)
      
#     # the following lines create text boxes in which the user can enter 
#     # the data required to make the prediction
#     sepal_length = st.text_input("Sepal Length", "Type Here")
#     sepal_width = st.text_input("Sepal Width", "Type Here")
#     petal_length = st.text_input("Petal Length", "Type Here")
#     petal_width = st.text_input("Petal Width", "Type Here")
#     result =""
      
#     # the below line ensures that when the button called 'Predict' is clicked, 
#     # the prediction function defined above is called to make the prediction 
#     # and store it in the variable result
#     if st.button("Predict"):
#         result = prediction(sepal_length, sepal_width, petal_length, petal_width)
#     st.success('The output is {}'.format(result))
     
if __name__=='__main__':
    
    main()