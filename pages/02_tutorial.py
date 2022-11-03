import streamlit as st
from wellness_app.loader.loader import *
import pandas as pd
from wellness_app.tuning_model.optimization import *
import shap
FOLDER_PATH = Path(__file__).parent.parent / "input_data"

def main():
    st.title("How to hack your wellness powered by A.I")
    st.markdown("Welcome to the tutorial")
    input_data=load_input_data(FOLDER_PATH)
    st.write("You need to load a CSV file with the following format:")
    st.dataframe(input_data.head())
    st.header("I. How to construct your model")
    st.markdown(
        """          
        Target :  
        - The last column is the target, this the variable you try to explain, it can be happiness, productivity, stress...  
        - Use categorical value to define it, exemple: -1 is bad, 0 neutral, +1 is good
        - Column name should we what you try to explain ("happiness", "productivity", etc...)
        
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
    st.header("II. How to interpret the result")
    st.header("What is SHAP")
    st.markdown(
        """
        Shapley values provide an understanding of the marginal contribution of a feature when building a predictive model. Feature importance scores are great for model explanation
        SHAP values is a concept from Game Theory, that tell us which variable contribute the most to the overall
    """
    )
    st.subheader("Top-down approach")
    shap_values = shap.TreeExplainer(top_model).shap_values(X_train)
    fig_shap_values=shap.summary_plot(shap_values, X_train, plot_type="bar")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(fig_shap_values)
    st.markdown(
        """
        This tell us about which features impact your target GLOBALLY. But it doesnt tell us the direction of the impact and which class it impacts. Do these variables impact happiness? or unhappiness?   
        We need to drill down and decompose it : answering which features impact which classes positively and negatively. So far we just have the big picture.    
        For exemple, meditating can make you happy.  
        But if you dont meditate, will you necessarly be unhappy? Not necessarly. On the other hand, scrolling on Instram can make your really unhappy. But not being on Instagram will not necessarly make a happy day for you.  
        This first graph explain the contribution of each variables to the overall output.  
        But if we want to understand which variables impact positively or negatively the output we need to drilldown a little more into the data
        """
    )
    st.subheader("Bottom-up approach")
    st.write("Which variable impact most when you are **happy**, class=2")
    fig_shap_values_happy=shap.summary_plot(shap_values[2],X_train, plot_type="violin")
    st.pyplot(fig_shap_values_happy)
    st.markdown("**How to interpret  results**")
    st.markdown(
        """
        Variables with high values are red, low values are blue.  
        Here we analyze what makes you happy, on the horizontal axis you can see which variables impact positively or negatively your happiness.   
        All the variables are ranked by order of importance  
        For exemple you can see that positive value in healthy food (**red dots**), means you eat healthy -> impact positively happinness (**shape values are positive**)  
        On the other hand :  negative value in sport (**blue dots**), means you didnt exercise -> impact negatively your hapiness (**shape values are negative**)
        The size of the violin inform you regar
        """
    )
    st.write("Which variable impact most when you are **unhappy**, class=0")
    fig_shap_values_unhappy=shap.summary_plot(shap_values[0],X_train, plot_type="violin")
    st.pyplot(fig_shap_values_unhappy)
        
if __name__=='__main__':
    main()