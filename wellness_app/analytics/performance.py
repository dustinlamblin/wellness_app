import streamlit as st


def rebased_output(input_data):
    """
    Rebased target  have score between 0 and 1
    """
    target=input_data.iloc[:,-1]
    xmin=min(target)
    xmax=max(target)
    return (target-xmin)/(xmax-xmin)
    
def rebased_input(input_data):
    """
    Rebased input data to have score between 0 and 1
    """
    input_reb=input_data.iloc[:,:-1].copy()
    for col in input_reb:
        xmin=min(input_reb[col])
        xmax=max(input_reb[col])
        input_reb[col]=(input_reb[col]-xmin)/(xmax-xmin)
    return input_reb

def most_frequent_activities(input_data):
    """
    Return the most frequent activities or features done by users on average.
    """
    input_reb=rebased_input(input_data)
    return input_reb.describe().T.sort_values("mean",ascending=False)[["mean"]]

def show_most_activities(frq_data):
    top_activity_1=frq_data.index[0]
    top_activity_2=frq_data.index[1]
    least_activity_1=frq_data.index[-1]
    least_activity_2=frq_data.index[-2]
    st.write(top_activity_1," and ", top_activity_2," are the activity that you practiced more often, while, ",least_activity_1," and ", least_activity_2," are your least frequent activities")