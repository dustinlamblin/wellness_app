import streamlit as st
from datetime import datetime, timedelta
import seaborn as sns
import numpy as np
import pandas as pd
import optuna
import sklearn
import shap
import pickle
from wellness_app.loader.loader import *
from wellness_app.tuning_model.optimization import fit_simple_random_forest,prepare_datasets
import matplotlib.pyplot as plt

FOLDER_PATH = Path.cwd().parent / "input_data"

def plot_features_importance(forest_importances):
    fig, ax = plt.subplots()
    sns.barplot(x=forest_importances.values,y=forest_importances.index,color="b")
    ax.set_title("Most impactful features")
    fig.tight_layout()
    return fig
    
def get_features_importances(top_model, X_train):
    importances=top_model.feature_importances_
    return pd.Series(importances, index=X_train.columns).sort_values(ascending=False)
