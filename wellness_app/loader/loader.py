import pickle
import pandas as pd
from pathlib import Path
import streamlit as st

def load_model_from_folder(FOLDER_PATH):
    path_saved_model = FOLDER_PATH / "happiness_model.sav"
    with open(path_saved_model, "rb") as model_file:
        finalized_model = pickle.load(model_file)
    return finalized_model

def load_input_data(FOLDER_PATH):
    path_saved_model = FOLDER_PATH / "happiness.csv"
    return pd.read_csv(path_saved_model)