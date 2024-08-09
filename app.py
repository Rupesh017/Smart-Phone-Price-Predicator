import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

print(sklearn.__version__)
# Load the pipeline object
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Load the DataFrame object
with open('df.pkl', 'rb') as f:
    df = pickle.load(f)
st.title("Smartphone prediction")
# Brand
brand = st.selectbox("Brand", sorted(df["brand_name"].unique()))
rating = st.selectbox("Rating", sorted(df["rating"].unique()))
has_5g = st.selectbox("has_5g", sorted(['YES', 'NO']))
has_nfc = st.selectbox("has_nfc", ['YES', 'NO'])
has_ir_blaster = st.selectbox("has_ir_blaster", ['YES', 'NO'])
processor_brand = st.selectbox("processor", sorted(df["processor_brand"].unique()))
num_cores = st.selectbox("num_of_core", df['num_cores'].unique())
processor_speed = st.selectbox("processor Speed", sorted(df["processor_speed"].unique()))
battery_capacity = st.selectbox("Battery_capacity", sorted(df["battery_capacity"].unique()))
fast_charging = st.selectbox("fast_charging_watt", sorted(df["fast_charging"].unique()))
ram_capacity = st.selectbox("Ram_in_GB", sorted(df["ram_capacity"].unique()))
internal_memory = st.selectbox("Internal_memory(GB)", sorted(df["internal_memory"].unique()))
screen_size = st.selectbox("screen_size", sorted(df["screen_size"].unique()))
refresh_rate = st.selectbox("Refresh_rate", sorted(df["refresh_rate"].unique()))
resolution = st.selectbox("Resolution", sorted(df["resolution"].unique()))
num_rear_cameras = st.selectbox("number of rear camera", sorted(df["num_rear_cameras"].unique()))
os = st.selectbox("os", df['os'].unique())
primary_camera_rear = st.selectbox("primary_camera_rear", sorted(df["primary_camera_rear"].unique()))
primary_camera_front = st.selectbox("primary_camera_front", sorted(df["primary_camera_front"].unique()))
extended_memory_available = st.selectbox("extended_memory_available", ['YES', 'NO'])
extended_upto = st.selectbox("extended_upto", sorted(df["extended_upto"].unique()))
if st.button("Predict_price"):
    if has_5g == 'YES':
        has_5g = True
    else:
        has_5g = False
    if has_nfc == 'YES':
        has_nfc = True
    else:
        has_nfc = False
    if has_ir_blaster == 'YES':
        has_ir_blaster = True
    else:
        has_ir_blaster = False
    if extended_memory_available == 'Yes':
        extended_memory_available = 1
    else:
        extended_memory_available = 0
    query = np.array([brand, rating, has_5g, has_nfc,
                      has_ir_blaster, processor_brand, num_cores, processor_speed,
                      battery_capacity, fast_charging, ram_capacity, internal_memory,
                      screen_size, refresh_rate, resolution, num_rear_cameras, os,
                      primary_camera_rear, primary_camera_front,
                      extended_memory_available, extended_upto])

    columns = ['brand_name', 'rating', 'has_5g', 'has_nfc', 'has_ir_blaster',
               'processor_brand', 'num_cores', 'processor_speed', 'battery_capacity',
               'fast_charging', 'ram_capacity', 'internal_memory', 'screen_size',
               'refresh_rate', 'resolution', 'num_rear_cameras', 'os',
               'primary_camera_rear', 'primary_camera_front', 'extended_memory_available',
               'extended_upto']
    query = query.reshape(-1, 21)
    query = pd.DataFrame(query, columns=columns)
    st.write(query)
    # print(query)
    st.title(np.exp(pipeline.predict(query)))