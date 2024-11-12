import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.preprocessing import OneHotEncoder

# Load scaler
scaler = joblib.load('scaler.pkl')

df = pd.read_csv('train.csv')

# Clean column names
df.columns = df.columns.str.replace(r'[\s\.]', '_', regex=True)
df.columns = df.columns.str.replace(r'-', '_', regex=True)

# Handle NaNs in object columns (same as before)
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
object_columns = df.select_dtypes(include=['object']).columns

# Object column NaN handling as before
for column in object_columns:
    df[column] = df[column].apply(lambda x: x if isinstance(x, str) and len(x) == 1 and not x.replace('.', '', 1).isdigit() else None)
    none_indices = df[df[column].isna()].index
    non_none_values = df[column].dropna().unique()
    if len(non_none_values) > 0:
        np.random.seed(0)
        random_values = np.random.choice(non_none_values, size=len(none_indices))
        df.loc[none_indices, column] = random_values
        
x = df.drop(['id', 'class'], axis=1)
y = df[['class']]

# Streamlit app
def mantar_pred(cap_diameter, cap_shape, cap_surface, cap_color, does_bruise_or_bleed, gill_attachment, gill_spacing,
                gill_color, stem_height, stem_width, stem_root, stem_surface, stem_color, veil_type, veil_color,
                has_ring, ring_type, spore_print_color, habitat, season):

    input_data = pd.DataFrame({
        'cap_diameter': [cap_diameter],
        'cap_shape': [cap_shape],
        'cap_surface': [cap_surface],
        'cap_color': [cap_color],
        'does_bruise_or_bleed': [does_bruise_or_bleed],
        'gill_attachment': [gill_attachment],
        'gill_spacing': [gill_spacing],
        'gill_color': [gill_color],
        'stem_height': [stem_height],
        'stem_width': [stem_width],
        'stem_root': [stem_root],
        'stem_surface': [stem_surface],
        'stem_color': [stem_color],
        'veil_type': [veil_type],
        'veil_color': [veil_color],
        'has_ring': [has_ring],
        'ring_type': [ring_type],
        'spore_print_color': [spore_print_color],
        'habitat': [habitat],
        'season': [season]
    })

    # Apply scaling using the pre-fitted scaler
    numeric_cols = ['cap_diameter', 'stem_height', 'stem_width']
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # OneHotEncode categorical columns
    ohe = OneHotEncoder(handle_unknown='ignore')
    cat_cols = ['cap_shape', 'cap_surface', 'cap_color', 'does_bruise_or_bleed', 'gill_attachment', 'gill_spacing',
                'gill_color', 'stem_root', 'stem_surface', 'stem_color', 'veil_type', 'veil_color', 'has_ring',
                'ring_type', 'spore_print_color', 'habitat', 'season']
    
    input_data_cat = ohe.fit_transform(input_data[cat_cols]).toarray()

    # Combine numeric and categorical data
    input_data_transformed = np.concatenate([input_data[numeric_cols], input_data_cat], axis=1)

    model = joblib.load('Mantar.pkl')

    prediction = model.predict(input_data_transformed)
    return float(prediction[0])

# Streamlit UI
st.title("Mantar Classification Model")
st.write("Enter your data:")

cap_diameter = st.slider('cap_diameter', float(df['cap_diameter'].min()), float(df['cap_diameter'].max()))
cap_shape = st.selectbox('cap_shape', df['cap_shape'].unique())
cap_surface = st.selectbox('cap_surface', df['cap_surface'].unique())
cap_color = st.selectbox('cap_color', df['cap_color'].unique())
does_bruise_or_bleed = st.selectbox('does_bruise_or_bleed', df['does_bruise_or_bleed'].unique())
gill_attachment = st.selectbox('gill_attachment', df['gill_attachment'].unique())
gill_spacing = st.selectbox('gill_spacing', df['gill_spacing'].unique())
gill_color = st.selectbox('gill_color', df['gill_color'].unique())
stem_height = st.slider('stem_height', float(df['stem_height'].min()), float(df['stem_height'].max()))
stem_width = st.slider('stem_width', float(df['stem_width'].min()), float(df['stem_width'].max()))
stem_root = st.selectbox('stem_root', df['stem_root'].unique())
stem_surface = st.selectbox('stem_surface', df['stem_surface'].unique())
stem_color = st.selectbox('stem_color', df['stem_color'].unique())
veil_type = st.selectbox('veil_type', df['veil_type'].unique())
veil_color = st.selectbox('veil_color', df['veil_color'].unique())
has_ring = st.selectbox('has_ring', df['has_ring'].unique())
ring_type = st.selectbox('ring_type', df['ring_type'].unique())
spore_print_color = st.selectbox('spore_print_color', df['spore_print_color'].unique())
habitat = st.selectbox('habitat', df['habitat'].unique())
season = st.selectbox('season', df['season'].unique())

if st.button('Predict'):
    mantar = mantar_pred(cap_diameter, cap_shape, cap_surface, cap_color, does_bruise_or_bleed, gill_attachment, 
                         gill_spacing, gill_color, stem_height, stem_width, stem_root, stem_surface, stem_color, 
                         veil_type, veil_color, has_ring, ring_type, spore_print_color, habitat, season)
    st.write(f'The predicted class for the mushroom is: {mantar:.2f}')
