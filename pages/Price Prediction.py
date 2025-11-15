import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="SmartEstate Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; margin: auto;}
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    border: none;
    font-size: 18px;
    width: 100%;
}
div.stButton > button:hover {background-color: #45a049; color: white;}
</style>
""", unsafe_allow_html=True)

# Load dataset for dropdowns
with open(r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\df.pkl",'rb') as f:
    df = pickle.load(f)

st.markdown("""
<h1 style='text-align:center; font-size:46px; margin-bottom:0;'>🏠 SmartEstate Price Predictor</h1>
<p style='text-align:center; font-size:18px; color:#555; margin-top:5px;'>AI-powered instant property price valuation</p>
""", unsafe_allow_html=True)

st.subheader("📌 Property Details")

col1, col2, col3 = st.columns(3)

with col1:
    property_type = st.selectbox('Property Type', ['flat','house'])
    bedrooms = float(st.selectbox('Bedrooms', sorted(df['bedRoom'].unique().tolist())))
    bathroom = float(st.selectbox('Bathrooms', sorted(df['bathroom'].unique().tolist())))
    balcony = st.selectbox('Balconies', sorted(df['balcony'].unique().tolist()))

with col2:
    sector = st.selectbox('Sector', sorted(df['sector'].unique().tolist()))
    built_up_area = float(st.number_input('Built Up Area (sqft)', min_value=100.0, step=50.0))
    servant_room = float(st.selectbox('Servant Room', [0.0, 1.0]))
    store_room = float(st.selectbox('Store Room', [0.0, 1.0]))

with col3:
    property_age = st.selectbox('Property Age', sorted(df['agePossession'].unique().tolist()))
    furnishing_type = st.selectbox('Furnishing Type', sorted(df['furnishing_type'].unique().tolist()))
    luxury_category = st.selectbox('Luxury Category', sorted(df['luxury_category'].unique().tolist()))
    floor_category = st.selectbox('Floor Category', sorted(df['floor_category'].unique().tolist()))

center_btn = st.columns([3,1,3])[1]
with center_btn:
    predict_btn = st.button("🔮 Predict Price", use_container_width=True)

# ------------------ PIPELINE DEFINITION (no .pkl) ------------------
columns_to_encode = ['property_type', 'balcony', 'luxury_category', 'floor_category']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), columns_to_encode),
        ('cat1', OneHotEncoder(drop='first', handle_unknown='ignore'), ['sector', 'agePossession', 'furnishing_type'])
    ],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit pipeline on a small sample to reduce memory usage
sample_df = df.sample(min(500, len(df)), random_state=42)
X_sample = sample_df.drop('price', axis=1, errors='ignore')
y_sample = np.log1p(sample_df['price']) if 'price' in sample_df else np.zeros(len(sample_df))
pipeline.fit(X_sample, y_sample)

# ------------------ PREDICTION ------------------
if predict_btn:
    data = [[property_type, sector, bedrooms, bathroom, balcony, property_age,
             built_up_area, servant_room, store_room,
             furnishing_type, luxury_category, floor_category]]

    columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
               'agePossession', 'built_up_area', 'servant room', 'store room',
               'furnishing_type', 'luxury_category', 'floor_category']

    one_df = pd.DataFrame(data, columns=columns)

    base_price = np.expm1(pipeline.predict(one_df))[0]
    low = base_price - 0.22
    high = base_price + 0.22

    st.markdown("<h2 style='text-align:center; margin-top:20px;'>🎯 Estimated Price Range</h2>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="
        margin-top: 10px;
        padding: 30px;
        border-radius: 16px;
        background: white;
        border: 1px solid #ddd;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        text-align:center;
    ">
        <h2 style='color:#4CAF50; font-size:38px; margin:0;'>₹ {round(low,2)} Cr — ₹ {round(high,2)} Cr</h2>
        <p style='margin-top:10px; font-size:17px; color:#555;'>Estimated market value based on your inputs.</p>
    </div>
    """, unsafe_allow_html=True)




# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# st.set_page_config(page_title="Viz Demo")



# with open(r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\df.pkl",'rb') as file:
#     df = pickle.load(file)

# with open(r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\pipeline.pkl",'rb') as file:
#     pipeline = pickle.load(file)


# st.header('Enter your inputs')

# # property_type
# property_type = st.selectbox('Property Type',['flat','house'])

# # sector
# sector = st.selectbox('Sector',sorted(df['sector'].unique().tolist()))

# bedrooms = float(st.selectbox('Number of Bedroom',sorted(df['bedRoom'].unique().tolist())))

# bathroom = float(st.selectbox('Number of Bathrooms',sorted(df['bathroom'].unique().tolist())))

# balcony = st.selectbox('Balconies',sorted(df['balcony'].unique().tolist()))

# property_age = st.selectbox('Property Age',sorted(df['agePossession'].unique().tolist()))

# built_up_area = float(st.number_input('Built Up Area'))

# servant_room = float(st.selectbox('Servant Room',[0.0, 1.0]))
# store_room = float(st.selectbox('Store Room',[0.0, 1.0]))

# furnishing_type = st.selectbox('Furnishing Type',sorted(df['furnishing_type'].unique().tolist()))
# luxury_category = st.selectbox('Luxury Category',sorted(df['luxury_category'].unique().tolist()))
# floor_category = st.selectbox('Floor Category',sorted(df['floor_category'].unique().tolist()))

# if st.button('Predict'):

#     # form a dataframe
#     data = [[property_type, sector, bedrooms, bathroom, balcony, property_age, built_up_area, servant_room, store_room, furnishing_type, luxury_category, floor_category]]
#     columns = ['property_type', 'sector', 'bedRoom', 'bathroom', 'balcony',
#                'agePossession', 'built_up_area', 'servant room', 'store room',
#                'furnishing_type', 'luxury_category', 'floor_category']

#     # Convert to DataFrame
#     one_df = pd.DataFrame(data, columns=columns)

#     #st.dataframe(one_df)

#     # predict
#     base_price = np.expm1(pipeline.predict(one_df))[0]
#     low = base_price - 0.22
#     high = base_price + 0.22

#     # display
#     st.text("The price of the flat is between {} Cr and {} Cr".format(round(low,2),round(high,2)))
    
    
 