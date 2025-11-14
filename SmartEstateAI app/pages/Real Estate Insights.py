import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNetCV
from sklearn.base import BaseEstimator, TransformerMixin


# 1. Custom Transformers

class TargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, target_col):
        self.target_col = target_col
        self.mapping_ = {}

    def fit(self, X, y=None):
        df = X.copy()
        df[self.target_col] = y
        self.mapping_ = df.groupby(X.columns[0])[self.target_col].mean().to_dict()
        return self

    def transform(self, X):
        return X.iloc[:, 0].map(self.mapping_).values.reshape(-1, 1)


class OrdinalMapper(BaseEstimator, TransformerMixin):
    def __init__(self, mapping):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.replace(self.mapping).values.reshape(-1, 1)


# 2. Load / Prepare Dataset

data = pd.read_csv(
    r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\forinsights.csv"
)

numeric_cols = ['bedRoom', 'bathroom', 'built_up_area', 'servant room']
binary_cols = ['property_type', 'luxury_category']
ordinal_cols = ['agePossession']
furnishing_cols = ['furnishing_type']
sector_col = ['sector']

age_mapping = {'new': 0, 'old': 1, 'under construction': 2}

furnishing_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_cols),
    ('bin', 'passthrough', binary_cols),
    ('age', OrdinalMapper(age_mapping), ordinal_cols),
    ('furn', furnishing_encoder, furnishing_cols),
    ('sector', TargetEncoder(target_col='price'), sector_col)
])

X = data.drop('price', axis=1)
y = data['price']

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', ElasticNetCV(cv=5, l1_ratio=0.5, alphas=np.logspace(-3, 3, 100), random_state=42))
])

pipeline.fit(X, y)


# 3. Simulation Function

def simulate_price_change(model_pipeline, row, feature_name, delta):
    row_copy = row.copy()
    original_price = row_copy['price'] if not pd.isna(row_copy['price']) else model_pipeline.predict(
        pd.DataFrame([row_copy.drop('price')]))[0]

    row_copy[feature_name] += delta
    X_new = pd.DataFrame([row_copy.drop('price')])
    new_price = model_pipeline.predict(X_new)[0]

    diff = new_price - original_price
    pct_diff = diff / original_price * 100

    return original_price, new_price, diff, pct_diff


# 4. Streamlit App UI

st.title("🏡 Real Estate Price Simulator")

st.markdown("""
Enhance property features and analyse how each change impacts the estimated price.
""")

st.markdown("---")

# ---------------------- INPUT SECTION ----------------------

st.subheader("🔧 Property Inputs")

col1, col2 = st.columns(2)

with col1:
    property_type = st.selectbox("Property Type", [0, 1], format_func=lambda x: "flat" if x == 0 else "house")
    bedRoom = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
    bathroom = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
    furnishing_type = st.selectbox("Furnishing Type", ['unfurnished', 'semi-furnished', 'furnished'])

with col2:
    sector = st.selectbox("Sector", sorted(data['sector'].unique()))
    built_up_area = st.number_input("Built-up Area (sqft)", min_value=200.0, max_value=10000.0, value=1200.0)
    servant_room = st.selectbox("Servant Room", [0, 1])
    luxury_category = st.selectbox("Luxury Category", [0, 1])

agePossession = st.selectbox("Age of Possession", ['new', 'old', 'under construction'])

st.markdown("---")

# Build input row
input_row = pd.Series({
    'property_type': property_type,
    'sector': sector,
    'bedRoom': bedRoom,
    'bathroom': bathroom,
    'agePossession': agePossession,
    'built_up_area': built_up_area,
    'servant room': servant_room,
    'furnishing_type': furnishing_type,
    'luxury_category': luxury_category,
    'price': np.nan
})


# ---------------------- SIMULATION TAB -----------------------

st.subheader("📈 Feature Simulation")

with st.container(border=True):
    feature_to_change = st.selectbox("Feature to Change", ['bedRoom', 'bathroom', 'built_up_area', 'servant room'])
    delta_value = st.number_input("Increase By", value=1.0, step=1.0)

    if st.button("Run Simulation"):
        original_price, new_price, diff, pct_diff = simulate_price_change(
            pipeline, input_row, feature_to_change, delta_value
        )

        st.success("Simulation Completed")

        colA, colB = st.columns(2)
        with colA:
            st.metric("Original Price (Cr)", f"{original_price:.2f}")
            st.metric("New Price (Cr)", f"{new_price:.2f}")
        with colB:
            st.metric("Change (Cr)", f"{diff:.2f}")
            st.metric("Percentage Change", f"{pct_diff:.2f}%")

st.markdown("---")

# ---------------------- INSIGHTS TAB -----------------------

tab1, tab2, tab3 = st.tabs(["🏙️ Sector Influence", "🛋️ Furnishing Effect", "📅 Age Possession Effect"])

# --- Sector Influence ---
with tab1:
    st.write("Average property prices across sectors:")
    sector_means = data.groupby('sector')['price'].mean().sort_values(ascending=False)
    st.dataframe(sector_means)

# --- Furnishing Effect ---
with tab2:
    furn_effects = []
    for furn in data['furnishing_type'].unique():
        row = input_row.copy()
        row['furnishing_type'] = furn
        X_new = pd.DataFrame([row.drop('price')])
        price_pred = pipeline.predict(X_new)[0]
        furn_effects.append({'Furnishing': furn, 'Predicted Price (crores)': round(price_pred, 2)})

    st.table(pd.DataFrame(furn_effects))

# --- Age Possession Effect ---
with tab3:
    age_effects = []
    for age, val in age_mapping.items():
        row = input_row.copy()
        row['agePossession'] = age
        X_new = pd.DataFrame([row.drop('price')])
        price_pred = pipeline.predict(X_new)[0]
        age_effects.append({'Age Category': age, 'Predicted Price (crores)': round(price_pred, 2)})

    st.table(pd.DataFrame(age_effects))








# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import ElasticNetCV
# from sklearn.base import BaseEstimator, TransformerMixin


# # 1. Custom Transformers

# class TargetEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, target_col):
#         self.target_col = target_col
#         self.mapping_ = {}
#     def fit(self, X, y=None):
#         df = X.copy()
#         df[self.target_col] = y
#         self.mapping_ = df.groupby(X.columns[0])[self.target_col].mean().to_dict()
#         return self
#     def transform(self, X):
#         return X.iloc[:,0].map(self.mapping_).values.reshape(-1,1)

# class OrdinalMapper(BaseEstimator, TransformerMixin):
#     def __init__(self, mapping):
#         self.mapping = mapping
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         return X.replace(self.mapping).values.reshape(-1,1)


# # 2. Load / Prepare Dataset


# data = pd.read_csv("dataset/forinsights.csv")  

# numeric_cols = ['bedRoom', 'bathroom', 'built_up_area', 'servant room']
# binary_cols = ['property_type', 'luxury_category']
# ordinal_cols = ['agePossession']
# furnishing_cols = ['furnishing_type']
# sector_col = ['sector']

# age_mapping = {'new':0, 'old':1, 'under construction':2}

# furnishing_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')


# preprocessor = ColumnTransformer(transformers=[
#     ('num', StandardScaler(), numeric_cols),
#     ('bin', 'passthrough', binary_cols),
#     ('age', OrdinalMapper(age_mapping), ordinal_cols),
#     ('furn', furnishing_encoder, furnishing_cols),
#     ('sector', TargetEncoder(target_col='price'), sector_col)
# ])

# X = data.drop('price', axis=1)
# y = data['price']

# pipeline = Pipeline(steps=[
#     ('preprocess', preprocessor),
#     ('model', ElasticNetCV(cv=5, l1_ratio=0.5, alphas=np.logspace(-3,3,100), random_state=42))
# ])

# pipeline.fit(X, y)


# # 3. Simulation Function

# def simulate_price_change(model_pipeline, row, feature_name, delta):
#     row_copy = row.copy()
#     original_price = row_copy['price'] if not pd.isna(row_copy['price']) else model_pipeline.predict(pd.DataFrame([row_copy.drop('price')]))[0]
#     row_copy[feature_name] += delta
#     X_new = pd.DataFrame([row_copy.drop('price')])
#     new_price = model_pipeline.predict(X_new)[0]
#     diff = new_price - original_price
#     pct_diff = diff / original_price * 100
#     return original_price, new_price, diff, pct_diff


# # 4. Streamlit App UI

# st.title("🏡 Real Estate Price Simulator")

# st.markdown("Adjust property features and simulate price changes. See how increasing bedrooms, plot area, or other attributes impacts price.")

# # User inputs
# property_type = st.selectbox("Property Type", [0,1], format_func=lambda x: "flat" if x==0 else "house")
# sector = st.selectbox("Sector", sorted(data['sector'].unique()))
# bedRoom = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
# bathroom = st.number_input("Bathrooms", min_value=1, max_value=5, value=2)
# agePossession = st.selectbox("Age of Possession", ['new','old','under construction'])
# built_up_area = st.number_input("Built-up Area (sqft)", min_value=200.0, max_value=10000.0, value=1200.0)
# servant_room = st.selectbox("Servant Room", [0,1])
# furnishing_type = st.selectbox("Furnishing Type", ['unfurnished','semi-furnished','furnished'])
# luxury_category = st.selectbox("Luxury Category", [0,1])

# # Build input row
# input_row = pd.Series({
#     'property_type': property_type,
#     'sector': sector,
#     'bedRoom': bedRoom,
#     'bathroom': bathroom,
#     'agePossession': agePossession,
#     'built_up_area': built_up_area,
#     'servant room': servant_room,
#     'furnishing_type': furnishing_type,
#     'luxury_category': luxury_category,
#     'price': np.nan
# })

# # Simulation controls
# st.subheader("📈 Simulate Feature Change")
# feature_to_change = st.selectbox("Feature to Change", ['bedRoom','bathroom','built_up_area','servant room'])
# delta_value = st.number_input("Increase By", value=1.0, step=1.0)

# if st.button("Simulate"):
#     original_price, new_price, diff, pct_diff = simulate_price_change(pipeline, input_row, feature_to_change, delta_value)
    
#     st.markdown(f"**Original Price:** {original_price:.2f} crores")
#     st.markdown(f"**Predicted New Price:** {new_price:.2f} crores")
#     st.markdown(f"**Price Change:** {diff:.2f} crores ({pct_diff:.2f}%)")


# # 5. Additional Insights

# st.subheader("🏙️ Sector Influence")
# sector_means = data.groupby('sector')['price'].mean().sort_values(ascending=False)
# st.dataframe(sector_means)

# st.subheader("🛋️ Furnishing Type Effect")
# furn_effects = []
# for furn in data['furnishing_type'].unique():
#     row = input_row.copy()
#     row['furnishing_type'] = furn
#     X_new = pd.DataFrame([row.drop('price')])
#     price_pred = pipeline.predict(X_new)[0]
#     furn_effects.append({'Furnishing': furn, 'Predicted Price (crores)': round(price_pred,2)})
# st.table(pd.DataFrame(furn_effects))

# st.subheader("📅 Age Possession Effect")
# age_effects = []
# for age, val in age_mapping.items():
#     row = input_row.copy()
#     row['agePossession'] = age
#     X_new = pd.DataFrame([row.drop('price')])
#     price_pred = pipeline.predict(X_new)[0]
#     age_effects.append({'Age Category': age, 'Predicted Price (crores)': round(price_pred,2)})
# st.table(pd.DataFrame(age_effects))
