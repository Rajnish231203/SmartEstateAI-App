import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import ast


# PAGE SETUP

st.set_page_config(
    page_title="SmartEstateAI – Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
# 📊 SmartEstateAI Analytics  
Explore market insights, trends and patterns across sectors.
---
""")


# LOAD DATA

new_df = pd.read_csv(
    r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\data_viz1.csv"
)


# CLEANING NUMERIC COLUMNS

numeric_cols = ['price','price_per_sqft','built_up_area','latitude','longitude']

for col in numeric_cols:
    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

group_df = new_df.groupby('sector')[numeric_cols].mean().reset_index()


# 1️ SECTOR PRICE GEOMAP

st.markdown("## 🗺️ Sector Price-per-Sqft Map")

with st.container():
    col1, col2 = st.columns([4, 1])
    with col2:
        st.info("""
        **Map Info**  
        • Bubble size → Area  
        • Color → Price per sqft  
        """)
    
    fig = px.scatter_mapbox(
        group_df, 
        lat="latitude", 
        lon="longitude", 
        color="price_per_sqft",
        size='built_up_area',
        color_continuous_scale=px.colors.cyclical.IceFire, 
        zoom=10,
        mapbox_style="open-street-map",
        height=650,
        hover_name=group_df.index
    )

    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# 2️ WORDCLOUD

st.markdown("## ☁️ Features Wordcloud")

wordcloud_df = pd.read_csv(
    r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\wordcloud_df.csv"
)

colA, colB = st.columns([1.5, 4])

with colA:
    wc_sector = st.selectbox(
        "Select Sector",
        ["overall"] + sorted(wordcloud_df['sector'].dropna().unique())
    )

with colB:
    st.info("Wordcloud is generated using all unique features across selected sector.")

# Filter data
if wc_sector == "overall":
    temp_df = wordcloud_df
else:
    temp_df = wordcloud_df[wordcloud_df['sector'] == wc_sector]

main = []
for item in temp_df['features'].dropna():
    try:
        parsed = ast.literal_eval(item)
        main.extend(parsed)
    except:
        pass

feature_text = " ".join(main)

if feature_text.strip() == "":
    st.warning("No feature data available for this sector.")
else:
    wordcloud = WordCloud(
        width=800,
        height=600,
        background_color='black',
        min_font_size=10
    ).generate(feature_text)

    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)

st.markdown("---")


# 3️ AREA VS PRICE SCATTER

st.markdown("## 📈 Area vs Price Scatter Plot")

col1, col2 = st.columns([1.5, 4])
with col1:
    property_type = st.selectbox('Property Type', ['flat','house'])

with col2:
    st.info("Scatter shows price variation across different built-up areas.")

df_temp = new_df[new_df['property_type'] == property_type]

fig1 = px.scatter(
    df_temp,
    x="built_up_area",
    y="price",
    color="bedRoom",
    title=f"Area vs Price ({property_type})",
    height=500
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")


# 4️ BHK PIE CHART

st.markdown("## 🥧 BHK Distribution")

sector_options = new_df['sector'].unique().tolist()
sector_options.insert(0, 'overall')

selected_sector = st.selectbox("Sector", sector_options)

if selected_sector == 'overall':
    fig2 = px.pie(new_df, names='bedRoom', title="Overall BHK Distribution")
else:
    fig2 = px.pie(
        new_df[new_df['sector'] == selected_sector],
        names='bedRoom',
        title=f"BHK Distribution – Sector {selected_sector}"
    )

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")


# 5 BHK PRICE BOXPLOT

st.markdown("## 📦 BHK Price Comparison (Boxplot)")

fig3 = px.box(
    new_df[new_df['bedRoom'] <= 4],
    x='bedRoom',
    y='price',
    title="Price Range Across BHK"
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")


# 6️ PROPERTY TYPE PRICE DISTRIBUTION

st.markdown("## 📉 Price Distribution by Property Type")

fig4 = plt.figure(figsize=(10, 4))
sns.distplot(new_df[new_df['property_type'] == 'house']['price'], label='House')
sns.distplot(new_df[new_df['property_type'] == 'flat']['price'], label='Flat')
plt.legend()

st.pyplot(fig4)






# import streamlit as st
# import pandas as pd
# import plotly.express as px
# import pickle
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
# import seaborn as sns
# import ast

# # -------------------------------
# # BASIC SETUP
# # -------------------------------
# st.set_page_config(page_title="Plotting Demo")
# st.title('Analytics')

# # -------------------------------
# # LOAD DATA
# # -------------------------------
# new_df = pd.read_csv(r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\data_viz1.csv")

# # This is GLOBAL feature_text (overall city wordcloud)
# # feature_text = pickle.load(open('feature_text.pkl','rb'))


# # -------------------------------
# # SECTOR-WISE SUMMARY (FOR MAP)
# # -------------------------------
# numeric_cols = ['price','price_per_sqft','built_up_area','latitude','longitude']

# # Convert only these columns to numeric (force non-numeric → NaN)
# for col in numeric_cols:
#     new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

# # Group ONLY numeric columns (no object columns will be touched)
# group_df = new_df.groupby('sector')[numeric_cols].mean().reset_index()

# # -------------------------------
# # MAP PLOT: SECTOR PRICE PER SQFT
# # -------------------------------
# st.header('Sector Price per Sqft Geomap')

# fig = px.scatter_mapbox(
#     group_df, 
#     lat="latitude", 
#     lon="longitude", 
#     color="price_per_sqft", 
#     size='built_up_area',
#     color_continuous_scale=px.colors.cyclical.IceFire, 
#     zoom=10,
#     mapbox_style="open-street-map",
#     width=1200,
#     height=700,
#     hover_name=group_df.index
# )
# st.plotly_chart(fig, use_container_width=True)

# # -------------------------------
# # FEATURE WORDCLOUD (CURRENTLY ONLY OVERALL)
# # -------------------------------

# st.header('Features Wordcloud')

# wordcloud_df = pd.read_csv(r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\wordcloud_df.csv")

# wc_sector = st.selectbox(
#     "Select Sector for Wordcloud",
#     ["overall"] + sorted(wordcloud_df['sector'].dropna().unique())
# )

# # Filter by selected sector
# if wc_sector == "overall":
#     temp_df = wordcloud_df
# else:
#     temp_df = wordcloud_df[wordcloud_df['sector'] == wc_sector]

# # Extract all features (each row has a list stored as a string)
# main = []
# for item in temp_df['features'].dropna():
#     try:
#         parsed = ast.literal_eval(item)
#         main.extend(parsed)
#     except:
#         pass

# # Final text
# feature_text = " ".join(main)


# if len(feature_text.strip()) == 0:
#     st.warning("No features available for this sector.")
# else:
#     # Generate WordCloud
#     wordcloud = WordCloud(
#         width=800, 
#         height=800,
#         background_color='black',
#         min_font_size=10
#     ).generate(feature_text)

#     # Create a Matplotlib figure explicitly
#     fig, ax = plt.subplots(figsize=(8, 8))
#     ax.imshow(wordcloud, interpolation='bilinear')
#     ax.axis("off")

#     # Pass the figure to Streamlit
#     st.pyplot(fig)


# # -------------------------------
# # AREA VS PRICE SCATTER
# # -------------------------------
# st.header('Area Vs Price')

# property_type = st.selectbox('Select Property Type', ['flat','house'])

# if property_type == 'house':
#     fig1 = px.scatter(
#         new_df[new_df['property_type'] == 'house'], 
#         x="built_up_area", 
#         y="price", 
#         color="bedRoom", 
#         title="Area Vs Price"
#     )
# else:
#     fig1 = px.scatter(
#         new_df[new_df['property_type'] == 'flat'], 
#         x="built_up_area", 
#         y="price", 
#         color="bedRoom",
#         title="Area Vs Price"
#     )

# st.plotly_chart(fig1, use_container_width=True)

# # -------------------------------
# # BHK PIE CHART (SECTOR-WISE OR OVERALL)
# # -------------------------------
# st.header('BHK Pie Chart')

# sector_options = new_df['sector'].unique().tolist()
# sector_options.insert(0,'overall')

# selected_sector = st.selectbox('Select Sector', sector_options)

# if selected_sector == 'overall':
#     fig2 = px.pie(new_df, names='bedRoom')
# else:
#     fig2 = px.pie(
#         new_df[new_df['sector'] == selected_sector], 
#         names='bedRoom'
#     )

# st.plotly_chart(fig2, use_container_width=True)

# # -------------------------------
# # BHK PRICE COMPARISON (BOXPLOT)
# # -------------------------------
# st.header('Side by Side BHK price comparison')

# fig3 = px.box(
#     new_df[new_df['bedRoom'] <= 4], 
#     x='bedRoom', 
#     y='price', 
#     title='BHK Price Range'
# )
# st.plotly_chart(fig3, use_container_width=True)

# # -------------------------------
# # PRICE DISTRIBUTION BY PROPERTY TYPE
# # -------------------------------
# st.header('Side by Side Distplot for property type')

# fig3 = plt.figure(figsize=(10, 4))
# sns.distplot(new_df[new_df['property_type'] == 'house']['price'], label='house')
# sns.distplot(new_df[new_df['property_type'] == 'flat']['price'], label='flat')

# plt.legend()
# st.pyplot(fig3)



