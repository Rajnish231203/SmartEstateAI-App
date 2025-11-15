import streamlit as st
import pickle
import pandas as pd
import numpy as np


# PAGE CONFIG

st.set_page_config(
    page_title="SmartEstateAI – Recommendations",
    page_icon="🏙️",
    layout="wide"
)


# LOAD DATA

location_df = pickle.load(open(
    r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\location_distance.pkl",
    'rb'
))

cosine_sim1 = pickle.load(open(
    r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\cosine_sim1.pkl",
    'rb'
))
cosine_sim2 = pickle.load(open(
    r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\cosine_sim2.pkl",
    'rb'
))
cosine_sim3 = pickle.load(open(
    r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\cosine_sim3.pkl",
    'rb'
))



# RECOMMENDER FUNCTION (UNCHANGED)

def recommend_properties_with_scores(property_name, top_n=5):
    cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3

    sim_scores = list(enumerate(
        cosine_sim_matrix[location_df.index.get_loc(property_name)]
    ))

    sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    top_indices = [i[0] for i in sorted_scores[1: top_n + 1]]
    top_scores = [i[1] for i in sorted_scores[1: top_n + 1]]

    top_properties = location_df.index[top_indices].tolist()

    recommendations_df = pd.DataFrame({
        'PropertyName': top_properties,
        'SimilarityScore': top_scores
    })

    return recommendations_df


# HEADER

st.markdown("""
# 🏙️ SmartEstateAI  
### Intelligent Recommendations Based on Location & Similarity
---
""")


# SECTION 1 — NEARBY LOCATION SEARCH

st.subheader("📍 Find Nearby Apartments")

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        selected_location = st.selectbox(
            "Choose a Location",
            sorted(location_df.columns.to_list())
        )

    with col2:
        radius = st.number_input(
            "Radius (in KMs)",
            min_value=1.0,
            step=0.5
        )

    if st.button("🔎 Search Nearby"):
        result_ser = location_df[location_df[selected_location] < radius * 1000][selected_location].sort_values()

        st.write("### Results:")
        if len(result_ser) == 0:
            st.warning("No locations found in this radius.")
        else:
            for key, value in result_ser.items():
                st.write(f"**{key}** — {round(value/1000, 1)} kms")


st.markdown("---")


# SECTION 2 — SIMILARITY RECOMMENDATION SYSTEM

st.subheader("🏢 Apartment Recommendations")

with st.container():
    colA, colB = st.columns([3, 1])

    with colA:
        selected_appartment = st.selectbox(
            "Select an Apartment",
            sorted(location_df.index.to_list())
        )

    if st.button("✨ Recommend Properties"):
        recommendation_df = recommend_properties_with_scores(selected_appartment)
        st.write("### Top Recommendations")
        st.dataframe(recommendation_df.style.format({"SimilarityScore": "{:.3f}"}))







# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np

# st.set_page_config(page_title="Recommend Appartments")

# location_df = pickle.load(open(r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\location_distance.pkl",'rb'))

# cosine_sim1 = pickle.load(open(r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\cosine_sim1.pkl",'rb'))
# cosine_sim2 = pickle.load(open(r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\cosine_sim2.pkl",'rb'))
# cosine_sim3 = pickle.load(open(r"C:\Users\rajni\CampusX Python\Projects\Capstone Projects\SmartEstateAI\SmartEstateAI app\datasets\cosine_sim3.pkl",'rb'))


# def recommend_properties_with_scores(property_name, top_n=5):
#     cosine_sim_matrix = 0.5 * cosine_sim1 + 0.8 * cosine_sim2 + 1 * cosine_sim3
#     # cosine_sim_matrix = cosine_sim3

#     # Get the similarity scores for the property using its name as the index
#     sim_scores = list(enumerate(cosine_sim_matrix[location_df.index.get_loc(property_name)]))

#     # Sort properties based on the similarity scores
#     sorted_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     # Get the indices and scores of the top_n most similar properties
#     top_indices = [i[0] for i in sorted_scores[1:top_n + 1]]
#     top_scores = [i[1] for i in sorted_scores[1:top_n + 1]]

#     # Retrieve the names of the top properties using the indices
#     top_properties = location_df.index[top_indices].tolist()

#     # Create a dataframe with the results
#     recommendations_df = pd.DataFrame({
#         'PropertyName': top_properties,
#         'SimilarityScore': top_scores
#     })

#     return recommendations_df


# # Test the recommender function using a property name
# recommend_properties_with_scores('DLF The Camellias')


# st.title('Recommendation By Nearby Locations')

# selected_location = st.selectbox('Location',sorted(location_df.columns.to_list()))

# radius = st.number_input('Radius in Kms')

# if st.button('Search'):
#     result_ser = location_df[location_df[selected_location] < radius*1000][selected_location].sort_values()

#     for key, value in result_ser.items():
#         st.text(str(key) + " " + str(round(value/1000)) + ' kms')

# st.title('Appartments Recommendations')
# selected_appartment = st.selectbox('Select an appartment',sorted(location_df.index.to_list()))

# if st.button('Recommend'):
#     recommendation_df = recommend_properties_with_scores(selected_appartment)

#     st.dataframe(recommendation_df)



