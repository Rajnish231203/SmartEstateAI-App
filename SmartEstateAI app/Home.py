import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="SmartEstate AI – Gurgaon Real Estate Analytics",
    page_icon="🏙️",
    layout="wide"
)

st.markdown("""
<h1 style="text-align:center; font-size: 42px; margin-bottom: -5px;">
🏙️ SmartEstate AI – Gurgaon Real Estate Analytics
</h1>
<p style="text-align:center; font-size: 18px; color: #666;">
AI-powered analytics | Price Prediction | Recommendations | Market Insights
</p>
""", unsafe_allow_html=True)

st.markdown("---")

left, right = st.columns([1.5, 1])

with left:
    st.subheader("🚀 Next-Gen Real Estate Intelligence")
    st.markdown("""
SmartEstate AI provides data-driven insights for real estate in Gurgaon:

- **AI-powered price prediction**
- **Smart property recommendations**
- **Interactive analytical dashboards**
- **Sector-level spatial insights**
""")

with right:
    st.image("https://cdn-icons-png.flaticon.com/512/854/854878.png", use_container_width=True)

st.markdown("---")

st.subheader("📦 Platform Modules")
mod1, mod2, mod3, mod4 = st.columns(4)

with mod1:
    st.markdown("### 📊 Analytics\n- Sector & spatial analysis\n- Price distribution\n- Sqft vs Price\n- Feature importance")

with mod2:
    st.markdown("### 💰 Price Prediction\n- Multiple ML models\n- Feature engineering\n- Real-time predictions")

with mod3:
    st.markdown("### 🏡 Recommendations\n- Content & collaborative filtering\n- Hybrid engine\n- Similarity scoring")

with mod4:
    st.markdown("### 📈 Market Insights\n- Trend analysis\n- Investment opportunities\n- Comparative market overview")

# st.markdown("---")

# st.subheader("⚡ Quick Navigation")
# colA, colB, colC, colD = st.columns(4)

# with colA:
#     st.page_link("Vizualizations and Analysis", label="📊 Analytics Dashboard")
# with colB:
#     st.page_link("Price Prediction", label="💰 Price Prediction")
# with colC:
#     st.page_link("Appartments Recommendation", label="🏡 Recommendations")
# with colD:
#     st.page_link("Real Estate Insights", label="📈 Market Insights")

st.markdown("---")

st.markdown("""
<p style="text-align:center; font-size:14px; color:#999; margin-top:30px;">
SmartEstate AI • Built with Streamlit • Powered by ML
</p>
""", unsafe_allow_html=True)
