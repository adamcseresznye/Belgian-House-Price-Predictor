import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="📄",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://adamcseresznye.github.io/blog/",
        "Report a bug": "https://github.com/adamcseresznye/house_price_prediction",
        "About": "Explore and Predict Belgian House Prices with Immoweb Data and CatBoost!",
    },
)


st.write("# Forecasting Belgian Property Prices with CatBoost!")

st.subheader("Introduction")

st.markdown(
    """
    This app is designed to predict house prices in Belgium using data gathered from [immoweb.be](https://www.immoweb.be/en), a prominent real
    estate platform in the country. Leveraging a CatBoost model, we aim to offer accurate and current price predictions.
    Explore the housing market and make informed choices with our prediction tool.
"""
)

file_path = Path(__file__).parent
image_path = file_path.joinpath("diagram.jpg")
image = Image.open(image_path)

st.image(
    image,
    caption="Data Acquisition, Processing, Model Training, and Performance Testing Workflow.",
)

st.subheader("Describing the Workflow")
st.markdown(
    """From the diagram, you can see that our data processing pipeline adheres to the traditional Extract, Transform, and Load (ETL) process.
            Initially, we extract data from the source using the `request_html` library. Following this, we execute multiple steps to refine the raw data,
            encompassing datatype conversion from strings to numerical values and converting low cardinal numerical data to boolean. Furthermore, we leverage
            the Google Maps API to enhance location precision and obtain coordinates based on the provided location information in the advertisements.
            """
)
st.markdown(
    """Once our data is prepared, we perform the essential step of splitting it into training and test sets. This separation ensures unbiased model
            performance evaluation later on. It's worth noting that during the project's experimental phases, we diligently evaluated various ML algorithms, including
            decision trees, `XGBoost`, and more. After rigorous experimentation, we selected `Catboost` due to its exceptional overall performance."""
)
st.markdown(
    """Upon loading our pre-defined and optimized hyperparameters, we are fully equipped to train and subsequently assess the model's performance using the test set.
            The results of the model's performance are saved for reference.
            This entire pipeline is initiated on a monthly basis through GitHub actions, ensuring that both our model and dataset remain up-to-date."""
)

st.markdown(
    """Visit the _"Explore data"_ page to gain insights into the factors influencing house prices in Belgium. Once you've grasped these principles,
            feel free to make predictions using the features available on the _"Make predictions"_ page and assess your accuracy based on our model.
            Have fun!🎈🎉😊🐍💻🎈"""
)

st.markdown(
    """**Planned Enhancements**:
- :white_check_mark: Collaborate with the Kaggle community to share initial data and gather insights on potential model improvements and data preprocessing techniques for enhanced predictions.
- :construction: Incorporate confidence intervals into predictions.
- :construction: Implement data upload to a database for enhanced data management.
- :construction: Explore the inclusion of advertisement time to account for seasonality in the model.
- :construction: Enhance loading speed.
"""
)

st.caption(
    """Disclaimer: The developer is not liable for the information provided by this app.
           It is intended for educational purposes only. Any use of the information for decision-making or financial
           purposes is at your own discretion and risk."""
)
with st.sidebar:
    st.subheader("📢 Get in touch 📢")
    st.markdown(
        "[![Title]('https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg')]('https://www.linkedin.com/in/adam-cseresznye')"
    )
    st.markdown(
        "[![Title]('https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png')]('https://github.com/adamcseresznye')"
    )
    st.markdown(
        "[![Title]('https://about.twitter.com/content/dam/about-twitter/x/brand-toolkit/logo-black.png.twimg.1920.png')]('https://twitter.com/csenye22')"
    )
