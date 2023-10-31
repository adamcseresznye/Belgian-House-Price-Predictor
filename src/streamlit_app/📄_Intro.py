import datetime

import numpy as np
import pandas as pd
import streamlit as st

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

st.write("# Welcome to the World of House Price Prediction!")

st.sidebar.success("Select an option.")

st.markdown(
    """
    This app is dedicated to predicting house prices in Belgium using data collected from [immoweb.be](https://www.immoweb.be/en),
    one of the leading real estate platforms in the country. With the most extensive collection of real estate ads from diverse locations,
    we've harnessed the power of a CatBoost model to provide you with accurate and up-to-date price predictions. Explore the housing market
    with confidence and make informed decisions using our powerful prediction tool.
"""
)
st.caption(
    """Disclaimer: The developer is not liable for the information provided by this app.
           It is intended for educational purposes only. Any use of the information for decision-making or financial
           purposes is at your own discretion and risk."""
)
