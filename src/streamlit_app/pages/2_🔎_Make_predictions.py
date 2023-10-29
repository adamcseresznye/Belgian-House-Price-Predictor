import catboost
import numpy as np
import pandas as pd
import streamlit as st

from data import utils
from models import predict_model

st.set_page_config(
    page_title="Make predictions",
    page_icon="🔎",
)


@st.cache_data
def fetch_data() -> pd.DataFrame:
    """
    Retrieves and returns selected columns from the most recent data file.

    Returns:
        pd.DataFrame: A DataFrame containing the specified columns from the latest data file.
    """

    columns_to_select = [
        "bedrooms",
        "state",
        "number_of_frontages",
        "street",
        "lng",
        "primary_energy_consumption",
        "bathrooms",
        "yearly_theoretical_total_energy_consumption",
        "surface_of_the_plot",
        "building_condition",
        "city",
        "lat",
        "cadastral_income",
        "living_area",
    ]

    most_recent_data = list(utils.Configuration.GIT_DATA.glob("*.gzip"))[-1]
    most_recent_data_df = pd.read_parquet(most_recent_data)[columns_to_select]

    return most_recent_data_df


@st.cache_resource
def fetch_model() -> catboost.CatBoostRegressor:
    """
    Load and return a CatBoost regression model.

    Returns:
        catboost.CatBoostRegressor: The loaded CatBoost regression model.
    """
    model = catboost.CatBoostRegressor()
    model.load_model(utils.Configuration.GIT_MODEL.joinpath("catboost_model"))

    return model


try:
    st.header("Define variables")
    st.image(
        "https://cf.bstatic.com/xdata/images/hotel/max1024x768/408003083.jpg?k=c49b5c4a2346b3ab002b9d1b22dbfb596cee523b53abef2550d0c92d0faf2d8b&o=&hp=1"
    )

    most_recent_data_df = fetch_data()

    col1, col2, col3 = st.columns(spec=3, gap="large")

    with col1:
        st.markdown("### Geography")
        state = st.selectbox(
            "In which region is the house located?",
            ((most_recent_data_df.state.unique())),
        )
        city = st.selectbox(
            "In which city is it situated?", ((most_recent_data_df.city.unique()))
        )
        street = st.selectbox(
            "On which street is it situated?", ((most_recent_data_df.street.unique()))
        )

        lat = st.number_input(
            "What is the estimated latitude of the location?", step=1.0, format="%.4f"
        )
        lng = st.number_input(
            "What is the estimated longitude of the location?", step=1.0, format="%.4f"
        )

    with col2:
        st.markdown("### Construction")
        building_condition = st.selectbox(
            "What is the condition of the building?",
            ((most_recent_data_df.building_condition.unique())),
        )
        bedrooms = st.number_input(
            "How many bedrooms does the property have?", step=1.0, format="%.0f"
        )
        bathrooms = st.number_input(
            "How many bathrooms does the property have?", step=1.0, format="%.0f"
        )
        number_of_frontages = st.number_input(
            "What is the count of frontages for this property?", step=1.0, format="%.0f"
        )
        surface_of_the_plot = st.number_input(
            "What is the total land area associated with this property in m2?",
            step=1.0,
            format="%.1f",
        )
        living_area = st.number_input(
            "What is the living area or the space designated for living within the property in m2?",
            step=1.0,
            format="%.1f",
        )

    with col3:
        st.markdown("### Energy, Taxes")
        yearly_theoretical_total_energy_consumption = st.number_input(
            "What is the estimated annual total energy consumption for this property?",
            step=1.0,
            format="%.1f",
        )
        primary_energy_consumption = st.number_input(
            "What is the primary energy consumption associated with this property?",
            step=1.0,
            format="%.1f",
        )
        cadastral_income = st.number_input(
            "What is the cadastral income or property tax assessment value for this property?",
            step=1.0,
            format="%.1f",
        )

    data = {
        "bedrooms": [bedrooms],
        "state": [state],
        "number_of_frontages": [number_of_frontages],
        "street": [street],
        "lng": [lng],
        "primary_energy_consumption": [primary_energy_consumption],
        "bathrooms": [bathrooms],
        "yearly_theoretical_total_energy_consumption": [
            yearly_theoretical_total_energy_consumption
        ],
        "surface_of_the_plot": [surface_of_the_plot],
        "building_condition": [building_condition],
        "city": [city],
        "lat": [lat],
        "cadastral_income": [cadastral_income],
        "living_area": [living_area],
    }
    X_test = pd.DataFrame(data)

    st.table(X_test)

    model = fetch_model()
    prediction = predict_model.predict_catboost(model=model, X=X_test)
    st.success(f"The predicted price for the house is {10** prediction[0]:,.0f} EUR")


except Exception as e:
    st.error(e)
