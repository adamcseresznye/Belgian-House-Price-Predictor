import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st
import streamlit.components.v1 as components
from sklearn import compose, impute, neighbors, pipeline, preprocessing

from data import pre_process, utils

sys.path.append(str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Explore the data",
    page_icon="📈",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://adamcseresznye.github.io/blog/",
        "Report a bug": "https://github.com/adamcseresznye/house_price_prediction",
        "About": "Explore and Predict Belgian House Prices with Immoweb Data and CatBoost!",
    },
)


@st.cache_data
def identify_outliers(df: pd.DataFrame) -> pd.Series:
    """
    Identify outliers in a DataFrame.

    This function uses a Local Outlier Factor (LOF) algorithm to identify outliers in a given
    DataFrame. It operates on both numerical and categorical features, and it returns a binary
    Series where `True` represents an outlier and `False` represents a non-outlier.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing features for outlier identification.

    Returns:
    - pd.Series: A Boolean Series indicating outliers (True) and non-outliers (False).

    Example:
    ```python
    # Load your DataFrame with features (df)
    df = load_data()

    # Identify outliers using the function
    outlier_mask = identify_outliers(df)

    # Use the outlier mask to filter your DataFrame
    filtered_df = df[~outlier_mask]  # Keep non-outliers
    ```

    Notes:
    - The function uses Local Outlier Factor (LOF) with default parameters for identifying outliers.
    - Numerical features are imputed using median values, and categorical features are one-hot encoded
    and imputed with median values.
    - The resulting Boolean Series is `True` for outliers and `False` for non-outliers.
    """

    # Extract numerical and categorical feature names
    NUMERICAL_FEATURES = df.select_dtypes("number").columns.tolist()
    CATEGORICAL_FEATURES = df.select_dtypes("object").columns.tolist()

    # Define transformers for preprocessing
    numeric_transformer = pipeline.Pipeline(
        steps=[("imputer", impute.SimpleImputer(strategy="median"))]
    )

    categorical_transformer = pipeline.Pipeline(
        steps=[
            ("encoder", preprocessing.OneHotEncoder(handle_unknown="ignore")),
            ("imputer", impute.SimpleImputer(strategy="median")),
        ]
    )

    # Create a ColumnTransformer to handle both numerical and categorical features
    preprocessor = compose.ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Initialize the LOF model
    clf = neighbors.LocalOutlierFactor()

    # Fit LOF to preprocessed data and make predictions
    y_pred = clf.fit_predict(preprocessor.fit_transform(df))

    # Adjust LOF predictions to create a binary outlier mask
    y_pred_adjusted = [1 if x == -1 else 0 for x in y_pred]
    outlier_mask = pd.Series(y_pred_adjusted) == 0

    return outlier_mask


@st.cache_data
def prepare_data_for_modelling(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for machine learning modeling.

    This function takes a DataFrame and prepares it for machine learning by performing the following steps:
    1. Fills missing values in categorical variables with 'missing value'.
    2. Identifies and filters out outlier values based on LocalOutlierFactor.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the dataset.

    Returns:
    - pd.DataFrame: The prepared DataFrame.

    Example use case:
    ```python
    # Load your dataset into a DataFrame (e.g., df)
    df = load_data()

    # Prepare the data for modeling
    prepared_data = prepare_data_for_modelling(df)

    # Now you can use prepared_data for machine learning tasks.
    ```

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.

    Returns:
        pd.DataFrame: The prepared DataFrame.
    """

    # Fill missing categorical variables with "missing value"
    for col in df.columns:
        if df[col].dtype.name in ("bool", "object", "category"):
            df[col] = df[col].fillna("missing value")

    outlier_mask = identify_outliers(df)

    return df.loc[outlier_mask, :]


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
        "price",
    ]

    most_recent_data = list(utils.Configuration.GIT_DATA.glob("*.gzip"))[-1]
    most_recent_data_df = pd.read_parquet(most_recent_data)[columns_to_select]

    return most_recent_data_df


try:
    st.header("Explore the data")

    with st.spinner("Loading data..."):
        most_recent_data_df = fetch_data()
    processed_most_recent_data_df = prepare_data_for_modelling(most_recent_data_df)

    st.map(
        processed_most_recent_data_df[["lat", "lng"]]
        .rename(columns={"lng": "lon"})
        .dropna(how="all")
        .sample(frac=0.15, random_state=42)
    )
    st.caption(
        "A selection of ~500 locations from where the ads were collected. Take a look and see if any of them seem familiar to you."
    )

    st.markdown(
        """Start by exploring the charts below to gain insights into the relationships between
                various variables and how they impact prices. Dive in and uncover more from the dataset."""
    )

    variables_to_choose_from = processed_most_recent_data_df.drop(
        columns=["price", "street", "lng", "lat", "city"]
    ).columns

    numerical_variables = list(
        {
            column: processed_most_recent_data_df[column].nunique()
            for column in variables_to_choose_from
            if processed_most_recent_data_df[column].nunique() > 20
        }.keys()
    )
    categorical_variables = list(
        {
            column: processed_most_recent_data_df[column].nunique()
            for column in variables_to_choose_from
            if processed_most_recent_data_df[column].nunique() < 20
        }.keys()
    )

    option = st.selectbox(
        "**What specific factor are you interested in exploring?**",
        (variables_to_choose_from),
        index=None,
        placeholder="Select a variable...",
    )
    if option in categorical_variables:
        sorted_index = (
            processed_most_recent_data_df.groupby(option)
            .price.median()
            .sort_values()
            .index.tolist()
        )
        boxplot = px.box(
            processed_most_recent_data_df,
            x=option,
            y="price",
            template="plotly_dark",
            log_y=True,
            category_orders=sorted_index,
            title=f"Analyzing the Effect of {option.replace('_', ' ').title()} on House Prices",
            labels={
                "price": "Price in Log10-Scale (EUR)",
                option: f"{option.replace('_', ' ').title()}",
            },
        )
        boxplot.update_xaxes(categoryorder="array", categoryarray=sorted_index)

        st.plotly_chart(boxplot, use_container_width=True, theme="streamlit")

    elif option in numerical_variables:
        scatter_plot = px.scatter(
            processed_most_recent_data_df,
            x=option,
            y="price",
            trendline="lowess",
            title=f"Analyzing the Effect of {option.replace('_', ' ').title()} on House Prices",
            trendline_color_override="#c91e01",
            opacity=0.5,
            height=500,
            labels={
                option: option.replace("_", " ").title(),
                "price": "Price in Log10-Scale (EUR)",
            },
            log_y=True,
        )
        st.plotly_chart(scatter_plot, use_container_width=True, theme="streamlit")
    else:
        st.info("Make your selection.")


except Exception as e:
    st.error(e)

st.sidebar.subheader("📢 Get in touch 📢")
cols1, cols2, cols3 = st.sidebar.columns(3)
cols1.markdown(
    "[![Foo](https://cdn3.iconfinder.com/data/icons/picons-social/57/11-linkedin-48.png)](https://www.linkedin.com/in/adam-cseresznye)"
)
cols2.markdown(
    "[![Foo](https://cdn1.iconfinder.com/data/icons/picons-social/57/github_rounded-48.png)](https://github.com/adamcseresznye)"
)
cols3.markdown(
    "[![Foo](https://cdn2.iconfinder.com/data/icons/threads-by-instagram/24/x-logo-twitter-new-brand-48.png)](https://twitter.com/csenye22)"
)
