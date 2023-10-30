from typing import List, Optional, Set, Tuple, Union

import catboost
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from lets_plot import *
from lets_plot.frontend_context._configuration import _as_html
from lets_plot.mapping import as_discrete
from lets_plot.plot.core import PlotSpec
from lets_plot.plot.plot import GGBunch
from sklearn import compose, impute, neighbors, pipeline, preprocessing

from data import pre_process, utils


def st_letsplot(plot, scrolling=True):
    """Embed a Let's Plot object within Streamlit app

    Parameters
    ----------
    plot:
        Let's Plot object
    scrolling: bool
        If content is larger than iframe size, provide scrollbars?

    Example
    -------
    >>> st_letsplot(p)
    """

    plot_dict = plot.as_dict()
    if isinstance(plot, PlotSpec):
        width, height = get_ggsize_or_default(plot_dict, default=500)
    elif isinstance(plot, GGBunch):
        # the inner list comprehension is a list of (width, height) tuples
        # the outer consists of two elements [sum(widths), sum(heights)]
        width, height = [
            sum(y)
            for y in zip(
                *[
                    get_ggsize_or_default(x["feature_spec"], default=500)
                    for x in plot_dict["items"]
                ]
            )
        ]
    else:
        height = 500
        width = 500

    # 20 an aribtrary pad to remove scrollbars from iframe, consider if worth removing
    return components.html(
        _as_html(plot_dict),
        height=height + 20,
        width=width + 20,
        scrolling=scrolling,
    )


def get_ggsize_or_default(plot_dict, default=500) -> (int, int):
    """
    Returns a tuple consisting of the width and height of the plot
    Lookup if there is a ggsize specification. If not return default value.
    :param plot_dict:
    :param default:
    :return: width, height
    """
    if "ggsize" in plot_dict.keys():
        return plot_dict["ggsize"]["width"], plot_dict["ggsize"]["height"]
    return default, default


st.set_page_config(
    page_title="Explore the data",
    page_icon="📈",
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
    1. Randomly shuffles the rows of the DataFrame.
    2. Converts the 'price' column to the base 10 logarithm.
    3. Fills missing values in categorical variables with 'missing value'.
    4. Separates the features (X) and the target (y).
    5. Identifies and filters out outlier values based on LocalOutlierFactor.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the dataset.

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: A tuple containing the prepared features (X) and the target (y).

    Example use case:
    ```python
    # Load your dataset into a DataFrame (e.g., df)
    df = load_data()

    # Prepare the data for modeling
    X, y = prepare_data_for_modelling(df)

    # Now you can use X and y for machine learning tasks.
    ```

    Args:
        df (pd.DataFrame): The input DataFrame containing the dataset.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the prepared features (X) and the target (y).
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
    most_recent_data_df = fetch_data()
    processed_most_recent_data_df = prepare_data_for_modelling(most_recent_data_df)

    st.header("Explore the data")
    st.map(
        processed_most_recent_data_df[["lat", "lng"]]
        .rename(columns={"lng": "lon"})
        .dropna(how="all")
        .sample(500, random_state=42)
    )
    st.caption(
        "A selection of 500 locations from where the ads were collected. Take a look and see if any of them seem familiar to you."
    )
    st.markdown(
        """Start by exploring the charts below to gain insights into the relationships between
                various variables and how they impact prices. Feel free to dive in and uncover more from the dataset."""
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
        "What specific factor are you interested in exploring?",
        (variables_to_choose_from),
        index=None,
        placeholder="Select a variable...",
    )
    if option in categorical_variables:
        box = processed_most_recent_data_df[[option, "price"]].pipe(
            lambda df: ggplot(
                df,
                aes(
                    as_discrete(option, order=1, order_by="..middle.."),
                    "price",
                ),
            )
            + geom_boxplot()
            + flavor_darcula()
            + scale_y_log10()
            + ggsize(700, 500)
        )
        st_letsplot(box)
    elif option in numerical_variables:
        linechart = processed_most_recent_data_df[[option, "price"]].pipe(
            lambda df: ggplot(df, aes(option, "price"))
            + geom_point()
            + scale_y_log10()
            + flavor_darcula()
            + ggsize(700, 500)
        )
        st_letsplot(linechart)
    else:
        st.write("Make your selection.")


except Exception as e:
    st.error(e)
