import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import geocoder
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from sklearn import compose, impute, model_selection, neighbors, pipeline, preprocessing
from tqdm import tqdm

from data import utils

load_dotenv()

# Set up logging
logging.basicConfig(
    filename=utils.Configuration.INTERIM_DATA_PATH / "build_features_error.log",
    filemode="w",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def pre_process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses and cleans the input DataFrame for data analysis.

    This function performs various preprocessing steps on the input DataFrame:
    - Renames columns to follow a consistent naming convention.
    - Extracts numeric values from specified columns and converts them to float.
    - Maps boolean values in specified columns to True, False, or None.
    - Performs data cleaning and type conversion for specific columns.

    Args:
        df (pd.DataFrame): The input DataFrame to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed DataFrame ready for analysis.

    Example:
        To preprocess a DataFrame for analysis:
        >>> data = pd.read_csv("raw_data.csv")
        >>> preprocessed_data = pre_process_dataframe(data)
        >>> print(preprocessed_data.head())

    Notes:
        - The function renames columns, extracts numeric values, and maps boolean values.
        - It also processes additional columns like 'flood_zone_type' and 'connection_to_sewer_network'.
        - Specific columns such as 'cadastral_income' and 'price' undergo special processing.
        - Any errors encountered during processing will be printed with column details.
    """

    def extract_numbers(df: pd.DataFrame, columns: list):
        """
        Extracts numeric values from specified columns in the DataFrame.

        Args:
            df (pandas.DataFrame): The DataFrame to extract values from.
            columns (list): List of column names to extract numeric values from.

        Returns:
            pandas.DataFrame: The DataFrame with extracted numeric values.
        """
        for column in columns:
            try:
                df[column] = df[column].str.extract(r"(\d+)").astype("float32")
            except Exception as e:
                print(f"Error processing column {column}: {e}")
        return df

    def map_values(df: pd.DataFrame, columns: list):
        """
        Maps boolean values in specified columns to True, False, or None.

        Args:
            df (pandas.DataFrame): The DataFrame to map values in.
            columns (list): List of column names with boolean values to be mapped.

        Returns:
            pandas.DataFrame: The DataFrame with mapped boolean values.
        """
        for column in columns:
            try:
                df[column] = df[column].map({"Yes": 1, None: np.nan, "No": 0})
            except Exception as e:
                print(f"Error processing column {column}: {e}")
        return df

    number_columns = [
        "construction_year",
        "street_frontage_width",
        "number_of_frontages",
        "covered_parking_spaces",
        "outdoor_parking_spaces",
        "living_area",
        "living_room_surface",
        "kitchen_surface",
        "bedrooms",
        "bedroom_1_surface",
        "bedroom_2_surface",
        "bedroom_3_surface",
        "bathrooms",
        "toilets",
        "surface_of_the_plot",
        "width_of_the_lot_on_the_street",
        "garden_surface",
        "primary_energy_consumption",
        "co2_emission",
        "yearly_theoretical_total_energy_consumption",
    ]

    boolean_columns = [
        "basement",
        "furnished",
        "gas_water__electricity",
        "double_glazing",
        "planning_permission_obtained",
        "tv_cable",
        "dining_room",
        "proceedings_for_breach_of_planning_regulations",
        "subdivision_permit",
        "tenement_building",
        "possible_priority_purchase_right",
        "office",
    ]

    return (
        df.sort_index(axis=1)
        .fillna(np.nan)
        .rename(
            columns=lambda column: column.lower()
            .replace(" ", "_")
            .replace("&", "")
            .replace(",", "")
        )
        .rename(columns={"co₂_emission": "co2_emission"})
        .pipe(lambda df: extract_numbers(df, number_columns))
        .pipe(lambda df: map_values(df, boolean_columns))
        .assign(
            flood_zone_type=lambda df: df.flood_zone_type.map(
                {
                    "Non flood zone": 0,
                    "No": 0,
                    "Possible flood zone": 1,
                }
            ),
            connection_to_sewer_network=lambda df: df.connection_to_sewer_network.map(
                {
                    "Connected": 1,
                    "Not connected": 0,
                }
            ),
            as_built_plan=lambda df: df.as_built_plan.map(
                {
                    "Yes, conform": 1,
                    "No": 0,
                }
            ),
            cadastral_income=lambda df: df.cadastral_income.str.split(" ", expand=True)[
                3
            ].astype("float32"),
            price=lambda df: df.price.str.rsplit(" ", expand=True, n=2)[1].astype(
                float
            ),
        )
    )


def separate_address(df: pd.DataFrame) -> pd.DataFrame:
    """Separates the address into city, street name, house number, and zip code.

    Args:
        df (pd.DataFrame): The DataFrame containing the address column.

    Returns:
        pd.DataFrame: The DataFrame with the address separated into different columns.
    """
    # Define a regular expression pattern to extract street, house number, and zip code
    pattern = r"(?P<street_name>.*?)\s*(?P<house_number>\d+\w*)?\s*(?P<zip>\d{4})"

    try:
        return df.assign(
            city=lambda df: df.address.str.rsplit("-", expand=True, n=1)[1].str.title(),
            **(lambda dfx: dfx.rename(columns={"address": "original_address"}))(
                df["address"].str.extract(pattern)
            ),
            street=lambda df: df.street_name.str.replace(
                r"[^a-zA-Z\s]", "", regex=True
            ),
        ).drop(columns=["street_name", "address"])
    except Exception as e:
        print(f"Error separating address: {e}")
        return df


def get_location_details(
    location: str, geolocator: Nominatim
) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    """
    Get the address, latitude, and longitude for a given location.

    Parameters:
    location (str): The location to get details for.
    geolocator (Nominatim): The geolocator object to use for geocoding.

    Returns:
    tuple: A tuple containing the address (str), latitude (float), and longitude (float) of the location.
           If the geocoder cannot find the location, all elements of the tuple will be None.
    """
    try:
        location = geolocator.geocode(location, language="en")
        return location.address, location.latitude, location.longitude
    except AttributeError:
        print(f"Unable to get details for location: {location}")
        return None, None, None


def get_location_details_from_google(
    location: str, key: str = os.environ["GOOGLE_MAPS_API_KEY"]
) -> Tuple[
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[str],
    Optional[Union[float, None]],
    Optional[Union[float, None]],
]:
    """
    This function uses the geocoder library to get details of a location using
    Google's Geocoding API.

    Args:
        location (str): The location for which details are to be fetched.
        key (str, optional): The API key to be used for the request. Defaults
                             to os.environ['GOOGLE_MAPS_API_KEY'].

    Returns:
        tuple: A tuple containing the following details about the location (in
               order):
            - House number (str or None)
            - Street (str or None)
            - City (str or None)
            - Postal code (str or None)
            - State (str or None)
            - Latitude (float or None)
            - Longitude (float or None)

    Raises:
        AttributeError: If unable to get details for the location.
    """
    try:
        location = geocoder.google(location, key=key)
        return (
            location.housenumber,
            location.street,
            location.city,
            location.postal,
            location.state,
            location.lat,
            location.lng,
        )
    except AttributeError:
        print(f"Unable to get details for location: {location}")
        return None, None, None, None, None, None


def map_addresses(df, addresses, latitudes, longitudes, column_name="address"):
    """
    Maps the addresses, latitudes, and longitudes to a specific column in a DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to map the addresses, latitudes, and longitudes to.
    addresses (list): The list of addresses to map.
    latitudes (list): The list of latitudes to map.
    longitudes (list): The list of longitudes to map.
    column_name (str): The name of the column in df to map the addresses, latitudes, and longitudes to. Defaults to "address".

    Returns:
    pd.DataFrame: The DataFrame with the mapped addresses, latitudes, and longitudes.
    """
    unique_addresses = pd.Series(df[column_name].unique()).to_list()
    return df.assign(
        full_address=lambda df: df[column_name].map(
            dict(zip(unique_addresses, addresses))
        ),
        latitude=lambda df: df[column_name].map(dict(zip(unique_addresses, latitudes))),
        longitude=lambda df: df[column_name].map(
            dict(zip(unique_addresses, longitudes))
        ),
    ).drop(columns=column_name)


def map_addresses_from_google(
    df: pd.DataFrame,
    housenumber: List[Optional[str]],
    street: List[Optional[str]],
    city: List[Optional[str]],
    postal: List[Optional[str]],
    state: List[Optional[str]],
    lat: List[Optional[float]],
    lng: List[Optional[float]],
    column_name: str = "address",
) -> pd.DataFrame:
    """
    This function maps the details of unique addresses to new columns in the
    dataframe and drops the original address column.

    Args:
        df (pd.DataFrame): The dataframe containing the addresses.
        housenumber (List[Optional[str]]): The list of house numbers.
        street (List[Optional[str]]): The list of streets.
        city (List[Optional[str]]): The list of cities.
        postal (List[Optional[str]]): The list of postal codes.
        state (List[Optional[str]]): The list of states.
        lat (List[Optional[float]]): The list of latitudes.
        lng (List[Optional[float]]): The list of longitudes.
        column_name (str, optional): The name of the address column in the
                                     dataframe. Defaults to "address".

    Returns:
        pd.DataFrame: The dataframe with new columns for each detail and the
                      original address column dropped.
    """
    unique_addresses = pd.Series(df[column_name].unique()).to_list()
    address_dict = dict(
        zip(unique_addresses, zip(housenumber, street, city, postal, state, lat, lng))
    )

    df[
        ["housenumber", "street", "city", "postal", "state", "lat", "lng"]
    ] = pd.DataFrame(df[column_name].map(address_dict).tolist(), index=df.index)

    return df.drop(columns=column_name)


def filter_out_missing_indexes(
    df: pd.DataFrame,
    filepath: Path = utils.Configuration.INTERIM_DATA_PATH.joinpath(
        f"{str(pd.Timestamp.now())[:10]}_Processed_dataset.parquet.gzip"
    ),
) -> pd.DataFrame:
    """
    Filter out rows with missing values in a DataFrame and save the processed dataset.

    This function filters out rows with all missing values (NaN) and retains only rows
    with non-missing values in the 'price' column. The resulting DataFrame is then saved
    in Parquet format with gzip compression.

    Args:
        df (pd.DataFrame): The input DataFrame.
        filepath (Path, optional): The path to save the processed dataset in Parquet format.
            Defaults to a timestamp-based filepath in the interim data directory.

    Returns:
        pd.DataFrame: The filtered DataFrame with missing rows removed.

    Example:
        To filter out missing rows and save the processed dataset:
        >>> data = pd.read_csv("raw_data.csv")
        >>> filtered_data = filter_out_missing_indexes(data)
        >>> print(filtered_data.head())

    Notes:
        - Rows with missing values in any column other than 'price' are removed.
        - The processed dataset is saved with gzip compression to conserve disk space.
    """
    processed_df = df.dropna(axis=0, how="all").query("price.notna()")
    processed_df.to_parquet(filepath, compression="gzip", index=False)
    return processed_df


def prepare_data_for_modelling(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare data for machine learning modeling.

    This function takes a DataFrame and prepares it for machine learning by performing the following steps:
    1. Randomly shuffles the rows of the DataFrame.
    2. Converts the 'price' column to the base 10 logarithm.
    3. Fills missing values in categorical variables with 'missing value'.
    4. Separates the features (X) and the target (y).

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
    """

    processed_df = (
        df.sample(frac=1, random_state=utils.Configuration.seed)
        .reset_index(drop=True)
        .assign(price=lambda df: np.log10(df.price))
    )

    # Fill missing categorical variables with "missing value"
    for col in processed_df.columns:
        if processed_df[col].dtype.name in ("bool", "object", "category"):
            processed_df[col] = processed_df[col].fillna("missing value")

    # Separate features (X) and target (y)
    X = processed_df.loc[:, utils.Configuration.features_to_keep]
    y = processed_df[utils.Configuration.target_col]

    print(f"Shape of X and y: {X.shape}, {y.shape}")

    return X, y


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
    clf = neighbors.LocalOutlierFactor(n_neighbors=20, contamination=0.1)

    # Fit LOF to preprocessed data and make predictions
    y_pred = clf.fit_predict(preprocessor.fit_transform(df))

    # Adjust LOF predictions to create a binary outlier mask
    y_pred_adjusted = [1 if x == -1 else 0 for x in y_pred]
    outlier_mask = pd.Series(y_pred_adjusted) == 0

    return outlier_mask
