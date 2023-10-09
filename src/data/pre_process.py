import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import geocoder
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn import model_selection
from tqdm import tqdm

import creds
from data import utils

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
    location: str, key: str = creds.api_key
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
                             to creds.api_key.

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
    Preprocesses a DataFrame for machine learning modeling and adds 'kfold' column for cross-validation.

    This function takes a DataFrame containing the dataset to be processed and performs the following steps:
    1. Adds a 'kfold' column to the DataFrame for cross-validation.
    2. Randomly samples and shuffles the dataset.
    3. Applies a log transformation to the 'price' column in the DataFrame.
    4. Drops specified columns from the DataFrame. They can be found in utils.Configuration.features_to_drop
    5. Handles missing values in boolean, object, and category columns by filling them with 'missing value'.

    Args:
        df (pd.DataFrame): The input dataset to create folds for.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the feature matrix (X) and the target variable (y).

    Example:
        >>> import pandas as pd
        >>> from my_module import prepare_data_for_modelling
        >>> data = pd.read_csv("my_data.csv")  # Load your dataset
        >>> X, y = prepare_data_for_modelling(data)
        >>> print(X.head())  # Display the first few rows of the feature matrix
        >>> print(y.head())  # Display the first few rows of the target variable
    """
    # Add 'folds' column initialized with -1 for all rows
    df["folds"] = -1

    # Shuffle the dataset by sampling and resetting the index
    df = df.sample(frac=1, random_state=utils.Configuration.seed).reset_index(drop=True)

    # Use K-Fold with 10 splits to assign fold numbers
    kf = model_selection.KFold(n_splits=10)
    for fold, (t_, v_) in enumerate(kf.split(X=df)):
        df.loc[v_, "folds"] = fold

    # Perform data preprocessing
    processed_df = (
        df.reset_index(drop=True)
        .assign(
            price=lambda df: np.log10(df.price)
        )  # Log transformation of 'price' column
        .drop(columns=utils.Configuration.features_to_drop)
    )

    # Handle missing values in boolean, object, and category columns
    # https://www.kdnuggets.com/2023/02/top-5-advantages-catboost-ml-brings-data-make-purr.html
    for col in processed_df.columns:
        if processed_df[col].dtype.name in ("bool", "object", "category"):
            processed_df[col] = processed_df[col].fillna("missing value")

    # Separate features (X) and target (y)
    X = processed_df.drop(columns=utils.Configuration.target_col)
    y = processed_df[utils.Configuration.target_col]

    print(f"Shape of X and y: {X.shape}, {y.shape}")

    # return X, y
    return processed_df
