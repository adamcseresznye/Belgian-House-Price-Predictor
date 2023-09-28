import logging
import re
import sys
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd

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
