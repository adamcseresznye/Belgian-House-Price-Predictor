import logging
import re
import sys
from typing import List, Set

import numpy as np
import pandas as pd

from data import utils

# Set up logging
logging.basicConfig(
    filename=utils.Configuration.RAW_DATA_PATH / "build_features_error.log",
    filemode="w",
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def pre_process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses a DataFrame by performing various data cleaning and transformation tasks.

    Args:
        df (pandas.DataFrame): The input DataFrame to be preprocessed.

    Returns:
        pandas.DataFrame: The preprocessed DataFrame.
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
                logging.error(f"Error processing column {column}: {e}")
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
                df[column] = df[column].map({"Yes": True, None: False, "No": False})
            except Exception as e:
                logging.error(f"Error processing column {column}: {e}")
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
                    "Non flood zone": False,
                    "No": False,
                    "Possible flood zone": True,
                }
            ),
            connection_to_sewer_network=lambda df: df.connection_to_sewer_network.map(
                {
                    "Connected": True,
                    "Not connected": False,
                }
            ),
            as_built_plan=lambda df: df.as_built_plan.map(
                {
                    "Yes, conform": True,
                    "No": False,
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
            city=lambda df: df.address.str.rsplit("-", expand=True, n=1)[1],
            **(lambda dfx: dfx.rename(columns={"address": "original_address"}))(
                df["address"].str.extract(pattern)
            ),
            street=lambda df: df.street_name.str.replace(
                r"[^a-zA-Z\s]", "", regex=True
            ),
        ).drop(columns=["street_name", "address"])
    except Exception as e:
        logging.error(f"Error separating address: {e}")
        return df
