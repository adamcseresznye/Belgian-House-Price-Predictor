import os
import random
from pathlib import Path

import numpy as np


class Configuration:
    VER = 1
    RAW_DATA_PATH = Path(__file__).parents[2].joinpath("data/raw")
    INTERIM_DATA_PATH = Path(__file__).parents[2].joinpath("data/interim")
    target_col = "price"

    features_to_keep_sales = [
        "day_of_retrieval",
        "ad_url",
        "Reference number of the EPC report",
        "Energy class",
        "Primary energy consumption",
        "Yearly theoretical total energy consumption",
        "CO₂ emission",
        "Tenement building",
        "Address",
        "Bedrooms",
        "Living area",
        "Bathrooms",
        "Surface of the plot",
        "Price",
        "Building condition",
        "Double glazing",
        "Number of frontages",
        "Website",
        "Toilets",
        "External reference",
        "Heating type",
        "Cadastral income",
        "Gas, water & electricity",
        "Latest land use designation",
        "Connection to sewer network",
        "Covered parking spaces",
        "Possible priority purchase right",
        "Proceedings for breach of planning regulations",
        "Construction year",
        "Subdivision permit",
        "Bedroom 1 surface",
        "Bedroom 2 surface",
        "Available as of",
        "Kitchen type",
        "Flood zone type",
        "Living room surface",
        "Planning permission obtained",
        "Kitchen surface",
        "TV cable",
        "Bedroom 3 surface",
        "Furnished",
        "Outdoor parking spaces",
        "Surroundings type",
        "Garden surface",
        "Basement",
        "Width of the lot on the street",
        "Street frontage width",
        "Office",
        "As built plan",
        "Dining room",
    ]
    features_to_drop = [
        "external_reference",
        "ad_url",
        "day_of_retrieval",
        "website",
        "reference_number_of_the_epc_report",
        "housenumber",
        # "bins",
    ]
    features_to_keep = [
        "bedrooms",
        "state",
        "kitchen_type",
        "number_of_frontages",
        "toilets",
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
    seed = 3407
    n_folds = 10
    verbose = 0
    early_stopping_round = 20

    catboost_params = {
        #'iterations': 342,
        #'depth': 3,
        #'learning_rate': 0.3779980855781628,
        #'random_strength': 1.5478223057973914,
        #'bagging_temperature': 0.689173368569372,
        #'l2_leaf_reg': 16,
        #'border_count': 37,
        "thread_count": os.cpu_count(),
        "loss_function": "RMSE",
        "iterations": 100,
        "learning_rate": 0.2,
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
