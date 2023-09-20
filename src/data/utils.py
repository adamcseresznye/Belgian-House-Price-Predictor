import os
import random
from pathlib import Path

import numpy as np


class Configuration:
    VER = 1
    RAW_DATA_PATH = Path.cwd().parents[1].joinpath("data/raw")
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
    seed = 3407
    n_folds = 10
    target_col = "Class"
    verbose = 0


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
