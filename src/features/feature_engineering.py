import gc
from typing import List, Optional, Tuple

import catboost
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, pipeline
from data import pre_process, utils

import gc
from pathlib import Path
from typing import List, Optional, Tuple

import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from IPython.display import clear_output
from lets_plot import *
from lets_plot.mapping import as_discrete
from sklearn import (
    compose,
    ensemble,
    impute,
    metrics,
    model_selection,
    neighbors,
    pipeline,
    preprocessing,
)
from tqdm.notebook import tqdm

from data import pre_process, utils
from models import train_model


def FE_categorical_transform(
    X: pd.DataFrame, y: pd.Series, transform_type: str = "mean"
) -> pd.DataFrame:
    """
    Feature Engineering: Transform categorical features using CatBoost Cross-Validation.

    This function performs feature engineering by transforming categorical features using CatBoost
    Cross-Validation. It calculates the mean and standard deviation of Out-Of-Fold (OOF) Root Mean
    Squared Error (RMSE) scores for various combinations of categorical and numerical features.

    Parameters:
    - X (pd.DataFrame): The input DataFrame containing both categorical and numerical features.
    - transform_type (str, optional): The transformation type, such as "mean" or other valid
      CatBoost transformations. Defaults to "mean".

    Returns:
    - pd.DataFrame: A DataFrame with columns "mean_OOFs," "std_OOFs," "categorical," and "numerical,"
      sorted by "mean_OOFs" in ascending order.

    Example:
    ```python
    # Load your DataFrame with features (X)
    X = load_data()

    # Perform feature engineering
    result_df = FE_categorical_transform(X, transform_type="mean")

    # View the DataFrame with sorted results
    print(result_df.head())
    ```

    Notes:
    - This function uses CatBoost Cross-Validation to assess the quality of transformations for
      various combinations of categorical and numerical features.
    - The resulting DataFrame provides insights into the effectiveness of different transformations.
    - Feature engineering can help improve the performance of machine learning models.
    """
    # Initialize a list to store results
    results = []

    # Get a list of categorical and numerical columns
    categorical_columns = X.select_dtypes("object").columns
    numerical_columns = X.select_dtypes("number").columns

    # Combine the loops to have a single progress bar
    for categorical in tqdm(categorical_columns, desc="Progress"):
        for numerical in tqdm(numerical_columns):
            # Create a deep copy of the input data
            temp = X.copy(deep=True)

            # Calculate the transformation for each group within the categorical column
            temp["new_column"] = temp.groupby(categorical)[numerical].transform(
                transform_type
            )

            # Run CatBoost Cross-Validation with the transformed data
            mean_OOF, std_OOF = train_model.run_catboost_CV(temp, y)

            # Store the results as a tuple
            result = (mean_OOF, std_OOF, categorical, numerical)
            results.append(result)

            del temp, mean_OOF, std_OOF

    # Create a DataFrame from the results and sort it by mean OOF scores
    result_df = pd.DataFrame(
        results, columns=["mean_OOFs", "std_OOFs", "categorical", "numerical"]
    )
    result_df = result_df.sort_values(by="mean_OOFs")
    return result_df