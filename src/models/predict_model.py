import gc
import os
from pathlib import Path
from typing import List, Optional, Tuple

import catboost
import numpy as np
import optuna
import pandas as pd
from sklearn import metrics, model_selection, pipeline
from tqdm import tqdm

from data import pre_process, utils


def predict_catboost(
    model: catboost.CatBoostRegressor,
    X: pd.DataFrame,
    thread_count: int = -1,
    verbose: int = None,
) -> np.ndarray:
    """
    Make predictions using a CatBoost model on the provided dataset.

    Parameters:
        model (catboost.CatBoostRegressor): The trained CatBoost model.
        X (pd.DataFrame): The dataset for which predictions are to be made.
        thread_count (int, optional): The number of threads for prediction. Default is -1 (auto).
        verbose (int, optional): Verbosity level. Default is None.

    Returns:
        np.ndarray: Predicted values.

    This function takes a trained CatBoost model, a dataset `X`, and optional parameters for
    specifying the number of threads (`thread_count`) and verbosity (`verbose`) during prediction.
    It returns an array of predicted values.

    Example:
        model = load_catboost_model()
        X_new = load_new_data()
        predictions = predict_catboost(model, X_new, thread_count=4, verbose=2)
    """
    prediction = model.predict(data=X, thread_count=thread_count, verbose=verbose)
    return prediction


def score_prediction(y_true, y_pred):
    """
    Calculate regression evaluation metrics based on
    true and predicted values.

    Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.

    Returns:
        tuple: A tuple containing Root Mean Squared Error (RMSE)
        and R-squared (R2).

    This function calculates RMSE and R2 to evaluate the goodness
    of fit between the true target values and predicted values.

    Example:
        y_true = [3, 5, 7, 9]
        y_pred = [2.8, 5.2, 7.1, 9.3]
        rmse, r2 = score_prediction(y_true, y_pred)
    """
    RMSE = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    R2 = metrics.r2_score(y_true, y_pred)

    return RMSE, R2


def save_model_performance(
    RMSE: float,
    R2: float,
    path_to_folder: Path = utils.Configuration.GIT_MODEL_PERFORMANCE,
) -> pd.DataFrame:
    """
    Save model performance metrics (RMSE and R2) to a Parquet file.

    Args:
        RMSE (float): Root Mean Square Error metric to be saved.
        R2 (float): R-squared metric to be saved.
        path_to_folder (Path, optional): The path to the folder where the Parquet file is stored.
            Defaults to utils.Configuration.GIT_MODEL_PERFORMANCE.

    Returns:
        pd.DataFrame: A DataFrame containing all saved model performance records.

    If the Parquet file exists, it loads the existing records, appends a new record with the current date,
    RMSE, and R2, and saves the updated records back to the file. If the file does not exist, it creates a new record and
    saves it as the Parquet file.

    Note:
    - The Parquet file should have a structure where "date," "RMSE," and "R2" are columns.
    - Date is recorded as a string in the format 'yyyy-mm-dd'.
    """
    # Define the path to the Parquet file
    all_records = path_to_folder.joinpath("model_performance.parquet.gzip")

    if all_records.is_file():
        # If the Parquet file already exists, load its contents
        all_records_df = pd.read_parquet(all_records)

        # Create a new record with the current date and the provided RMSE and R2 values
        new_record = pd.DataFrame(
            {"date": str(pd.Timestamp.now())[:10], "RMSE": RMSE, "R2": R2}, index=[0]
        )

        # Concatenate the new record with the existing records
        updated_records = pd.concat([all_records_df, new_record], ignore_index=True)

        # Save the updated records to the Parquet file
        updated_records.to_parquet(all_records, compression="gzip")

        return updated_records
    else:
        # If the Parquet file doesn't exist, create a new record and save it
        new_record = pd.DataFrame(
            {"date": str(pd.Timestamp.now())[:10], "RMSE": RMSE, "R2": R2}, index=[0]
        )

        new_record.to_parquet(all_records, compression="gzip")

        return new_record


def save_prediction_plots(truth, prediction):
    """
    Generate and save two plots related to model predictions and residuals.

    Args:
        truth (pandas.Series): Actual truth values.
        prediction (pandas.Series): Model predictions.

    Returns:
        None

    The function creates two plots and saves them as SVG files:
    1. A scatter plot contrasting predicted house prices with actual house prices.
    2. A histogram assessing the residuals from the Catboost model.

    Note:
    - The saved files will be named "fig1.svg" and "fig2.svg" in the current directory.
    """
    results = (
        pd.concat(
            [truth.reset_index(drop=True), pd.Series(prediction)], axis="columns"
        ).rename(columns={"price": "original_values", 0: "predicted_values"})
        # .apply(lambda x: 10**x)
        .assign(residuals=lambda df: df.original_values - df.predicted_values)
    )
    fig1 = results.pipe(
        lambda df: ggplot(df, aes("original_values", "predicted_values"))
        + geom_point()
        + geom_smooth()
        + labs(
            title="Contrasting Predicted House Prices with Actual House Prices",
            x="log10 True Prices (EUR)",
            y="log10 Predicted Prices (EUR)",
            caption="https://www.immoweb.be/",
        )
        + theme(
            plot_subtitle=element_text(
                size=12, face="italic"
            ),  # Customize subtitle appearance
            plot_title=element_text(size=15, face="bold"),  # Customize title appearance
        )
        + ggsize(800, 600)
    )
    fig2 = (
        results.pipe(
            lambda df: ggplot(df, aes("residuals")) + geom_histogram(stat="bin")
        )
        + labs(
            title="Assessing the Residuals from the Catboost Model",
            subtitle=""" Normally distributed residuals imply consistent and accurate model predictions, aligning with statistical assumptions.
            """,
            x="Distribution of Residuals",
            y="",
            caption="https://www.immoweb.be/",
        )
        + theme(
            plot_subtitle=element_text(
                size=12, face="italic"
            ),  # Customize subtitle appearance
            plot_title=element_text(size=15, face="bold"),  # Customize title appearance
        )
        + ggsize(800, 600)
    )
    ggsave(fig1, "fig1.svg", path=".", iframe=False)
    ggsave(fig2, "fig2.svg", path=".", iframe=False)
    return None
