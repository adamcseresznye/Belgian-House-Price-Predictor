import gc
from typing import List, Optional, Tuple

import catboost
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, pipeline
from tqdm import tqdm

from data import pre_process, utils


def run_catboost_CV_old_version(
    train: pd.DataFrame, pipeline: pipeline.Pipeline = None
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Run CatBoostRegressor for cross-validation on a given dataset.

    This function performs cross-validation using CatBoostRegressor. It selects categorical columns, trains a CatBoost
    model on each fold, calculates out-of-fold (OOF) predictions, and returns a DataFrame with OOF predictions and
    validation scores for each fold.

    Args:
        train (pd.DataFrame): The training dataset containing both features and the target variable.
        pipeline (Optional[Pipeline]): An optional data preprocessing pipeline.
            If provided, it is used to transform the features before training.

    Returns:
        Tuple[pd.DataFrame, List[float]]: A tuple containing:
        - oof_df (pd.DataFrame): A DataFrame with columns 'target_col', 'prediction', and 'fold' containing OOF predictions.
        - val_scores (List[float]): A list of validation scores for each fold.

    Example:
        >>> import pandas as pd
        >>> from models import train_model
        >>> from sklearn.pipeline import Pipeline
        >>> from catboost import CatBoostRegressor
        >>> train_data = pd.read_csv("train.csv")  # Load your training data
        >>> preprocessor = Pipeline(...)  # Define your preprocessing pipeline
        >>> oof_df, val_scores = train_model.run_catboost(train_data, pipeline=preprocessor)
        >>> print(oof_df.head())  # Display the OOF predictions and folds
        >>> print(val_scores)  # Display the validation scores for each fold
    """
    # Create a numpy array to store out-of-folds predictions
    oof_predictions = np.zeros(len(train))
    oof_fold = np.zeros(len(train))
    val_scores = np.zeros(len(set(train.folds.unique())))

    for fold in tqdm(set(train.folds.unique())):
        # Identify features
        features = train.columns[~train.columns.str.contains("price|folds")]

        numerical_features = train.select_dtypes("number").columns.to_list()
        categorical_features = train.select_dtypes("object").columns.to_list()

        assert len(numerical_features) + len(categorical_features) == train.shape[1]

        # Split folds
        train_folds = train[train.folds != fold]
        val_fold = train[train.folds == fold]

        # Get target variables
        tr_y = train_folds[utils.Configuration.target_col]
        val_y = val_fold[utils.Configuration.target_col]

        # Get feature matrices
        tr_X = train_folds.loc[:, features]
        val_X = val_fold.loc[:, features]

        # Apply optional data preprocessing pipeline
        if pipeline is not None:
            tr_X = pipeline.fit_transform(tr_X)
            val_X = pipeline.transform(val_X)

        # Create CatBoost datasets
        catboost_train = catboost.Pool(
            tr_X,
            tr_y,
            cat_features=categorical_features,
        )
        catboost_valid = catboost.Pool(
            val_X,
            val_y,
            cat_features=categorical_features,
        )

        # Initialize and train the CatBoost model
        model = catboost.CatBoostRegressor(**utils.Configuration.catboost_params)
        model.fit(
            catboost_train,
            eval_set=[catboost_valid],
            early_stopping_rounds=utils.Configuration.early_stopping_round,
            verbose=utils.Configuration.verbose,
            use_best_model=True,
        )

        # Calculate OOF validation predictions
        valid_pred = model.predict(val_X)

        # Append values to the OOF arrays
        oof_predictions[val_fold.index] = valid_pred
        oof_fold[val_fold.index] = fold

        # Calculate OOF scores
        val_scores[fold - 1] = metrics.mean_squared_error(
            val_y, valid_pred, squared=False
        )

        del tr_X, val_X, tr_y, val_y, valid_pred, model
        gc.collect()

    # Create a DataFrame with OOF predictions and fold labels
    oof_df = pd.DataFrame(
        {
            utils.Configuration.target_col: train[utils.Configuration.target_col],
            "prediction": oof_predictions,
            "fold": oof_fold.astype(int),
        }
    )

    return oof_df, val_scores


def run_catboost_CV(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 10,
    n_repeats: int = 1,
    pipeline: Optional[object] = None,
) -> Tuple[float, float]:
    """
    Perform Cross-Validation with CatBoost for regression.

    This function conducts Cross-Validation using CatBoost for regression tasks. It iterates
    through folds, trains CatBoost models, and computes the mean and standard deviation of the
    Root Mean Squared Error (RMSE) scores across folds.

    Parameters:
    - X (pd.DataFrame): The feature matrix.
    - y (pd.Series): The target variable.
    - n_splits (int, optional): The number of splits in K-Fold cross-validation.
      Defaults to 10.
    - n_repeats (int, optional): The number of times the K-Fold cross-validation is repeated.
      Defaults to 1.
    - pipeline (object, optional): Optional data preprocessing pipeline. If provided,
      it's applied to the data before training the model. Defaults to None.

    Returns:
    - Tuple[float, float]: A tuple containing the mean RMSE and standard deviation of RMSE
      scores across cross-validation folds.

    Example:
    ```python
    # Load your feature matrix (X) and target variable (y)
    X, y = load_data()

    # Perform Cross-Validation with CatBoost
    mean_rmse, std_rmse = run_catboost_CV(X, y, n_splits=5, n_repeats=2, pipeline=data_pipeline)

    print(f"Mean RMSE: {mean_rmse:.4f}")
    print(f"Standard Deviation of RMSE: {std_rmse:.4f}")
    ```

    Notes:
    - Ensure that the input data `X` and `y` are properly preprocessed and do not contain any
      missing values.
    - The function uses CatBoost for regression with optional data preprocessing via the `pipeline`.
    - RMSE is a common metric for regression tasks, and lower values indicate better model
      performance.
    """
    results = []

    # Extract feature names and data types
    features = X.columns[~X.columns.str.contains("price")]
    numerical_features = X.select_dtypes("number").columns.to_list()
    categorical_features = X.select_dtypes("object").columns.to_list()

    # Create a K-Fold cross-validator
    CV = model_selection.RepeatedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=utils.Configuration.seed
    )

    for train_fold_index, val_fold_index in tqdm(CV.split(X)):
        X_train_fold, X_val_fold = X.loc[train_fold_index], X.loc[val_fold_index]
        y_train_fold, y_val_fold = y.loc[train_fold_index], y.loc[val_fold_index]

        # Apply optional data preprocessing pipeline
        if pipeline is not None:
            X_train_fold = pipeline.fit_transform(X_train_fold)
            X_val_fold = pipeline.transform(X_val_fold)

        # Create CatBoost datasets
        catboost_train = catboost.Pool(
            X_train_fold,
            y_train_fold,
            cat_features=categorical_features,
        )
        catboost_valid = catboost.Pool(
            X_val_fold,
            y_val_fold,
            cat_features=categorical_features,
        )

        # Initialize and train the CatBoost model
        model = catboost.CatBoostRegressor(**utils.Configuration.catboost_params)
        model.fit(
            catboost_train,
            eval_set=[catboost_valid],
            early_stopping_rounds=utils.Configuration.early_stopping_round,
            verbose=utils.Configuration.verbose,
            use_best_model=True,
        )

        # Calculate OOF validation predictions
        valid_pred = model.predict(X_val_fold)

        RMSE_score = metrics.mean_squared_error(y_val_fold, valid_pred, squared=False)

        del (
            X_train_fold,
            y_train_fold,
            X_val_fold,
            y_val_fold,
            catboost_train,
            catboost_valid,
            model,
            valid_pred,
        )
        gc.collect()

        results.append(RMSE_score)

    return np.mean(results), np.std(results)
