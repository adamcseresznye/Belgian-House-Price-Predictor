import os
import pickle
import sys
from pathlib import Path

import optuna
import pandas as pd
from requests_html import HTMLSession
from sklearn import model_selection
from tqdm import tqdm

from data import make_dataset, pre_process, utils
from models import train_model

# https://github.com/psf/requests-html/issues/275#issuecomment-513992564
session = HTMLSession(
    browser_args=[
        "--no-sandbox",
        "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
    ]
)

try:
    GOOGLE_MAPS_API_KEY = os.environ["GOOGLE_MAPS_API_KEY"]
except:
    raise Exception("Set environment variable GOOGLE_MAPS_API_KEY")


def main():
    try:
        # Use the get_last_page_number_from_url function to retrieve the last page number
        last_page_number = make_dataset.get_last_page_number_from_url(session=session)
        # Create an instance of the ImmowebScraper class
        scraper = make_dataset.ImmowebScraper(session, last_page=3)
        # Run the data scraping and processing pipeline
        scraped_dataset = scraper.immoweb_scraping_pipeline()
        # perform basic pre-processing such as map booleans, convert floats and such
        df_pre_processed = pre_process.pre_process_dataframe(scraped_dataset)
        # Fetch the geo data using Google's Geocoding API
        housenumber, street, city, postal, state, lat, lng = zip(
            *[
                pre_process.get_location_details_from_google(loc)
                for loc in tqdm(df_pre_processed.address.unique(), position=0)
            ]
        )
        # map the obtained addresses
        mapped_df = pre_process.map_addresses_from_google(
            df_pre_processed,
            housenumber,
            street,
            city,
            postal,
            state,
            lat,
            lng,
        )
        # filter out missing values and save dataframe for streamlit app
        pre_processed_df = pre_process.filter_out_missing_indexes(mapped_df)

        # prepare data for modelling
        X, y = pre_process.prepare_data_for_modelling(pre_processed_df)

        # train-test split
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=utils.Configuration.seed,
        )
        print(f"Shape of X_train: {X_train.shape}, Shape of X_test: {X_test.shape}")

        # train model
        catboost_params_optuna = pd.read_pickle(
            utils.Configuration.GIT_HYPERPARAMETERS.joinpath("CatBoost_params.pickle")
        )
        # train model
        print("Model training has started.")
        model = train_model.train_catboost(X_train, y_train, catboost_params_optuna)

        # save model
        model.save_model(utils.Configuration.GIT_MODEL.joinpath("catboost_model"))
        print(
            f"The workflow has completed successfully, and the model has been saved at:{utils.Configuration.GIT_MODEL}"
        )

    except Exception as e:
        # Handle exceptions here, and exit with a non-zero exit code
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
