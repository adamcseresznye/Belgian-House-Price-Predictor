import os
import sys
from pathlib import Path

import pandas as pd
from requests_html import HTMLSession
from tqdm import tqdm

from data import make_dataset, pre_process, utils

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

        print(f"Web scraping completed. Raw data saved at {scraper.path}")

        df_pre_processed = pre_process.pre_process_dataframe(scraped_dataset)

        housenumber, street, city, postal, state, lat, lng = zip(
            *[
                pre_process.get_location_details_from_google(loc)
                for loc in tqdm(df_pre_processed.address.unique(), position=0)
            ]
        )
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

        pre_process.filter_out_missing_indexes(mapped_df)
        print(
            f"Pre-processing completed. Raw data saved at {utils.Configuration.INTERIM_DATA_PATH}"
        )
    except Exception as e:
        # Handle exceptions here, and exit with a non-zero exit code
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
