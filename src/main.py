import sys

import pandas as pd
from requests_html import HTMLSession

from data import make_dataset, pre_process, utils

# https://github.com/psf/requests-html/issues/275#issuecomment-513992564
session = HTMLSession(
    browser_args=[
        "--no-sandbox",
        "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
    ]
)


def main():
    try:
        # Use the get_last_page_number_from_url function to retrieve the last page number
        # last_page_number = make_dataset.get_last_page_number_from_url()
        # Create an instance of the ImmowebScraper class
        scraper = make_dataset.ImmowebScraper(session, last_page=3)
        # Run the data scraping and processing pipeline
        scraped_dataset = scraper.immoweb_scraping_pipeline()

        print(f"Web scraping completed. Raw data saved at {scraper.path}")

        df_pre_processed = pre_process.pre_process_dataframe(scraped_dataset)
        df_address_separated = pre_process.separate_address(df_pre_processed)
        pre_process.filter_out_missing_indexes(df_address_separated)

        print(
            f"Pre-processing completed. Raw data saved at {utils.Configuration.INTERIM_DATA_PATH}"
        )
    except Exception as e:
        # Handle exceptions here, and exit with a non-zero exit code
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
